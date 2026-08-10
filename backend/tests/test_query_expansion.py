import pytest

from rag.query_expansion import MAX_VARIANT_CHARS, QueryExpander


class FakeLLMService:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_prompt: str | None = None
        self.calls = 0

    def generate_response(self, user_message: str) -> str:
        self.calls += 1
        self.last_prompt = user_message
        if self._error is not None:
            raise self._error
        return self._response


def test_each_line_becomes_a_variant():
    llm = FakeLLMService(
        "How do I cancel my subscription?\n"
        "Where do I end my plan?\n"
        "What is the process for stopping billing?"
    )

    variants = QueryExpander(llm, count=3).expand("how do I stop paying for this")

    assert variants == [
        "How do I cancel my subscription?",
        "Where do I end my plan?",
        "What is the process for stopping billing?",
    ]


def test_numbering_and_bullets_are_stripped():
    """Models add them back however firmly the prompt says not to."""
    llm = FakeLLMService("1. First phrasing?\n2) Second phrasing?\n- Third phrasing?")

    variants = QueryExpander(llm, count=3).expand("a question")

    assert variants == ["First phrasing?", "Second phrasing?", "Third phrasing?"]


def test_the_original_question_is_never_returned_as_a_variant():
    """The caller always retrieves for it; returning it would double its weight."""
    llm = FakeLLMService("How do I cancel?\nhow do i stop paying for this\nWhere do I end my plan?")

    variants = QueryExpander(llm, count=3).expand("How do I stop paying for this?")

    assert all("stop paying" not in v.lower() for v in variants)
    assert len(variants) == 2


def test_duplicate_variants_are_dropped():
    """Repetition is not agreement - fusion would count the same chunks twice."""
    llm = FakeLLMService("How do I cancel?\nHow do I cancel?\nHow do I cancel?")

    assert QueryExpander(llm, count=3).expand("a question") == ["How do I cancel?"]


def test_no_more_than_count_variants_are_returned():
    llm = FakeLLMService("\n".join(f"phrasing {i}?" for i in range(20)))

    assert len(QueryExpander(llm, count=3).expand("a question")) == 3


def test_blank_lines_are_ignored():
    llm = FakeLLMService("First?\n\n   \nSecond?\n")

    assert QueryExpander(llm, count=3).expand("a question") == ["First?", "Second?"]


def test_a_rambling_line_is_not_treated_as_a_query():
    """A model that ignores "no preamble" writes a paragraph that matches everything weakly."""
    llm = FakeLLMService("x" * (MAX_VARIANT_CHARS + 1) + "\nA real phrasing?")

    assert QueryExpander(llm, count=3).expand("a question") == ["A real phrasing?"]


def test_the_prompt_asks_for_the_configured_number():
    llm = FakeLLMService("a?\nb?")

    QueryExpander(llm, count=2).expand("a question")

    assert "2 alternative versions" in llm.last_prompt
    assert "a question" in llm.last_prompt


def test_the_prompt_tells_the_model_to_preserve_identifiers():
    """
    Retrieval here is hybrid: a variant that paraphrases ERR-4021 away has
    destroyed the signal the BM25 half needed.
    """
    llm = FakeLLMService("a?")

    QueryExpander(llm, count=1).expand("what causes ERR-4021")

    assert "ERR-4021" in llm.last_prompt
    assert "identifier" in llm.last_prompt.lower()


def test_an_llm_failure_degrades_to_no_variants_rather_than_raising():
    """
    The question can still be answered on the original phrasing, which is
    exactly the non-expanded behaviour - a known-good path, not a guess.
    """
    llm = FakeLLMService(error=RuntimeError("provider timed out"))

    assert QueryExpander(llm, count=3).expand("a question") == []


def test_an_empty_llm_response_yields_no_variants():
    assert QueryExpander(FakeLLMService(""), count=3).expand("a question") == []


def test_expand_rejects_an_empty_question():
    with pytest.raises(ValueError):
        QueryExpander(FakeLLMService("a?"), count=3).expand("   ")


def test_a_non_positive_count_is_rejected_at_construction():
    with pytest.raises(ValueError):
        QueryExpander(FakeLLMService(""), count=0)


def test_expansion_costs_exactly_one_llm_call():
    llm = FakeLLMService("a?\nb?\nc?")

    QueryExpander(llm, count=3).expand("a question")

    assert llm.calls == 1
