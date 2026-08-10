import pytest

from rag.contextualizer import MAX_REWRITE_CHARS, QueryContextualizer


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


HISTORY = [
    {"role": "user", "content": "how much annual leave do I build up each month?"},
    {"role": "assistant", "content": "Annual leave accrues at 2.08 days per month."},
]


def test_a_dependent_question_is_rewritten_to_stand_alone():
    llm = FakeLLMService("how far ahead do I have to request annual leave?")

    result = QueryContextualizer(llm).contextualize(
        "how far ahead do I have to request it?", HISTORY
    )

    assert result == "how far ahead do I have to request annual leave?"


def test_the_first_turn_of_a_conversation_costs_no_llm_call():
    """
    Nothing can refer to an earlier turn when there isn't one. The guard is
    exact rather than a heuristic, which is what keeps /rag/ask free.
    """
    llm = FakeLLMService("some rewrite")

    for history in (None, []):
        assert QueryContextualizer(llm).contextualize("a question", history) == "a question"

    assert llm.calls == 0


def test_the_conversation_is_put_in_front_of_the_model():
    llm = FakeLLMService("rewritten")

    QueryContextualizer(llm).contextualize("and what about it?", HISTORY)

    assert "annual leave" in llm.last_prompt
    assert "and what about it?" in llm.last_prompt


def test_the_prompt_forbids_paraphrasing_identifiers():
    """Same hazard as expansion: BM25 matches the literal token or nothing."""
    llm = FakeLLMService("rewritten")

    QueryContextualizer(llm).contextualize("and ERR-4021?", HISTORY)

    assert "identifier" in llm.last_prompt.lower()


def test_an_already_standalone_question_comes_back_unchanged():
    llm = FakeLLMService("what causes ERR-4021?")

    result = QueryContextualizer(llm).contextualize("what causes ERR-4021?", HISTORY)

    assert result == "what causes ERR-4021?"


def test_surrounding_quotes_are_stripped():
    """Models like to quote a rewrite; the quotes would become search terms."""
    llm = FakeLLMService('"how far ahead do I request annual leave?"')

    result = QueryContextualizer(llm).contextualize("how far ahead?", HISTORY)

    assert result == "how far ahead do I request annual leave?"


def test_a_preamble_line_is_skipped():
    llm = FakeLLMService("Here is the rewritten question:\nwhen is annual leave requested?")

    result = QueryContextualizer(llm).contextualize("when?", HISTORY)

    assert result == "when is annual leave requested?"


def test_an_llm_failure_degrades_to_the_original_question():
    llm = FakeLLMService(error=RuntimeError("provider down"))

    result = QueryContextualizer(llm).contextualize("and what about it?", HISTORY)

    assert result == "and what about it?"


def test_an_empty_rewrite_degrades_to_the_original_question():
    llm = FakeLLMService("   \n  ")

    assert QueryContextualizer(llm).contextualize("and it?", HISTORY) == "and it?"


def test_a_model_that_answers_instead_of_rewriting_is_ignored():
    """
    Searching for the model's own answer retrieves whatever that prose
    resembles, which is worse than searching for the original question.
    """
    llm = FakeLLMService("x" * (MAX_REWRITE_CHARS + 1))

    assert QueryContextualizer(llm).contextualize("and it?", HISTORY) == "and it?"


def test_contextualize_rejects_an_empty_question():
    with pytest.raises(ValueError):
        QueryContextualizer(FakeLLMService("x")).contextualize("  ", HISTORY)


def test_contextualizing_costs_exactly_one_llm_call():
    llm = FakeLLMService("rewritten")

    QueryContextualizer(llm).contextualize("and it?", HISTORY)

    assert llm.calls == 1
