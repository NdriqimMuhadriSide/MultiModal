"""
Chunking strategy tests.

Strategies are exercised with stand-in services rather than the real embedding
model or a live LLM: what is being tested is the *shape* each strategy produces
- how many chunks, what gets embedded versus returned, which ones share a
parent - and none of that depends on real vectors. The one place a real model
would matter, semantic chunking's sense of "distance", is faked with vectors
chosen to make the topic shift unambiguous.
"""
import pytest

from rag.chunking import ChunkBudget, ContextualEnrichment, build_strategy
from rag.chunking.blocks import prose_runs, segment_blocks
from rag.chunking.parent_document import ParentDocumentChunking
from rag.chunking.propositional import PropositionalChunking
from rag.chunking.recursive import RecursiveChunking
from rag.chunking.semantic import SemanticChunking
from rag.chunking.sentence_window import SentenceWindowChunking
from rag.layout import Block

BUDGET = ChunkBudget(size=200, overlap=40, measure=len)
# Wide enough to hold PROSE in a single passage, so the LLM-backed strategies
# are asked about it exactly once and the assertions can name what comes back.
WIDE = ChunkBudget(size=600, overlap=50, measure=len)

PROSE = (
    "The trial began in March. It was halted six weeks later after three sites "
    "reported calibration failures. Dr. Patel reviewed the 3.5 kg samples. "
    "Yields rose sharply in the second quarter. Revenue grew by three percent. "
)


def _table(rows: int) -> Block:
    body = "\n".join(f"| City{index} | {index} |" for index in range(rows))
    return Block(kind="table", text=f"| city | orders |\n| --- | --- |\n{body}")


def _prose(text: str = PROSE) -> Block:
    return Block(kind="text", text=text)


# --- Block-aware segmentation -----------------------------------------------


def test_a_table_that_fits_is_kept_whole():
    segments = segment_blocks([_table(rows=2)], BUDGET)

    assert len(segments) == 1
    assert segments[0].atomic is True


def test_an_oversized_table_is_split_by_row():
    segments = segment_blocks([_table(rows=40)], BUDGET)

    assert len(segments) > 1
    assert all(segment.atomic for segment in segments)
    assert all(len(segment.text) <= BUDGET.size for segment in segments)


def test_every_part_of_a_split_table_repeats_the_header():
    """
    Otherwise a chunk from the middle of a 500-row export reads
    "| Bergen | 4412 |" with nothing saying which column is which.
    """
    segments = segment_blocks([_table(rows=40)], BUDGET)

    for segment in segments:
        assert segment.text.startswith("| city | orders |\n| --- | --- |")


def test_a_table_is_never_split_inside_a_row():
    segments = segment_blocks([_table(rows=40)], BUDGET)

    for segment in segments:
        for line in segment.text.split("\n"):
            assert line.startswith("|") and line.endswith("|")


def test_code_fences_are_kept_whole():
    fence = Block(kind="text", text="```python\nx = 1\n# a comment\n```")

    segments = segment_blocks([fence], BUDGET)

    assert segments[0].atomic is True


def test_prose_either_side_of_a_table_stays_one_run():
    """So it is still chunked together with overlap, rather than in halves."""
    grouped = prose_runs(segment_blocks([_prose(), _table(rows=2), _prose()], BUDGET))

    # prose run, table, prose run
    assert [isinstance(item, list) for item in grouped] == [True, False, True]


# --- Recursive (the default) ------------------------------------------------


def test_recursive_embeds_exactly_what_it_returns():
    drafts = RecursiveChunking().split([_prose(PROSE * 3)], BUDGET)

    assert len(drafts) > 1
    assert all(draft.embed_text is None for draft in drafts)
    assert all(draft.parent_key is None for draft in drafts)


def test_recursive_keeps_a_table_out_of_its_prose_chunks():
    drafts = RecursiveChunking().split([_prose(), _table(rows=2), _prose()], BUDGET)

    table_drafts = [draft for draft in drafts if "| city |" in draft.text]
    assert len(table_drafts) == 1
    assert table_drafts[0].text.startswith("| city |")


def test_no_chunk_exceeds_the_budget():
    drafts = RecursiveChunking().split([_prose(PROSE * 5), _table(rows=40)], BUDGET)

    assert all(len(draft.text) <= BUDGET.size for draft in drafts)


# --- Semantic ---------------------------------------------------------------


class FakeTopicEmbeddings:
    """
    Embeds by topic keyword, so "distance" is unambiguous: sentences sharing a
    keyword are identical vectors, sentences that don't are orthogonal.
    """

    def embed_texts(self, texts):
        from rag.embedding_service import EmbeddedChunk

        return [
            EmbeddedChunk(
                text=text,
                embedding=[1.0, 0.0] if "network" in text or "gradient" in text else [0.0, 1.0],
            )
            for text in texts
        ]


def test_semantic_cuts_where_the_subject_changes():
    strategy = SemanticChunking(FakeTopicEmbeddings(), breakpoint_percentile=50)
    text = (
        "Neural networks learn by adjusting weights. "
        "Backpropagation computes the gradient. "
        "The kitchen was painted yellow. "
        "New cabinets were installed."
    )

    groups = strategy._semantic_groups(text)

    assert len(groups) == 2
    assert "Neural networks" in groups[0] and "gradient" in groups[0]
    assert "kitchen" in groups[1]


def test_semantic_leaves_a_short_passage_alone():
    """Two sentences give one distance, and a percentile of one number is it."""
    strategy = SemanticChunking(FakeTopicEmbeddings())

    assert strategy._semantic_groups("One sentence. Two sentences.") == [
        "One sentence. Two sentences."
    ]


def test_semantic_still_respects_the_token_budget():
    # Groups are chosen by meaning and know nothing about the budget, so
    # oversized ones must still go through the ordinary splitter.
    strategy = SemanticChunking(FakeTopicEmbeddings())

    drafts = strategy.split([_prose(PROSE * 6)], BUDGET)

    assert all(len(draft.text) <= BUDGET.size for draft in drafts)


# --- Sentence window --------------------------------------------------------


def test_sentence_window_embeds_a_sentence_and_returns_more():
    drafts = SentenceWindowChunking(window_sentences=2).split([_prose()], BUDGET)

    assert len(drafts) == 5  # one per sentence
    widened = [draft for draft in drafts if len(draft.text) > len(draft.embedded())]
    assert widened, "no window was wider than the sentence it embedded"


def test_the_embedded_sentence_always_appears_in_what_is_returned():
    drafts = SentenceWindowChunking(window_sentences=2).split([_prose()], BUDGET)

    for draft in drafts:
        assert draft.embedded() in draft.text


def test_neighbouring_sentences_share_a_parent_so_they_can_be_collapsed():
    drafts = SentenceWindowChunking(window_sentences=2).split([_prose()], BUDGET)

    keys = [draft.parent_key for draft in drafts]
    assert all(key for key in keys)
    assert len(set(keys)) < len(keys), "every sentence got its own key; nothing would collapse"


def test_a_window_never_exceeds_the_budget():
    tight = ChunkBudget(size=60, overlap=10, measure=len)

    drafts = SentenceWindowChunking(window_sentences=3).split([_prose()], tight)

    assert all(len(draft.text) <= tight.size for draft in drafts)


def test_a_table_has_no_sentences_to_window_over():
    drafts = SentenceWindowChunking().split([_table(rows=2)], BUDGET)

    assert len(drafts) == 1
    assert drafts[0].embed_text is None


# --- Parent document --------------------------------------------------------


def test_children_are_embedded_and_the_parent_is_returned():
    drafts = ParentDocumentChunking(child_tokens=40).split([_prose(PROSE * 2)], BUDGET)

    assert len(drafts) > 1
    for draft in drafts:
        assert draft.embedded() != draft.text
        assert draft.embedded() in draft.text or draft.embedded().strip() in draft.text


def test_all_children_of_one_parent_return_the_same_text():
    drafts = ParentDocumentChunking(child_tokens=40).split([_prose()], BUDGET)

    by_parent: dict[str, set[str]] = {}
    for draft in drafts:
        by_parent.setdefault(draft.parent_key, set()).add(draft.text)

    assert all(len(texts) == 1 for texts in by_parent.values())


def test_children_are_smaller_than_their_parent():
    drafts = ParentDocumentChunking(child_tokens=40).split([_prose(PROSE * 2)], BUDGET)

    assert all(len(draft.embedded()) <= 40 for draft in drafts)


def test_parents_are_distinguishable_from_each_other():
    drafts = ParentDocumentChunking(child_tokens=40).split([_prose(PROSE * 4)], BUDGET)

    assert len({draft.parent_key for draft in drafts}) > 1


# --- Propositional ----------------------------------------------------------


class FakeLLM:
    """Returns a canned reply, and records what it was asked."""

    def __init__(self, reply: str = "", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.calls: list[str] = []

    def generate_response(self, user_message: str, history=None, system_prompt: str = "") -> str:
        self.calls.append(user_message)
        if self.fail:
            raise RuntimeError("the model is down")
        return self.reply


def test_propositions_become_the_embedded_text():
    llm = FakeLLM(
        "The trial began in March 2026.\n"
        "The trial was halted six weeks after it began.\n"
        "Three sites reported calibration failures."
    )

    drafts = PropositionalChunking(llm).split([_prose()], WIDE)

    assert [draft.embedded() for draft in drafts] == [
        "The trial began in March 2026.",
        "The trial was halted six weeks after it began.",
        "Three sites reported calibration failures.",
    ]
    # ...while the source passage is what a model would actually read.
    assert all("Dr. Patel" in draft.text for draft in drafts)


def test_propositions_from_one_passage_share_a_parent():
    llm = FakeLLM("The trial began in March 2026.\nThree sites reported failures.")

    drafts = PropositionalChunking(llm).split([_prose()], WIDE)

    assert len({draft.parent_key for draft in drafts}) == 1


def test_bullets_and_numbering_are_stripped():
    llm = FakeLLM("1. The trial began in March 2026.\n- Three sites reported failures.")

    drafts = PropositionalChunking(llm).split([_prose()], WIDE)

    assert drafts[0].embedded() == "The trial began in March 2026."
    assert drafts[1].embedded() == "Three sites reported failures."


def test_fragments_are_discarded():
    llm = FakeLLM("ok\n-\nThe trial began in March 2026.")

    drafts = PropositionalChunking(llm).split([_prose()], WIDE)

    assert [draft.embedded() for draft in drafts] == ["The trial began in March 2026."]


def test_a_failing_model_leaves_the_passage_whole():
    """A flaky API must degrade the index, never corrupt it."""
    drafts = PropositionalChunking(FakeLLM(fail=True)).split([_prose()], WIDE)

    assert len(drafts) == 1
    assert drafts[0].embed_text is None
    assert drafts[0].parent_key is None


def test_an_empty_reply_leaves_the_passage_whole():
    drafts = PropositionalChunking(FakeLLM("")).split([_prose()], WIDE)

    assert len(drafts) == 1
    assert drafts[0].embed_text is None


def test_tables_are_not_sent_to_the_model():
    """A table is already a set of atomic facts, one per row."""
    llm = FakeLLM("something")

    PropositionalChunking(llm).split([_table(rows=2)], WIDE)

    assert llm.calls == []


# --- Contextual enrichment --------------------------------------------------


def test_context_is_prepended_to_both_texts():
    llm = FakeLLM("This excerpt is from ACME's Q2 2023 filing.")
    enriched = ContextualEnrichment(
        RecursiveChunking(), llm, reserved_tokens=60, document_context="ACME Q2 2023"
    )

    drafts = enriched.split([_prose()], BUDGET)

    assert drafts[0].text.startswith("This excerpt is from ACME's Q2 2023 filing.")
    assert drafts[0].embedded().startswith("This excerpt is from ACME's Q2 2023 filing.")


def test_enrichment_composes_with_any_strategy():
    llm = FakeLLM("Context line.")
    enriched = ContextualEnrichment(
        ParentDocumentChunking(child_tokens=40), llm, reserved_tokens=60
    )

    drafts = enriched.split([_prose()], BUDGET)

    # The parent/child split survives; only the texts changed.
    assert all(draft.parent_key for draft in drafts)
    assert all(draft.embedded() != draft.text for draft in drafts)


def test_an_over_long_context_line_is_cut_to_its_first_sentence():
    """A model asked for one sentence sometimes writes four."""
    llm = FakeLLM("The first sentence. " + "And more padding. " * 20)
    enriched = ContextualEnrichment(RecursiveChunking(), llm, reserved_tokens=25)

    drafts = enriched.split([_prose()], BUDGET)

    assert drafts[0].text.startswith("The first sentence.")
    assert "padding" not in drafts[0].text.split("\n\n")[0]


def test_a_failing_model_leaves_chunks_unenriched():
    enriched = ContextualEnrichment(RecursiveChunking(), FakeLLM(fail=True), reserved_tokens=60)

    drafts = enriched.split([_prose()], BUDGET)

    assert drafts[0].embed_text is None


# --- Selection --------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["recursive", "semantic", "sentence_window", "parent_document", "propositional"]
)
def test_every_named_strategy_can_be_built(name):
    strategy = build_strategy(name, embedding_service=FakeTopicEmbeddings(), llm_service=FakeLLM())

    assert strategy.name == name


def test_an_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        build_strategy("magic")


def test_a_strategy_missing_its_dependency_is_rejected():
    """Better than silently chunking the corpus a different way than configured."""
    with pytest.raises(ValueError, match="embedding service"):
        build_strategy("semantic")
    with pytest.raises(ValueError, match="LLM service"):
        build_strategy("propositional")


def test_the_default_strategy_needs_no_model_at_all():
    assert build_strategy("recursive").name == "recursive"
