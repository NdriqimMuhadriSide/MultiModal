import pytest

from rag.keyword_index import BM25, KeywordIndex, tokenize
from rag.vector_store import ChunkRecord


def _record(chunk_id: str, text: str, **metadata) -> ChunkRecord:
    return ChunkRecord(chunk_id=chunk_id, text=text, metadata=metadata)


class FakeVectorStore:
    """Enough of rag/vector_store.py for the index's refresh logic."""

    def __init__(self, records: list[ChunkRecord]) -> None:
        self.records = records
        self._writes = 0
        self.all_chunks_calls = 0

    def replace(self, records: list[ChunkRecord]) -> None:
        self.records = records
        self._writes += 1

    def all_chunks(self) -> list[ChunkRecord]:
        self.all_chunks_calls += 1
        return self.records

    def fingerprint(self) -> tuple[int, int]:
        return (self._writes, len(self.records))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_punctuation_splits_identifiers_into_matchable_terms():
    """"ERR-4021" has to match a question that writes it either way."""
    assert tokenize("Error ERR-4021 occurred") == ["error", "err", "4021", "occurred"]


def test_tokenizing_is_case_insensitive():
    assert tokenize("Refund") == tokenize("REFUND") == ["refund"]


def test_stopwords_are_dropped():
    assert tokenize("what is the refund policy") == ["refund", "policy"]


def test_digits_survive_tokenization():
    """Years, versions and article numbers are exactly what this index is for."""
    assert "2016" in tokenize("Regulation (EU) 2016/679")


# ---------------------------------------------------------------------------
# BM25 ranking
# ---------------------------------------------------------------------------


def test_the_chunk_containing_the_query_term_ranks_first():
    index = BM25(
        [
            _record("a", "Our returns policy covers unopened items."),
            _record("b", "Error ERR-4021 means the upload timed out."),
            _record("c", "Office hours are nine to five."),
        ]
    )

    matches = index.search("ERR-4021")

    assert [match.chunk_id for match in matches] == ["b"]


def test_chunks_sharing_no_term_with_the_query_are_not_returned():
    """A zero score is a non-match, not a weak match."""
    index = BM25([_record("a", "Office hours are nine to five.")])

    assert index.search("photosynthesis") == []


def test_a_rarer_term_outweighs_a_common_one():
    """
    IDF is what makes this useful next to dense search: the term that appears
    once in the corpus decides the ranking, not the one in every chunk.
    """
    index = BM25(
        [_record(f"common{i}", "the invoice was processed") for i in range(10)]
        + [_record("rare", "the invoice mentions ERR-4021")]
    )

    matches = index.search("invoice ERR-4021")

    assert matches[0].chunk_id == "rare"


def test_term_frequency_saturates():
    """
    Twenty repetitions must not score twenty times one repetition, or keyword
    stuffing would beat a genuinely relevant passage.
    """
    index = BM25(
        [
            _record("once", "refund " + "filler " * 19),
            _record("many", "refund " * 20),
        ]
    )

    scores = {match.chunk_id: match.score for match in index.search("refund")}

    assert scores["many"] > scores["once"]
    assert scores["many"] < 20 * scores["once"]


def test_a_common_term_never_scores_negative():
    """
    The textbook IDF goes negative past 50% document frequency, which would
    make containing a query word actively harmful. This variant does not.
    """
    index = BM25([_record(f"c{i}", "invoice") for i in range(10)])

    assert all(match.score > 0 for match in index.search("invoice"))


def test_search_respects_top_k():
    index = BM25([_record(f"c{i}", f"invoice number {i}") for i in range(10)])

    assert len(index.search("invoice", top_k=3)) == 3


def test_a_shorter_chunk_wins_when_both_match_equally():
    """Length normalisation: the same hit means more in less surrounding text."""
    index = BM25(
        [
            _record("short", "refund policy"),
            _record("long", "refund policy " + "unrelated wording here " * 20),
        ]
    )

    assert index.search("refund")[0].chunk_id == "short"


def test_searching_an_empty_corpus_returns_nothing():
    assert BM25([]).search("anything") == []


def test_search_rejects_an_empty_query():
    with pytest.raises(ValueError):
        BM25([_record("a", "text")]).search("")


def test_search_rejects_a_non_positive_top_k():
    with pytest.raises(ValueError):
        BM25([_record("a", "text")]).search("text", top_k=0)


def test_a_query_of_only_stopwords_matches_nothing():
    """Nothing was asked for, so nothing should be put in front of the model."""
    index = BM25([_record("a", "the refund policy is here")])

    assert index.search("what is the") == []


def test_matches_carry_the_chunk_s_metadata():
    index = BM25([_record("a", "refund policy", filename="policy.pdf", page_number=4)])

    match = index.search("refund")[0]

    assert match.metadata["filename"] == "policy.pdf"
    assert match.metadata["page_number"] == 4


# ---------------------------------------------------------------------------
# Staying in step with the store
# ---------------------------------------------------------------------------


def test_the_index_is_built_lazily_and_reused():
    store = FakeVectorStore([_record("a", "refund policy")])
    index = KeywordIndex(store)

    assert store.all_chunks_calls == 0
    index.search("refund")
    index.search("refund")
    assert store.all_chunks_calls == 1


def test_a_newly_ingested_chunk_is_searchable_without_any_wiring():
    """
    Pulled from the store rather than pushed to on ingest, so a future write
    path can't forget to notify it.
    """
    store = FakeVectorStore([_record("a", "refund policy")])
    index = KeywordIndex(store)
    index.search("refund")

    store.replace([_record("a", "refund policy"), _record("b", "ERR-4021 explained")])

    assert [match.chunk_id for match in index.search("ERR-4021")] == ["b"]


def test_a_deleted_chunk_stops_being_returned():
    store = FakeVectorStore([_record("a", "refund policy"), _record("b", "ERR-4021")])
    index = KeywordIndex(store)
    index.search("refund")

    store.replace([_record("a", "refund policy")])

    assert index.search("ERR-4021") == []


def test_a_delete_and_reingest_of_the_same_size_still_rebuilds():
    """The count alone wouldn't catch this; the write counter does."""
    store = FakeVectorStore([_record("a", "refund policy")])
    index = KeywordIndex(store)
    index.search("refund")

    store.replace([_record("b", "ERR-4021 explained")])

    assert [match.chunk_id for match in index.search("ERR-4021")] == ["b"]
