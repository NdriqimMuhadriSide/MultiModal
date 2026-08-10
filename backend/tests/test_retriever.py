import pytest

from rag.keyword_index import KeywordMatch
from rag.retriever import Retriever
from rag.vector_store import SearchResult


class FakeEmbeddingService:
    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("text must not be empty.")
        return [float(len(text))]


class FakeVectorStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_top_k: int | None = None

    def search(self, query_embedding, top_k: int = 5):
        self.last_top_k = top_k
        return self._results[:top_k]


class FakeKeywordIndex:
    """Stands in for rag/keyword_index.py's BM25 - the retriever only needs its order."""

    def __init__(self, matches: list[KeywordMatch]) -> None:
        self._matches = matches
        self.last_top_k: int | None = None

    def search(self, query: str, top_k: int = 5):
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        self.last_top_k = top_k
        return self._matches[:top_k]


def _search_result(chunk_id: str, text: str, similarity: float, **metadata) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, metadata=metadata, similarity=similarity)


def _keyword_match(chunk_id: str, text: str, score: float, **metadata) -> KeywordMatch:
    return KeywordMatch(chunk_id=chunk_id, text=text, metadata=metadata, score=score)


def _dense_retriever(store, **kwargs) -> Retriever:
    """
    A dense-only retriever, whatever RETRIEVAL_MODE happens to default to.

    Pinned explicitly in every dense test below so those tests keep asserting
    on dense behaviour rather than quietly becoming hybrid tests when the
    default changes.
    """
    return Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Dense retrieval - unchanged behaviour
# ---------------------------------------------------------------------------


def test_retrieve_returns_chunks_with_source_metadata():
    results = [
        _search_result(
            "policy.pdf::p1::c0",
            "Refunds are issued within 30 days of purchase.",
            similarity=0.8,
            filename="policy.pdf",
            page_number=1,
        )
    ]
    retriever = _dense_retriever(FakeVectorStore(results))

    chunks = retriever.retrieve("What is our refund policy?")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "policy.pdf::p1::c0"
    assert chunks[0].filename == "policy.pdf"
    assert chunks[0].page == 1
    assert chunks[0].score == 0.8
    assert chunks[0].dense_score == 0.8
    assert chunks[0].keyword_score is None
    assert chunks[0].matched_by == "dense"
    assert "Refunds are issued" in chunks[0].text


def test_retrieve_filters_out_low_similarity_chunks():
    results = [
        _search_result("a", "irrelevant chunk", similarity=0.05, filename="doc.pdf", page_number=1),
        _search_result("b", "relevant chunk", similarity=0.9, filename="doc.pdf", page_number=2),
    ]
    retriever = _dense_retriever(FakeVectorStore(results), min_similarity=0.2)

    chunks = retriever.retrieve("a question")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "b"


def test_retrieve_overfetches_so_dedup_still_leaves_top_k():
    """
    Chunking strategies that point many chunks at one passage get collapsed
    after the search, so asking the store for exactly top_k would come back
    short. The extra candidates are what make up the difference.
    """
    from app.core.config import settings

    store = FakeVectorStore([])
    retriever = _dense_retriever(store)

    retriever.retrieve("a question", top_k=3)

    assert store.last_top_k == 3 * settings.retrieval_overfetch


def test_retrieve_defaults_to_top_5():
    from app.core.config import settings

    store = FakeVectorStore([])
    retriever = _dense_retriever(store)

    retriever.retrieve("a question")

    assert store.last_top_k == 5 * settings.retrieval_overfetch


def test_retrieve_returns_no_more_than_top_k():
    store = FakeVectorStore(
        [
            SearchResult(
                chunk_id=f"c{index}",
                text=f"chunk {index}",
                metadata={"filename": "doc.pdf", "page_number": 1},
                similarity=0.9,
            )
            for index in range(20)
        ]
    )
    retriever = _dense_retriever(store)

    assert len(retriever.retrieve("a question", top_k=3)) == 3


def test_chunks_pointing_at_the_same_passage_are_collapsed():
    """
    Parent-document and sentence-window embed several chunks per passage. If
    all of them survived, one strong paragraph would take every slot in top-k
    and the model would see the same text five times.
    """
    store = FakeVectorStore(
        [
            SearchResult(
                chunk_id=f"c{index}",
                text="the same parent passage",
                metadata={"filename": "doc.pdf", "page_number": 1, "parent_id": "p1"},
                similarity=0.9 - index / 100,
            )
            for index in range(4)
        ]
        + [
            SearchResult(
                chunk_id="other",
                text="a different passage",
                metadata={"filename": "doc.pdf", "page_number": 2, "parent_id": "p2"},
                similarity=0.5,
            )
        ]
    )
    retriever = _dense_retriever(store)

    results = retriever.retrieve("a question", top_k=5)

    assert [chunk.chunk_id for chunk in results] == ["c0", "other"]


def test_chunks_without_a_parent_are_never_collapsed_together():
    """An empty key means "stands alone", not "these all match"."""
    store = FakeVectorStore(
        [
            SearchResult(
                chunk_id=f"c{index}",
                text=f"chunk {index}",
                metadata={"filename": "doc.pdf", "page_number": 1, "parent_id": ""},
                similarity=0.9,
            )
            for index in range(3)
        ]
    )
    retriever = _dense_retriever(store)

    assert len(retriever.retrieve("a question", top_k=5)) == 3


def test_retrieve_rejects_empty_question():
    retriever = _dense_retriever(FakeVectorStore([]))

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_retrieve_returns_empty_list_when_store_has_no_results():
    retriever = _dense_retriever(FakeVectorStore([]))

    assert retriever.retrieve("anything") == []


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        Retriever(
            embedding_service=FakeEmbeddingService(),
            vector_store=FakeVectorStore([]),
            mode="fuzzy",
        )


def test_hybrid_without_a_keyword_index_is_rejected():
    """
    Better a startup error than a deployment that says hybrid, runs dense, and
    looks entirely healthy while answering worse.
    """
    with pytest.raises(ValueError, match="needs a keyword index"):
        Retriever(
            embedding_service=FakeEmbeddingService(),
            vector_store=FakeVectorStore([]),
            mode="hybrid",
        )


def test_keyword_mode_never_embeds_the_question():
    """Keyword-only retrieval should not load or call the embedding model."""

    class ExplodingEmbeddingService:
        def embed_text(self, text: str):
            raise AssertionError("keyword mode must not embed the question")

    retriever = Retriever(
        embedding_service=ExplodingEmbeddingService(),
        vector_store=FakeVectorStore([_search_result("dense-only", "x", 0.99)]),
        keyword_index=FakeKeywordIndex(
            [_keyword_match("k1", "an exact term match", 12.0, filename="doc.pdf", page_number=3)]
        ),
        mode="keyword",
    )

    chunks = retriever.retrieve("ERR-4021")

    assert [chunk.chunk_id for chunk in chunks] == ["k1"]
    assert chunks[0].matched_by == "keyword"
    assert chunks[0].keyword_score == 12.0
    assert chunks[0].dense_score is None
    assert chunks[0].page == 3


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


def _hybrid_retriever(dense, keyword, **kwargs) -> Retriever:
    return Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(dense),
        keyword_index=FakeKeywordIndex(keyword),
        mode="hybrid",
        **kwargs,
    )


def test_hybrid_returns_chunks_only_one_half_found():
    """
    The whole point: a chunk containing the literal term the user typed
    reaches the model even though the embedding never surfaced it.
    """
    retriever = _hybrid_retriever(
        dense=[_search_result("d1", "semantically close", 0.7, filename="a.pdf", page_number=1)],
        keyword=[_keyword_match("k1", "mentions ERR-4021", 9.0, filename="b.pdf", page_number=2)],
    )

    chunks = retriever.retrieve("what causes ERR-4021", top_k=5)

    assert {chunk.chunk_id for chunk in chunks} == {"d1", "k1"}
    assert {chunk.matched_by for chunk in chunks} == {"dense", "keyword"}


def test_hybrid_merges_a_chunk_both_halves_found_into_one_result():
    """Two hits on the same chunk are one piece of evidence, not two slots."""
    retriever = _hybrid_retriever(
        dense=[_search_result("shared", "text", 0.6, filename="a.pdf", page_number=1)],
        keyword=[_keyword_match("shared", "text", 8.0, filename="a.pdf", page_number=1)],
    )

    chunks = retriever.retrieve("a question", top_k=5)

    assert len(chunks) == 1
    assert chunks[0].matched_by == "both"
    assert chunks[0].dense_score == 0.6
    assert chunks[0].keyword_score == 8.0


def test_hybrid_ranks_agreement_above_either_halfs_first_place():
    """
    RRF's defining behaviour: a chunk ranked 2nd by both retrievers beats one
    ranked 1st by a single retriever, because agreement is the stronger signal.
    """
    retriever = _hybrid_retriever(
        dense=[
            _search_result("dense-first", "d", 0.9, filename="a.pdf", page_number=1),
            _search_result("agreed", "both", 0.8, filename="a.pdf", page_number=2),
        ],
        keyword=[
            _keyword_match("keyword-first", "k", 20.0, filename="a.pdf", page_number=3),
            _keyword_match("agreed", "both", 15.0, filename="a.pdf", page_number=2),
        ],
    )

    chunks = retriever.retrieve("a question", top_k=3)

    assert chunks[0].chunk_id == "agreed"
    assert chunks[0].matched_by == "both"


def test_hybrid_scores_are_fused_not_raw():
    """
    A fused score is a rank score, not a similarity - it must not be mistaken
    for one by anything downstream, so it should not equal either input.
    """
    retriever = _hybrid_retriever(
        dense=[_search_result("d1", "text", 0.6, filename="a.pdf", page_number=1)],
        keyword=[_keyword_match("k1", "text", 8.0, filename="a.pdf", page_number=2)],
    )

    chunks = retriever.retrieve("a question", top_k=5)

    for chunk in chunks:
        assert chunk.score not in (0.6, 8.0)
        assert 0 < chunk.score < 1


def test_hybrid_still_applies_the_similarity_floor_to_the_dense_half():
    """A weak dense match stays excluded; keyword hits are judged by BM25 instead."""
    retriever = _hybrid_retriever(
        dense=[_search_result("weak", "barely related", 0.05, filename="a.pdf", page_number=1)],
        keyword=[_keyword_match("k1", "exact term", 9.0, filename="b.pdf", page_number=2)],
        min_similarity=0.2,
    )

    chunks = retriever.retrieve("a question", top_k=5)

    assert [chunk.chunk_id for chunk in chunks] == ["k1"]


def test_hybrid_collapses_duplicate_passages_across_both_halves():
    """
    Collapsing has to happen after fusion: if each half deduped alone, a parent
    found once by each would still take two slots.
    """
    retriever = _hybrid_retriever(
        dense=[
            _search_result(
                "child-a", "same passage", 0.8, filename="a.pdf", page_number=1, parent_id="p1"
            )
        ],
        keyword=[
            _keyword_match(
                "child-b", "same passage", 9.0, filename="a.pdf", page_number=1, parent_id="p1"
            )
        ],
    )

    chunks = retriever.retrieve("a question", top_k=5)

    assert len(chunks) == 1


def test_hybrid_overfetches_both_halves():
    from app.core.config import settings

    store = FakeVectorStore([])
    index = FakeKeywordIndex([])
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        keyword_index=index,
        mode="hybrid",
    )

    retriever.retrieve("a question", top_k=3)

    assert store.last_top_k == 3 * settings.retrieval_overfetch
    assert index.last_top_k == 3 * settings.retrieval_overfetch


def test_hybrid_with_a_zero_weighted_half_keeps_the_other_halfs_real_scores(monkeypatch):
    """
    HYBRID_KEYWORD_WEIGHT=0 means "dense only", and dense only should report
    cosine similarities rather than rank scores derived from one list.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "hybrid_keyword_weight", 0.0)
    retriever = _hybrid_retriever(
        dense=[_search_result("d1", "text", 0.6, filename="a.pdf", page_number=1)],
        keyword=[_keyword_match("k1", "text", 8.0, filename="b.pdf", page_number=2)],
    )

    chunks = retriever.retrieve("a question", top_k=5)

    assert [chunk.chunk_id for chunk in chunks] == ["d1"]
    assert chunks[0].score == 0.6


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


class FakeReranker:
    """Scores by a lookup on chunk text, so a test can state the verdict it wants."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores = scores_by_text
        self.scored_texts: list[str] | None = None
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        self.scored_texts = list(texts)
        return [self._scores.get(text, 0.5) for text in texts]


def test_reranking_reorders_the_shortlist():
    """
    The chunk retrieval ranked last wins when the cross-encoder says it is the
    one that answers the question - which is the entire point of the pass.
    """
    store = FakeVectorStore(
        [
            _search_result("c0", "about the topic", 0.9, filename="a.pdf", page_number=1),
            _search_result("c1", "answers the question", 0.4, filename="a.pdf", page_number=2),
        ]
    )
    retriever = _dense_retriever(
        store,
        reranker=FakeReranker({"about the topic": 0.10, "answers the question": 0.95}),
    )

    chunks = retriever.retrieve("a question", top_k=2)

    assert [chunk.chunk_id for chunk in chunks] == ["c1", "c0"]
    assert chunks[0].rerank_score == 0.95
    assert chunks[0].score == 0.95


def test_reranking_scores_the_whole_overfetched_shortlist_not_just_top_k():
    """Reranking only helps if it can promote a chunk from outside the top_k."""
    store = FakeVectorStore(
        [
            _search_result(f"c{i}", f"chunk {i}", 0.9 - i / 100, filename="a.pdf", page_number=i)
            for i in range(12)
        ]
    )
    reranker = FakeReranker({"chunk 11": 0.99})
    retriever = _dense_retriever(store, reranker=reranker)

    chunks = retriever.retrieve("a question", top_k=3)

    assert len(reranker.scored_texts) == 12
    assert chunks[0].chunk_id == "c11"


def test_reranking_runs_after_duplicate_collapsing():
    """Scoring two chunks of one passage spends a forward pass to learn nothing."""
    store = FakeVectorStore(
        [
            SearchResult(
                chunk_id=f"c{i}",
                text="the same parent passage",
                metadata={"filename": "a.pdf", "page_number": 1, "parent_id": "p1"},
                similarity=0.9 - i / 100,
            )
            for i in range(4)
        ]
    )
    reranker = FakeReranker({})
    retriever = _dense_retriever(store, reranker=reranker)

    retriever.retrieve("a question", top_k=5)

    assert reranker.scored_texts == ["the same parent passage"]


def test_reranking_drops_candidates_below_the_floor(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "rerank_min_score", 0.5)
    store = FakeVectorStore(
        [
            _search_result("keep", "relevant", 0.9, filename="a.pdf", page_number=1),
            _search_result("drop", "irrelevant", 0.8, filename="a.pdf", page_number=2),
        ]
    )
    retriever = _dense_retriever(
        store, reranker=FakeReranker({"relevant": 0.9, "irrelevant": 0.01})
    )

    chunks = retriever.retrieve("a question", top_k=5)

    assert [chunk.chunk_id for chunk in chunks] == ["keep"]


def test_a_zero_floor_only_reorders_and_never_drops(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "rerank_min_score", 0.0)
    store = FakeVectorStore(
        [
            _search_result("a", "one", 0.9, filename="a.pdf", page_number=1),
            _search_result("b", "two", 0.8, filename="a.pdf", page_number=2),
        ]
    )
    retriever = _dense_retriever(store, reranker=FakeReranker({"one": 0.0, "two": 0.0}))

    assert len(retriever.retrieve("a question", top_k=5)) == 2


def test_reranking_keeps_retrieval_order_for_candidates_it_cannot_separate():
    """Equal verdicts must not shuffle results run to run."""
    store = FakeVectorStore(
        [
            _search_result(f"c{i}", f"chunk {i}", 0.9 - i / 100, filename="a.pdf", page_number=i)
            for i in range(5)
        ]
    )
    retriever = _dense_retriever(store, reranker=FakeReranker({}))

    chunks = retriever.retrieve("a question", top_k=5)

    assert [chunk.chunk_id for chunk in chunks] == ["c0", "c1", "c2", "c3", "c4"]


def test_reranking_preserves_the_retrieval_scores_it_replaced():
    store = FakeVectorStore([_search_result("c0", "text", 0.77, filename="a.pdf", page_number=1)])
    retriever = _dense_retriever(store, reranker=FakeReranker({"text": 0.42}))

    chunk = retriever.retrieve("a question", top_k=1)[0]

    assert chunk.dense_score == 0.77
    assert chunk.rerank_score == 0.42
    assert chunk.score == 0.42


def test_without_a_reranker_nothing_is_scored_and_the_field_stays_none():
    store = FakeVectorStore([_search_result("c0", "text", 0.77, filename="a.pdf", page_number=1)])
    retriever = _dense_retriever(store)

    chunk = retriever.retrieve("a question", top_k=1)[0]

    assert chunk.rerank_score is None
    assert chunk.score == 0.77


def test_reranking_an_empty_shortlist_never_calls_the_model():
    reranker = FakeReranker({})
    retriever = _dense_retriever(FakeVectorStore([]), reranker=reranker)

    assert retriever.retrieve("a question") == []
    assert reranker.calls == 0


def test_reranking_composes_with_hybrid_retrieval():
    """A keyword-only chunk must be rerankable like any other."""
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(
            [_search_result("d1", "dense hit", 0.9, filename="a.pdf", page_number=1)]
        ),
        keyword_index=FakeKeywordIndex(
            [_keyword_match("k1", "keyword hit", 9.0, filename="b.pdf", page_number=2)]
        ),
        mode="hybrid",
        reranker=FakeReranker({"dense hit": 0.2, "keyword hit": 0.9}),
    )

    chunks = retriever.retrieve("a question", top_k=2)

    assert [chunk.chunk_id for chunk in chunks] == ["k1", "d1"]
    assert chunks[0].matched_by == "keyword"
    assert chunks[0].keyword_score == 9.0
    assert chunks[0].rerank_score == 0.9


# ---------------------------------------------------------------------------
# Multi-query retrieval
# ---------------------------------------------------------------------------


class FakeQueryExpander:
    def __init__(self, variants: list[str]) -> None:
        self._variants = variants
        self.expanded: str | None = None

    def expand(self, question: str) -> list[str]:
        self.expanded = question
        return self._variants


class PerQueryVectorStore:
    """Returns different dense results depending on which query was embedded."""

    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self._results = results_by_query
        self.queries: list[str] = []

    def search(self, query_embedding, top_k: int = 5):
        # FakeEmbeddingService encodes the query as its length, so the length
        # is what identifies which phrasing is being searched for.
        length = int(query_embedding[0])
        for query, results in self._results.items():
            if len(query) == length:
                self.queries.append(query)
                return results[:top_k]
        return []


def test_a_variant_can_contribute_a_chunk_the_original_never_found():
    """The entire point of expansion."""
    store = PerQueryVectorStore(
        {
            "how do I stop paying": [
                _search_result("wrong", "about billing generally", 0.5, filename="a.pdf", page_number=1)
            ],
            "how do I cancel my subscription?": [
                _search_result("right", "cancellation clause", 0.9, filename="a.pdf", page_number=2)
            ],
        }
    )
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        multi_query=True, query_expander=FakeQueryExpander(["how do I cancel my subscription?"]),
    )

    chunks = retriever.retrieve("how do I stop paying", top_k=5)

    assert {chunk.chunk_id for chunk in chunks} == {"wrong", "right"}


def test_found_by_query_records_which_phrasing_surfaced_each_chunk():
    store = PerQueryVectorStore(
        {
            "original question": [
                _search_result("a", "found by original", 0.8, filename="a.pdf", page_number=1)
            ],
            "a rewritten phrasing!": [
                _search_result("b", "found by variant", 0.8, filename="a.pdf", page_number=2)
            ],
        }
    )
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        multi_query=True, query_expander=FakeQueryExpander(["a rewritten phrasing!"]),
    )

    found = {chunk.chunk_id: chunk.found_by_query for chunk in retriever.retrieve("original question")}

    assert found["a"] == "original question"
    assert found["b"] == "a rewritten phrasing!"


def test_the_original_question_is_always_searched_for_first():
    """It is the safeguard against a rewrite that drops a literal identifier."""
    store = PerQueryVectorStore({"what causes ERR-4021": [], "vari": []})
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        multi_query=True, query_expander=FakeQueryExpander(["vari"]),
    )

    retriever.retrieve("what causes ERR-4021")

    assert store.queries[0] == "what causes ERR-4021"


def test_a_chunk_several_phrasings_agree_on_outranks_one_only_a_variant_found():
    agreed = _search_result("agreed", "agreed text", 0.5, filename="a.pdf", page_number=1)
    store = PerQueryVectorStore(
        {
            "aaaa": [
                _search_result("orig-only", "orig text", 0.9, filename="a.pdf", page_number=2),
                agreed,
            ],
            "bbbbb": [
                _search_result("var-only", "var text", 0.9, filename="a.pdf", page_number=3),
                agreed,
            ],
        }
    )
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        multi_query=True, query_expander=FakeQueryExpander(["bbbbb"]),
    )

    chunks = retriever.retrieve("aaaa", top_k=3)

    assert chunks[0].chunk_id == "agreed"


def test_expansion_returning_nothing_leaves_retrieval_exactly_as_it_was():
    """An LLM failure must degrade to the plain path, not a different one."""
    results = [_search_result("a", "text", 0.8, filename="a.pdf", page_number=1)]
    plain = _dense_retriever(FakeVectorStore(results)).retrieve("a question")
    expanded = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(results),
        mode="dense",
        multi_query=True, query_expander=FakeQueryExpander([]),
    ).retrieve("a question")

    assert [c.chunk_id for c in plain] == [c.chunk_id for c in expanded]
    assert plain[0].score == expanded[0].score


def test_without_an_expander_the_question_is_the_only_query():
    store = PerQueryVectorStore({"a question": []})
    _dense_retriever(store).retrieve("a question")

    assert store.queries == ["a question"]


def test_reranking_scores_against_the_original_question_not_a_variant():
    class RecordingReranker:
        def __init__(self):
            self.query = None

        def score(self, query, texts):
            self.query = query
            return [0.5] * len(texts)

    reranker = RecordingReranker()
    store = PerQueryVectorStore(
        {"the real question": [_search_result("a", "t", 0.8, filename="a.pdf", page_number=1)]}
    )
    Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        reranker=reranker,
        multi_query=True, query_expander=FakeQueryExpander(["some other phrasing entirely!!"]),
    ).retrieve("the real question")

    assert reranker.query == "the real question"


def test_multi_query_keeps_one_result_per_chunk_and_best_scores():
    shared_a = _search_result("shared", "same chunk", 0.4, filename="a.pdf", page_number=1)
    shared_b = _search_result("shared", "same chunk", 0.9, filename="a.pdf", page_number=1)
    store = PerQueryVectorStore({"aaaa": [shared_a], "bbbbb": [shared_b]})
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        multi_query=True, query_expander=FakeQueryExpander(["bbbbb"]),
    )

    chunks = retriever.retrieve("aaaa", top_k=5)

    assert len(chunks) == 1
    assert chunks[0].dense_score == 0.9


# ---------------------------------------------------------------------------
# Corrective RAG
# ---------------------------------------------------------------------------


class ScriptedReranker:
    """Returns a scripted score per call, so a retry can be made to improve or not."""

    def __init__(self, *rounds: dict[str, float]) -> None:
        self._rounds = list(rounds)
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = self._rounds[min(self.calls, len(self._rounds) - 1)]
        self.calls += 1
        return [scores.get(text, 0.0) for text in texts]


def _corrective_retriever(store, reranker, expander, **kwargs) -> Retriever:
    return Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        reranker=reranker,
        query_expander=expander,
        corrective=True,
        **kwargs,
    )


def test_corrective_requires_a_reranker():
    """Without one there is no score that means the same thing across questions."""
    with pytest.raises(ValueError, match="needs a reranker"):
        Retriever(
            embedding_service=FakeEmbeddingService(),
            vector_store=FakeVectorStore([]),
            mode="dense",
            query_expander=FakeQueryExpander([]),
            corrective=True,
        )


def test_corrective_works_without_a_query_expander(monkeypatch):
    """
    The retry is optional; the grade is not. Refusing needs no LLM, and on this
    project's eval set refusing was where all of the benefit came from - so
    gating it behind an API key would have been the wrong trade.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_reject_score", 0.02)
    monkeypatch.setattr(settings, "corrective_accept_score", 0.5)
    store = FakeVectorStore([_search_result("a", "unrelated", 0.5, filename="a.pdf", page_number=1)])
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        reranker=FakeReranker({"unrelated": 0.001}),
        corrective=True,
    )

    assert retriever.retrieve("a question", top_k=5) == []


def test_multi_query_requires_a_query_expander():
    with pytest.raises(ValueError, match="needs a query expander"):
        Retriever(
            embedding_service=FakeEmbeddingService(),
            vector_store=FakeVectorStore([]),
            mode="dense",
            multi_query=True,
        )


def test_good_retrieval_answers_without_a_retry(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_accept_score", 0.5)
    store = FakeVectorStore([_search_result("a", "relevant", 0.9, filename="a.pdf", page_number=1)])
    expander = FakeQueryExpander(["a rewritten phrasing"])
    retriever = _corrective_retriever(store, FakeReranker({"relevant": 0.99}), expander)

    chunks = retriever.retrieve("a question", top_k=5)

    assert [chunk.chunk_id for chunk in chunks] == ["a"]
    assert expander.expanded is None  # never asked - no LLM call was spent


def test_irrelevant_retrieval_returns_nothing_so_the_pipeline_refuses(monkeypatch):
    """
    The refusal *is* the empty list: rag/rag_service.py answers "I don't know"
    without calling the LLM when there is no context to build.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_reject_score", 0.02)
    monkeypatch.setattr(settings, "corrective_accept_score", 0.5)
    store = FakeVectorStore(
        [_search_result("a", "unrelated", 0.5, filename="a.pdf", page_number=1)]
    )
    retriever = _corrective_retriever(
        store, FakeReranker({"unrelated": 0.001}), FakeQueryExpander([])
    )

    assert retriever.retrieve("a question about something else", top_k=5) == []


def test_an_ambiguous_grade_triggers_a_retry_with_rephrased_queries(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_accept_score", 0.5)
    monkeypatch.setattr(settings, "corrective_reject_score", 0.02)
    store = PerQueryVectorStore(
        {
            "a middling question": [
                _search_result("meh", "middling", 0.5, filename="a.pdf", page_number=1)
            ],
            "a much better phrasing!": [
                _search_result("good", "better", 0.5, filename="a.pdf", page_number=2)
            ],
        }
    )
    expander = FakeQueryExpander(["a much better phrasing!"])
    reranker = ScriptedReranker({"middling": 0.2}, {"middling": 0.2, "better": 0.95})
    retriever = _corrective_retriever(store, reranker, expander)

    chunks = retriever.retrieve("a middling question", top_k=5)

    assert expander.expanded == "a middling question"
    assert chunks[0].chunk_id == "good"


def test_a_retry_that_finds_nothing_better_is_discarded(monkeypatch):
    """A failed retry must not be allowed to reorder a result set it didn't beat."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_accept_score", 0.9)
    monkeypatch.setattr(settings, "corrective_reject_score", 0.02)
    store = PerQueryVectorStore(
        {
            "aaaa": [_search_result("first", "original hit", 0.5, filename="a.pdf", page_number=1)],
            "bbbbb": [_search_result("other", "worse hit", 0.5, filename="a.pdf", page_number=2)],
        }
    )
    reranker = ScriptedReranker(
        {"original hit": 0.3}, {"original hit": 0.3, "worse hit": 0.1}
    )
    retriever = _corrective_retriever(store, reranker, FakeQueryExpander(["bbbbb"]))

    chunks = retriever.retrieve("aaaa", top_k=5)

    assert [chunk.chunk_id for chunk in chunks] == ["first"]


def test_a_refusal_is_only_reached_after_a_retry_has_been_tried(monkeypatch):
    """
    Giving up on the first reading would abandon a question that was merely
    worded unlike the document - the exact failure expansion exists to fix.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_accept_score", 0.5)
    monkeypatch.setattr(settings, "corrective_reject_score", 0.02)
    store = FakeVectorStore([_search_result("a", "unrelated", 0.5, filename="a.pdf", page_number=1)])
    expander = FakeQueryExpander([])
    retriever = _corrective_retriever(store, FakeReranker({"unrelated": 0.001}), expander)

    assert retriever.retrieve("a question", top_k=5) == []
    assert expander.expanded == "a question"


def test_a_retry_can_rescue_a_question_that_would_otherwise_be_refused(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_accept_score", 0.5)
    monkeypatch.setattr(settings, "corrective_reject_score", 0.02)
    store = PerQueryVectorStore(
        {
            "aaaa": [_search_result("bad", "nothing useful", 0.5, filename="a.pdf", page_number=1)],
            "bbbbb": [_search_result("good", "the real answer", 0.5, filename="a.pdf", page_number=2)],
        }
    )
    reranker = ScriptedReranker(
        {"nothing useful": 0.001}, {"nothing useful": 0.001, "the real answer": 0.97}
    )
    retriever = _corrective_retriever(store, reranker, FakeQueryExpander(["bbbbb"]))

    chunks = retriever.retrieve("aaaa", top_k=5)

    assert chunks[0].chunk_id == "good"
    assert chunks[0].rerank_score == 0.97


def test_an_ambiguous_result_a_retry_cannot_improve_is_still_answered(monkeypatch):
    """
    Ambiguous means "not confident", not "nothing" - collapsing it into a
    refusal would turn the three bands back into a plain cutoff.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_accept_score", 0.5)
    monkeypatch.setattr(settings, "corrective_reject_score", 0.02)
    store = FakeVectorStore([_search_result("a", "weak", 0.5, filename="a.pdf", page_number=1)])
    reranker = ScriptedReranker({"weak": 0.3}, {"weak": 0.1})
    retriever = _corrective_retriever(store, reranker, FakeQueryExpander(["retry phrasing"]))

    assert [chunk.chunk_id for chunk in retriever.retrieve("a question", top_k=5)] == ["a"]


def test_multi_query_already_expanded_so_corrective_does_not_expand_twice(monkeypatch):
    """The variants are already in the pool; asking again would buy nothing."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "corrective_accept_score", 0.9)
    monkeypatch.setattr(settings, "corrective_reject_score", 0.02)

    class CountingExpander:
        def __init__(self):
            self.calls = 0

        def expand(self, question):
            self.calls += 1
            return []

    expander = CountingExpander()
    store = FakeVectorStore([_search_result("a", "middling", 0.5, filename="a.pdf", page_number=1)])
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        mode="dense",
        reranker=FakeReranker({"middling": 0.3}),
        query_expander=expander,
        multi_query=True,
        corrective=True,
    )

    retriever.retrieve("a question", top_k=5)

    assert expander.calls == 1


def test_without_corrective_a_bad_result_set_is_still_returned():
    """The whole difference: today's behaviour answers from whatever ranked highest."""
    store = FakeVectorStore([_search_result("a", "unrelated", 0.5, filename="a.pdf", page_number=1)])
    retriever = _dense_retriever(store, reranker=FakeReranker({"unrelated": 0.001}))

    assert [chunk.chunk_id for chunk in retriever.retrieve("a question", top_k=5)] == ["a"]
