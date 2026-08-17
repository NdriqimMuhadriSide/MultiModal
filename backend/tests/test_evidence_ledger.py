"""
Tests for the evidence ledger and the search tool that writes to it
(agents/knowledge_base_tool.py).

The label is a promise: [E3] in a final answer has to mean the same passage
it meant when the model read it three steps earlier. One agent numbering
alone keeps that promise easily. Two agents in a tree only keep it if they
share a ledger, and the failure when they do not is silent - the answer
cites a real label pointing at the wrong text.
"""
from agents.knowledge_base_tool import EvidenceLedger, KnowledgeBaseSearch
from rag.retriever import RetrievedChunk


def chunk(chunk_id: str, text: str = "some text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        filename="doc.pdf",
        page=1,
        score=0.9,
        section="S",
    )


class FakeRetriever:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def retrieve(self, query, top_k=5):
        self.queries.append(query)
        return self.results.pop(0) if self.results else []


# ---- The ledger ------------------------------------------------------------


def test_a_passage_keeps_its_label_however_often_it_comes_back():
    ledger = EvidenceLedger()

    assert ledger.record(chunk("a")) == ("E1", True)
    assert ledger.record(chunk("b")) == ("E2", True)
    assert ledger.record(chunk("a")) == ("E1", False)


def test_sources_are_index_aligned_with_their_labels():
    ledger = EvidenceLedger()
    ledger.record(chunk("a"))
    ledger.record(chunk("b"))

    # index i is the passage labelled [E(i+1)] - the invariant every citation
    # in every answer depends on.
    assert [source.chunk_id for source in ledger.sources()] == ["a", "b"]


def test_reset_clears_in_place_so_shared_holders_see_it():
    """
    In place rather than rebound. Every agent in a tree holds this same
    object; replacing it would leave them numbering into the previous run's.
    """
    ledger = EvidenceLedger()
    ledger.record(chunk("a"))
    held_elsewhere = ledger

    ledger.reset()

    assert held_elsewhere.sources() == []
    assert ledger.record(chunk("b")) == ("E1", True)


# ---- Sharing between searchers ---------------------------------------------


def test_two_searchers_sharing_a_ledger_never_reuse_a_label():
    """
    The bug this exists to prevent. Separate ledgers give both searchers an
    [E1] for different passages, and a supervisor merging their answers ends
    up with a citation list where half the labels point at the wrong text.
    """
    ledger = EvidenceLedger()
    first = KnowledgeBaseSearch(
        FakeRetriever([[chunk("limits-c1", "The limit is 50.")]]), ledger=ledger
    )
    second = KnowledgeBaseSearch(
        FakeRetriever([[chunk("approval-c1", "Approval is needed.")]]), ledger=ledger
    )

    first_output = first.run({"query": "limit"})
    second_output = second.run({"query": "approval"})

    assert "[E1]" in first_output
    assert "[E2]" in second_output
    assert [source.chunk_id for source in first.sources()] == [
        "limits-c1",
        "approval-c1",
    ]


def test_a_searcher_that_borrows_a_ledger_does_not_clear_it_on_reset():
    """
    A specialist resets its own per-run state at the top of every run. If
    that wiped the ledger it borrowed, it would erase the labels the
    supervisor had already handed out before delegating.
    """
    ledger = EvidenceLedger()
    owner_view = KnowledgeBaseSearch(FakeRetriever([]), ledger=ledger)
    borrower = KnowledgeBaseSearch(
        FakeRetriever([[chunk("c1")]]), ledger=ledger
    )
    borrower.run({"query": "something"})

    borrower.reset()

    assert [source.chunk_id for source in owner_view.sources()] == ["c1"]


def test_a_searcher_that_owns_its_ledger_does_clear_it_on_reset():
    search = KnowledgeBaseSearch(FakeRetriever([[chunk("c1")]]))
    search.run({"query": "something"})

    search.reset()

    assert search.sources() == []


def test_the_query_cache_is_cleared_on_reset_even_for_a_borrowed_ledger():
    """
    The cache stops one agent paying twice for the same retrieval within its
    own run. Two agents in a tree asking the same thing is a repeat worth
    answering - they each need the passages in their own scratchpad.
    """
    retriever = FakeRetriever([[chunk("c1")], [chunk("c1")]])
    search = KnowledgeBaseSearch(retriever, ledger=EvidenceLedger())

    search.run({"query": "refund window"})
    search.reset()
    search.run({"query": "refund window"})

    assert retriever.queries == ["refund window", "refund window"]


def test_an_already_shown_passage_is_pointed_at_rather_than_repeated():
    """
    Every observation is re-sent on every later step, so a passage shown
    twice is paid for on every turn after that.
    """
    ledger = EvidenceLedger()
    search = KnowledgeBaseSearch(
        FakeRetriever(
            [[chunk("c1", "The full text of the passage.")], [chunk("c1", "The full text of the passage.")]]
        ),
        ledger=ledger,
    )

    search.run({"query": "first phrasing"})
    second = search.run({"query": "second phrasing"})

    assert "already shown above, not repeated" in second
    assert "The full text of the passage." not in second
