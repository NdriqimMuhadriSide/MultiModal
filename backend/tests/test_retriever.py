import pytest

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


def _search_result(chunk_id: str, text: str, similarity: float, **metadata) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, metadata=metadata, similarity=similarity)


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
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(results),
    )

    chunks = retriever.retrieve("What is our refund policy?")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "policy.pdf::p1::c0"
    assert chunks[0].filename == "policy.pdf"
    assert chunks[0].page == 1
    assert chunks[0].score == 0.8
    assert "Refunds are issued" in chunks[0].text


def test_retrieve_filters_out_low_similarity_chunks():
    results = [
        _search_result("a", "irrelevant chunk", similarity=0.05, filename="doc.pdf", page_number=1),
        _search_result("b", "relevant chunk", similarity=0.9, filename="doc.pdf", page_number=2),
    ]
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(results),
        min_similarity=0.2,
    )

    chunks = retriever.retrieve("a question")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "b"


def test_retrieve_passes_top_k_through_to_vector_store():
    store = FakeVectorStore([])
    retriever = Retriever(embedding_service=FakeEmbeddingService(), vector_store=store)

    retriever.retrieve("a question", top_k=3)

    assert store.last_top_k == 3


def test_retrieve_defaults_to_top_5():
    store = FakeVectorStore([])
    retriever = Retriever(embedding_service=FakeEmbeddingService(), vector_store=store)

    retriever.retrieve("a question")

    assert store.last_top_k == 5


def test_retrieve_rejects_empty_question():
    retriever = Retriever(embedding_service=FakeEmbeddingService(), vector_store=FakeVectorStore([]))

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_retrieve_returns_empty_list_when_store_has_no_results():
    retriever = Retriever(embedding_service=FakeEmbeddingService(), vector_store=FakeVectorStore([]))

    assert retriever.retrieve("anything") == []
