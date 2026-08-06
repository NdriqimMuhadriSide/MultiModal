import pytest

from rag.rag_service import RAGService
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

    def search(self, query_embedding, top_k: int = 5):
        return self._results[:top_k]


class FakeLLMService:
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def generate_response(self, user_message: str) -> str:
        self.last_prompt = user_message
        return "Refunds are allowed within 30 days."


def _search_result(chunk_id: str, text: str, similarity: float, **metadata) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, metadata=metadata, similarity=similarity)


def _rag_service(results: list[SearchResult], llm: FakeLLMService, min_similarity: float = 0.2) -> RAGService:
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(results),
        min_similarity=min_similarity,
    )
    return RAGService(retriever=retriever, llm_service=llm)


def test_ask_builds_context_and_returns_answer_with_sources():
    results = [
        _search_result(
            "policy.pdf::p1::c0",
            "Refunds are issued within 30 days of purchase.",
            similarity=0.8,
            filename="policy.pdf",
            page_number=1,
        )
    ]
    llm = FakeLLMService()
    service = _rag_service(results, llm)

    result = service.ask("What is our refund policy?", top_k=3)

    assert result.answer == "Refunds are allowed within 30 days."
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "policy.pdf::p1::c0"
    assert result.sources[0].filename == "policy.pdf"
    assert result.sources[0].score == 0.8

    # The prompt sent to the LLM should contain the context and the question,
    # per the required RAG_PROMPT_TEMPLATE.
    assert llm.last_prompt is not None
    assert "Refunds are issued within 30 days" in llm.last_prompt
    assert "What is our refund policy?" in llm.last_prompt
    assert "say that you do not know" in llm.last_prompt


def test_ask_filters_out_low_similarity_results():
    results = [
        _search_result("a", "irrelevant chunk", similarity=0.05, filename="doc.pdf", page_number=1),
    ]
    llm = FakeLLMService()
    service = _rag_service(results, llm, min_similarity=0.2)

    result = service.ask("unrelated question")

    # Below min_similarity -> treated as "no relevant context" -> no LLM call.
    assert llm.last_prompt is None
    assert result.sources == []
    assert "don't know" in result.answer.lower()


def test_ask_returns_dont_know_when_store_is_empty():
    llm = FakeLLMService()
    service = _rag_service([], llm)

    result = service.ask("anything")

    assert llm.last_prompt is None
    assert result.sources == []
    assert "don't know" in result.answer.lower()


def test_ask_rejects_empty_question():
    service = _rag_service([], FakeLLMService())

    with pytest.raises(ValueError):
        service.ask("")


def test_ask_defaults_to_top_5_chunks():
    results = [
        _search_result(f"c{i}", f"chunk {i}", similarity=0.9, filename="doc.pdf", page_number=i)
        for i in range(10)
    ]
    llm = FakeLLMService()
    service = _rag_service(results, llm)

    result = service.ask("a question")

    assert len(result.sources) == 5
