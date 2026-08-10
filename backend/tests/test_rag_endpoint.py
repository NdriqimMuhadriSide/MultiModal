from fastapi.testclient import TestClient

from app.main import app
from app.services.rag_service import get_rag_service
from rag.rag_service import RAGAnswer, RAGSource

client = TestClient(app)


class StubRAGService:
    """Bypasses embedding/Chroma/LLM so tests don't touch real services."""

    def ask(self, question: str, top_k: int = 5) -> RAGAnswer:
        return RAGAnswer(
            answer=f"stub answer for: {question}",
            sources=[
                RAGSource(
                    chunk_id="policy.pdf::p1::c0",
                    filename="policy.pdf",
                    page=1,
                    score=0.75,
                    dense_score=0.75,
                    matched_by="dense",
                )
            ],
        )


def test_ask_rag_returns_answer_and_sources():
    app.dependency_overrides[get_rag_service] = lambda: StubRAGService()
    try:
        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "What is our refund policy?", "top_k": 3},
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is our refund policy?"
    assert body["answer"] == "stub answer for: What is our refund policy?"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["chunk_id"] == "policy.pdf::p1::c0"
    assert body["sources"][0]["similarity"] == 0.75
    assert body["sources"][0]["score"] == 0.75
    assert body["sources"][0]["matched_by"] == "dense"


def test_ask_rag_rejects_empty_question():
    app.dependency_overrides[get_rag_service] = lambda: StubRAGService()
    try:
        response = client.post("/api/v1/rag/ask", json={"question": ""})
    finally:
        app.dependency_overrides.pop(get_rag_service, None)

    assert response.status_code == 422


def test_chat_rag_returns_answer_and_camel_case_sources():
    app.dependency_overrides[get_rag_service] = lambda: StubRAGService()
    try:
        response = client.post(
            "/api/v1/rag/chat",
            json={"question": "What is our refund policy?"},
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub answer for: What is our refund policy?"
    assert len(body["sources"]) == 1
    source = body["sources"][0]
    assert set(source.keys()) == {"filename", "page", "chunkId", "section"}
    assert source["filename"] == "policy.pdf"
    assert source["page"] == 1
    assert source["chunkId"] == "policy.pdf::p1::c0"


def test_chat_rag_rejects_empty_question():
    app.dependency_overrides[get_rag_service] = lambda: StubRAGService()
    try:
        response = client.post("/api/v1/rag/chat", json={"question": ""})
    finally:
        app.dependency_overrides.pop(get_rag_service, None)

    assert response.status_code == 422


def test_chat_rag_maps_runtime_error_to_bad_gateway():
    class FailingRAGService:
        def ask(self, question: str, top_k: int = 5):
            raise RuntimeError("LLM request failed: connection refused")

    app.dependency_overrides[get_rag_service] = lambda: FailingRAGService()
    try:
        response = client.post("/api/v1/rag/chat", json={"question": "anything"})
    finally:
        app.dependency_overrides.pop(get_rag_service, None)

    assert response.status_code == 502
