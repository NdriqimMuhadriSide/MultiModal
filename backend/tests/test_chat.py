from fastapi.testclient import TestClient

from app.main import app
from app.services.chat_service import ChatResult, ChatService, get_chat_service
from memory.conversation_memory import get_conversation_memory

client = TestClient(app)


class StubChatService(ChatService):
    """Bypasses the real LLM service and real memory so tests don't call Groq."""

    def __init__(self) -> None:  # no super().__init__ - no real LLM/memory needed
        self.calls: list[tuple[str, str | None]] = []

    def get_answer(self, message: str, conversation_id: str | None = None) -> ChatResult:
        self.calls.append((message, conversation_id))
        resolved_id = conversation_id or "generated-conv-id"
        return ChatResult(conversation_id=resolved_id, answer=f"stub answer for: {message}")


def test_chat_returns_answer_and_generates_conversation_id_when_absent():
    app.dependency_overrides[get_chat_service] = lambda: StubChatService()
    try:
        response = client.post("/api/v1/chat", json={"message": "Explain RAG"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub answer for: Explain RAG"
    assert body["conversation_id"] == "generated-conv-id"


def test_chat_reuses_provided_conversation_id():
    stub = StubChatService()
    app.dependency_overrides[get_chat_service] = lambda: stub
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "And what about vector DBs?", "conversation_id": "abc-123"},
        )
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "abc-123"
    assert stub.calls == [("And what about vector DBs?", "abc-123")]


def test_chat_rejects_empty_message():
    app.dependency_overrides[get_chat_service] = lambda: StubChatService()
    try:
        response = client.post("/api/v1/chat", json={"message": ""})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 422


def test_get_conversation_history_returns_stored_messages(tmp_path):
    from memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    memory.add_message("conv-1", role="user", content="hello")
    memory.add_message("conv-1", role="assistant", content="hi there")

    app.dependency_overrides[get_conversation_memory] = lambda: memory
    try:
        response = client.get("/api/v1/chat/conv-1/history")
    finally:
        app.dependency_overrides.pop(get_conversation_memory, None)

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv-1"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "hello"
    assert body["messages"][1]["content"] == "hi there"
