from fastapi.testclient import TestClient

from app.main import app
from app.services.agent_service import AgentChatResult, get_agent_chat_service

client = TestClient(app)


class StubAgentChatService:
    """Stands in for the memory-aware AgentChatService the endpoint depends on."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def get_answer(
        self, message: str, conversation_id: str | None = None
    ) -> AgentChatResult:
        self.calls.append((message, conversation_id))
        return AgentChatResult(
            conversation_id=conversation_id or "generated-id",
            answer=f"stub answer for: {message}",
            tool_used="answer_directly",
        )


def test_ask_agent_returns_answer_and_tool_used():
    stub = StubAgentChatService()
    app.dependency_overrides[get_agent_chat_service] = lambda: stub
    try:
        response = client.post("/api/v1/agent/ask", json={"message": "hello"})
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "hello"
    assert body["answer"] == "stub answer for: hello"
    assert body["tool_used"] == "answer_directly"
    assert body["conversation_id"] == "generated-id"


def test_ask_agent_passes_conversation_id_through():
    """A conversation_id sent by the client must reach the service and come
    back unchanged, so the frontend can keep a thread on the same id."""
    stub = StubAgentChatService()
    app.dependency_overrides[get_agent_chat_service] = lambda: stub
    try:
        response = client.post(
            "/api/v1/agent/ask",
            json={"message": "follow-up", "conversation_id": "existing-id"},
        )
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "existing-id"
    assert stub.calls == [("follow-up", "existing-id")]


def test_ask_agent_rejects_empty_message():
    app.dependency_overrides[get_agent_chat_service] = lambda: StubAgentChatService()
    try:
        response = client.post("/api/v1/agent/ask", json={"message": ""})
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 422
