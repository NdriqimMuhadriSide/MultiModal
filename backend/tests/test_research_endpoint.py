"""
Tests for POST /research/ask - the HTTP layer only.

The loop itself is covered in test_research_agent.py; these pin the
contract the frontend reads, which is where the trace and the stop reason
have to survive the trip out.
"""
from fastapi.testclient import TestClient

from agents.research_agent import ResearchStep
from app.main import app
from app.schemas.rag import RAGChatSource
from app.services.research_service import (
    ResearchChatResult,
    get_research_chat_service,
)

client = TestClient(app)


class StubResearchChatService:
    """Stands in for the memory-aware ResearchChatService the endpoint depends on."""

    def __init__(self, result: ResearchChatResult | None = None) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._result = result

    def research(
        self, question: str, conversation_id: str | None = None
    ) -> ResearchChatResult:
        self.calls.append((question, conversation_id))
        if self._result is not None:
            return self._result
        return ResearchChatResult(
            conversation_id=conversation_id or "generated-id",
            answer=f"stub answer for: {question}",
        )


def _post(stub, body: dict):
    app.dependency_overrides[get_research_chat_service] = lambda: stub
    try:
        return client.post("/api/v1/research/ask", json=body)
    finally:
        app.dependency_overrides.pop(get_research_chat_service, None)


def test_research_returns_the_answer_and_conversation_id():
    stub = StubResearchChatService()
    response = _post(stub, {"question": "how do refunds and returns differ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub answer for: how do refunds and returns differ?"
    assert body["conversationId"] == "generated-id"
    assert stub.calls == [("how do refunds and returns differ?", None)]


def test_research_serialises_the_trace_and_citations():
    stub = StubResearchChatService(
        ResearchChatResult(
            conversation_id="conv-1",
            answer="Refunds run 14 days [E1].",
            steps=[
                ResearchStep(
                    thought="I should look up refunds.",
                    action_json='{"tool": "search", "input": {"query": "refunds"}}',
                    tool="search",
                    tool_input={"query": "refunds"},
                    observation="1 passage(s) ...",
                )
            ],
            sources=[
                RAGChatSource(
                    filename="handbook.pdf", page=3, chunk_id="c1", section="Refunds"
                )
            ],
            stopped_because="step_limit",
        )
    )
    body = _post(stub, {"question": "q", "conversation_id": "conv-1"}).json()

    assert body["stoppedBecause"] == "step_limit"
    assert body["steps"] == [
        {
            "thought": "I should look up refunds.",
            "tool": "search",
            "toolInput": '{"query": "refunds"}',
            "observation": "1 passage(s) ...",
            # Empty because this agent calls tools, not other agents. The
            # field is on every step regardless, so one renderer handles a
            # research trace and a delegating supervisor's alike.
            "children": [],
        }
    ]
    # Same citation shape POST /rag/chat and POST /agent/ask return.
    assert body["sources"] == [
        {"filename": "handbook.pdf", "page": 3, "chunkId": "c1", "section": "Refunds"}
    ]


def test_research_rejects_an_empty_question():
    response = _post(StubResearchChatService(), {"question": ""})
    assert response.status_code == 422


def test_research_maps_an_llm_failure_to_502():
    class FailingService(StubResearchChatService):
        def research(self, question, conversation_id=None):
            raise RuntimeError("LLM request failed: upstream timeout")

    response = _post(FailingService(), {"question": "q"})
    assert response.status_code == 502
    assert "upstream timeout" in response.json()["detail"]
