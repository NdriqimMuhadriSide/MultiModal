"""
Tests for POST /agent/ask - the supervisor's non-streaming endpoint.

The endpoint is a pure HTTP layer, so the service is stubbed throughout:
what is under test is the wire shape, not the supervisor's reasoning (that
is tests/test_supervisor_agent.py).
"""
from fastapi.testclient import TestClient

from agents.agent_loop import AgentStep
from app.main import app
from app.schemas.rag import RAGChatSource
from app.services.agent_service import AgentChatResult, get_agent_chat_service

client = TestClient(app)


class StubAgentChatService:
    """Stands in for the memory-aware AgentChatService the endpoint depends on."""

    def __init__(
        self,
        tool: str = "answer_directly",
        sources=None,
        steps=None,
        stopped_because: str = "finished",
    ) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.tool = tool
        self.sources = sources or []
        self.steps = steps or []
        self.stopped_because = stopped_because

    def get_answer(
        self, message: str, conversation_id: str | None = None
    ) -> AgentChatResult:
        self.calls.append((message, conversation_id))
        return AgentChatResult(
            conversation_id=conversation_id or "generated-id",
            answer=f"stub answer for: {message}",
            tool_used=self.tool,
            sources=self.sources,
            steps=self.steps,
            stopped_because=self.stopped_because,
        )


def _post(stub, payload):
    app.dependency_overrides[get_agent_chat_service] = lambda: stub
    try:
        return client.post("/api/v1/agent/ask", json=payload)
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)


def test_ask_agent_returns_answer_and_tool_used():
    stub = StubAgentChatService()
    response = _post(stub, {"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "hello"
    assert body["answer"] == "stub answer for: hello"
    assert body["tool_used"] == "answer_directly"
    assert body["conversation_id"] == "generated-id"
    assert body["stopped_because"] == "finished"


def test_ask_agent_passes_conversation_id_through():
    """A conversation_id sent by the client must reach the service and come
    back unchanged, so the frontend can keep a thread on the same id."""
    stub = StubAgentChatService()
    response = _post(stub, {"message": "follow-up", "conversation_id": "existing-id"})

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "existing-id"
    assert stub.calls == [("follow-up", "existing-id")]


def test_ask_agent_returns_citations_for_a_delegated_document_answer():
    """
    Same wire shape POST /rag/chat returns - camelCase chunkId included - so
    the frontend renders a supervisor citation with the same component and
    type it already uses for a document answer.
    """
    stub = StubAgentChatService(
        tool="research_documents",
        sources=[
            RAGChatSource(
                filename="handbook.pdf",
                page=3,
                chunk_id="handbook.pdf::p3::c1",
                section="4. Leave",
            )
        ],
    )
    response = _post(stub, {"message": "leave policy?"})

    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) == 1
    assert set(sources[0].keys()) == {"filename", "page", "chunkId", "section"}
    assert sources[0]["chunkId"] == "handbook.pdf::p3::c1"


def test_ask_agent_returns_no_citations_for_a_tool_with_nothing_to_cite():
    stub = StubAgentChatService()
    assert _post(stub, {"message": "hello"}).json()["sources"] == []


def test_ask_agent_nests_a_delegated_specialists_steps_under_the_delegation():
    """
    The whole point of the trace under delegation: a reader can see not just
    that the supervisor asked a specialist, but what the specialist did. A
    flat list would report the delegation and silently drop the four steps
    behind its answer.
    """
    stub = StubAgentChatService(
        tool="research_documents",
        steps=[
            AgentStep(
                thought="The documents will know.",
                action_json='{"tool": "research_documents"}',
                tool="research_documents",
                tool_input={"question": "What is the refund window?"},
                observation="Refunds are accepted within 30 days [E1].",
                children=[
                    AgentStep(
                        thought="Search for it.",
                        action_json='{"tool": "search"}',
                        tool="search",
                        tool_input={"query": "refund window"},
                        observation="1 passage(s) ...",
                    )
                ],
            )
        ],
    )
    response = _post(stub, {"message": "refund window?"})

    steps = response.json()["steps"]
    assert len(steps) == 1
    assert steps[0]["tool"] == "research_documents"
    assert steps[0]["toolInput"] == '{"question": "What is the refund window?"}'

    children = steps[0]["children"]
    assert len(children) == 1
    assert children[0]["tool"] == "search"
    # Recursion bottoms out rather than being truncated at one level.
    assert children[0]["children"] == []


def test_ask_agent_reports_a_run_that_ran_out_of_budget():
    """
    An answer written because the tree's shared budget ran out is not the
    same as one written because the work was done, and the reader is told
    which they are looking at.
    """
    stub = StubAgentChatService(stopped_because="step_limit")
    assert _post(stub, {"message": "hi"}).json()["stopped_because"] == "step_limit"


def test_ask_agent_rejects_empty_message():
    assert _post(StubAgentChatService(), {"message": ""}).status_code == 422
