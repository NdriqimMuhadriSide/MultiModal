"""
Tests for POST /agent/ask/stream and AgentChatService.stream_answer.

This path differs from /chat/stream in what it streams. A supervisor's
answer is already whole inside its `finish` action by the time the loop sees
it, so there are no token deltas to forward - what arrives instead is one
frame per completed step, tagged with the depth that says whether the
supervisor took it or a specialist did.

The supervisor's own reasoning is tested in tests/test_supervisor_agent.py;
here the agent is faked so the frame contract and the memory flow are what
is under test.
"""
import json

from fastapi.testclient import TestClient

from agents.agent_loop import AgentStep, StepEvent
from agents.supervisor_agent import SupervisorResult, SupervisorResultEvent
from app.main import app
from app.services.agent_service import AgentChatService, get_agent_chat_service
from memory.attachment_store import AttachmentStore
from memory.conversation_memory import ConversationMemory
from rag.rag_service import RAGSource

client = TestClient(app)


def parse_events(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]


def step(tool: str = "finish", observation: str = "(final answer)") -> AgentStep:
    return AgentStep(
        thought="thinking",
        action_json=json.dumps({"tool": tool}),
        tool=tool,
        tool_input={},
        observation=observation,
    )


class StubAgentChatService(AgentChatService):
    """Yields fixed events without touching the supervisor, RAG, or SQLite."""

    def __init__(self, events: list[dict], fail_at: int | None = None) -> None:
        self._events = events
        self._fail_at = fail_at

    def stream_answer(self, message: str, conversation_id: str | None = None):
        resolved_id = conversation_id or "generated-conv-id"

        def gen():
            for index, event in enumerate(self._events):
                if self._fail_at is not None and index == self._fail_at:
                    raise RuntimeError("LLM stream failed: upstream closed")
                yield event

        return resolved_id, gen()


class FakeAgent:
    """
    Stands in for SupervisorAgent: emits scripted steps, then a result.

    Records what it was handed, because two of the things this service is
    responsible for - the history window and the conversation's image - are
    only observable from in here.
    """

    def __init__(
        self,
        tool_used: str = "answer_directly",
        answer: str = "the answer",
        steps: list[tuple[AgentStep, int]] | None = None,
        sources: list[RAGSource] | None = None,
        stopped_because: str = "finished",
    ) -> None:
        self.tool_used = tool_used
        self.answer = answer
        self.steps = steps or [(step(), 0)]
        self.sources = sources or []
        self.stopped_because = stopped_because
        self.received_history: list[dict[str, str]] | None = None
        self.received_image = None

    def stream(self, message, history=None, image=None):
        self.received_history = history
        self.received_image = image

        def gen():
            for agent_step, depth in self.steps:
                yield StepEvent(agent_step, depth=depth)
            yield SupervisorResultEvent(
                SupervisorResult(
                    answer=self.answer,
                    steps=[s for s, _ in self.steps],
                    sources=self.sources,
                    stopped_because=self.stopped_because,
                    tool_used=self.tool_used,
                )
            )

        return gen()


class NoAttachments(AttachmentStore):
    """An attachment store for conversations that never carried an image."""

    def __init__(self) -> None:
        pass

    def load(self, ref):
        return None


def _service(agent, memory, **kwargs) -> AgentChatService:
    return AgentChatService(
        agent=agent,
        memory=memory,
        attachments=NoAttachments(),
        history_limit=kwargs.pop("history_limit", 10),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Endpoint: the SSE frame contract
# ---------------------------------------------------------------------------


def test_agent_stream_emits_start_steps_tool_answer_then_done():
    stub = StubAgentChatService(
        [
            {"type": "step", "index": 1, "depth": 0, "step": {"tool": "finish"}},
            {"type": "tool", "tool": "answer_directly"},
            {"type": "answer", "content": "hello"},
            {"type": "done", "stopped_because": "finished"},
        ]
    )
    app.dependency_overrides[get_agent_chat_service] = lambda: stub
    try:
        response = client.post("/api/v1/agent/ask/stream", json={"message": "hi"})
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 200
    events = parse_events(response.text)
    assert [e["type"] for e in events] == ["start", "step", "tool", "answer", "done"]
    assert events[0]["conversation_id"] == "generated-conv-id"


def test_agent_stream_reports_midstream_failure_as_an_error_event():
    """
    By the time the model can fail the response has already committed as a
    200, so the failure has to arrive inside the stream rather than as a
    status code.
    """
    stub = StubAgentChatService(
        [
            {"type": "step", "index": 1, "depth": 0, "step": {"tool": "search"}},
            {"type": "answer", "content": "never sent"},
        ],
        fail_at=1,
    )
    app.dependency_overrides[get_agent_chat_service] = lambda: stub
    try:
        response = client.post("/api/v1/agent/ask/stream", json={"message": "hi"})
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 200
    events = parse_events(response.text)
    assert [e["type"] for e in events] == ["start", "step", "error"]
    assert "upstream closed" in events[-1]["detail"]


def test_agent_stream_rejects_empty_message():
    app.dependency_overrides[get_agent_chat_service] = lambda: StubAgentChatService([])
    try:
        response = client.post("/api/v1/agent/ask/stream", json={"message": ""})
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Service: frame order, depth, and the memory flow
# ---------------------------------------------------------------------------


def test_stream_answer_emits_steps_before_the_tool_and_the_answer(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = _service(FakeAgent(), memory)

    _, events = service.stream_answer("hello")

    assert [e["type"] for e in events] == ["step", "tool", "answer", "done"]


def test_stream_answer_tags_a_specialists_step_as_deeper_than_the_supervisors(tmp_path):
    """
    Depth is what lets the client indent a delegated step instead of
    presenting a specialist's search as something the supervisor did itself.
    """
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    agent = FakeAgent(
        tool_used="research_documents",
        steps=[
            (step("search", "1 passage(s)"), 1),
            (step("research_documents", "Refunds within 30 days."), 0),
            (step(), 0),
        ],
    )
    service = _service(agent, memory)

    _, events = service.stream_answer("refund window?")
    frames = [e for e in events if e["type"] == "step"]

    assert [f["depth"] for f in frames] == [1, 0, 0]
    assert [f["step"]["tool"] for f in frames] == [
        "search",
        "research_documents",
        "finish",
    ]


def test_stream_answer_emits_sources_after_the_tool_and_before_the_answer(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    agent = FakeAgent(
        tool_used="research_documents",
        sources=[
            RAGSource(
                chunk_id="handbook.pdf::p3::c1",
                filename="handbook.pdf",
                page=3,
                score=0.81,
                section="4. Leave",
            )
        ],
    )
    service = _service(agent, memory)

    _, events = service.stream_answer("what does my handbook say about leave?")
    emitted = list(events)

    assert [e["type"] for e in emitted] == ["step", "tool", "sources", "answer", "done"]
    # camelCase chunkId, exactly as POST /rag/chat and POST /agent/ask return
    # it - one citation shape across all three, one frontend type.
    assert emitted[2]["sources"] == [
        {
            "filename": "handbook.pdf",
            "page": 3,
            "chunkId": "handbook.pdf::p3::c1",
            "section": "4. Leave",
        }
    ]


def test_stream_answer_emits_no_sources_event_when_there_is_nothing_to_cite(tmp_path):
    """
    An empty event would make the client tell "no citations" apart from
    "citations not sent yet" for no gain. Absent means neither.
    """
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = _service(FakeAgent(), memory)

    _, events = service.stream_answer("what is the capital of France?")

    assert not any(e["type"] == "sources" for e in events)


def test_stream_answer_persists_the_answer(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = _service(FakeAgent(answer="It is 42."), memory)

    conversation_id, events = service.stream_answer("meaning of life?")
    assert [m.role for m in memory.get_full_history(conversation_id)] == ["user"]

    list(events)

    stored = memory.get_full_history(conversation_id)
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[1].content == "It is 42."


def test_stream_answer_stores_nothing_when_the_client_leaves_before_the_answer(tmp_path):
    """
    A behaviour change worth pinning down. While this endpoint streamed
    tokens, a tab closed mid-answer left half the text worth keeping. A
    supervisor produces its answer in one frame, so leaving before that frame
    means there is genuinely nothing to store - and storing an empty
    assistant turn would put a blank bubble in the user's history.
    """
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    agent = FakeAgent(steps=[(step("search"), 1), (step(), 0)])
    service = _service(agent, memory)

    conversation_id, events = service.stream_answer("hi")
    next(events)  # the first step frame
    events.close()  # tab closed before the answer arrived

    assert [m.role for m in memory.get_full_history(conversation_id)] == ["user"]


def test_stream_answer_passes_prior_turns_to_the_agent(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    memory.add_message("conv-1", role="user", content="earlier question")
    memory.add_message("conv-1", role="assistant", content="earlier answer")

    agent = FakeAgent()
    service = _service(agent, memory)

    _, events = service.stream_answer("follow up", conversation_id="conv-1")
    list(events)

    assert agent.received_history == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]


def test_stream_answer_passes_no_image_when_the_conversation_has_never_carried_one(
    tmp_path,
):
    """
    The ordinary text conversation. `read_image` reports there is no picture
    rather than the run failing, which is what keeps a supervisor usable from
    a JSON endpoint.
    """
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    agent = FakeAgent()
    service = _service(agent, memory)

    _, events = service.stream_answer("hello")
    list(events)

    assert agent.received_image is None
