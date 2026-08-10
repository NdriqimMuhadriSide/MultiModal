"""
Tests for POST /agent/ask/stream, AgentChatService.stream_answer, and
AssistantAgent.stream.

The agent path differs from /chat/stream in two ways worth covering
directly: the router's choice arrives as its own event before any text,
and one of the three tools (the external API) produces its answer without
an LLM and so cannot stream at all.
"""
import json

from fastapi.testclient import TestClient

from app.main import app
from app.services.agent_service import AgentChatService, get_agent_chat_service
from memory.conversation_memory import ConversationMemory

client = TestClient(app)


def parse_events(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]


class StubAgentChatService(AgentChatService):
    """Yields fixed events without touching the router, RAG, or SQLite."""

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
    """Stands in for AssistantAgent, returning a tool and fixed chunks."""

    def __init__(self, tool: str, chunks: list[str]) -> None:
        self.tool = tool
        self.chunks = chunks
        self.received_history = None

    def stream(self, message: str, history=None):
        self.received_history = history

        def gen():
            yield from self.chunks

        return self.tool, gen()


# ---------------------------------------------------------------------------
# Endpoint: SSE framing
# ---------------------------------------------------------------------------


def test_agent_stream_emits_start_tool_deltas_then_done():
    app.dependency_overrides[get_agent_chat_service] = lambda: StubAgentChatService(
        [
            {"type": "tool", "tool": "search_knowledge_base"},
            {"type": "delta", "content": "According to"},
            {"type": "delta", "content": " the docs..."},
        ]
    )
    try:
        response = client.post("/api/v1/agent/ask/stream", json={"message": "what do my docs say?"})
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_events(response.text)
    assert events[0] == {"type": "start", "conversation_id": "generated-conv-id"}
    # The tool must be known before any text, so the bubble can be labelled
    # while the answer is still being written.
    assert events[1] == {"type": "tool", "tool": "search_knowledge_base"}
    assert [e["content"] for e in events if e["type"] == "delta"] == [
        "According to",
        " the docs...",
    ]
    assert events[-1] == {"type": "done"}


def test_agent_stream_reports_midstream_failure_as_an_error_event():
    app.dependency_overrides[get_agent_chat_service] = lambda: StubAgentChatService(
        [
            {"type": "tool", "tool": "answer_directly"},
            {"type": "delta", "content": "half an answer"},
            {"type": "delta", "content": "never sent"},
        ],
        fail_at=2,
    )
    try:
        response = client.post("/api/v1/agent/ask/stream", json={"message": "hi"})
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 200
    events = parse_events(response.text)
    assert events[-1]["type"] == "error"
    assert "upstream closed" in events[-1]["detail"]
    assert not any(e["type"] == "done" for e in events)


def test_agent_stream_rejects_empty_message():
    app.dependency_overrides[get_agent_chat_service] = lambda: StubAgentChatService([])
    try:
        response = client.post("/api/v1/agent/ask/stream", json={"message": ""})
    finally:
        app.dependency_overrides.pop(get_agent_chat_service, None)

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Service: memory-write timing and event ordering
# ---------------------------------------------------------------------------


def test_stream_answer_emits_the_tool_before_any_delta(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = AgentChatService(
        agent=FakeAgent("answer_directly", ["a", "b"]), memory=memory, history_limit=10
    )

    _, events = service.stream_answer("hello")
    kinds = [e["type"] for e in events]
    assert kinds == ["tool", "delta", "delta"]


def test_stream_answer_persists_the_joined_answer(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = AgentChatService(
        agent=FakeAgent("answer_directly", ["It ", "is ", "42."]),
        memory=memory,
        history_limit=10,
    )

    conversation_id, events = service.stream_answer("meaning of life?")
    assert [m.role for m in memory.get_full_history(conversation_id)] == ["user"]

    list(events)

    stored = memory.get_full_history(conversation_id)
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[1].content == "It is 42."


def test_stream_answer_persists_partial_text_when_the_client_disconnects(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = AgentChatService(
        agent=FakeAgent("answer_directly", ["first ", "second ", "third"]),
        memory=memory,
        history_limit=10,
    )

    conversation_id, events = service.stream_answer("hi")
    next(events)  # the tool event
    next(events)  # one delta
    events.close()  # tab closed mid-answer

    stored = memory.get_full_history(conversation_id)
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[1].content == "first "


def test_stream_answer_passes_prior_turns_to_the_agent(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    memory.add_message("conv-1", role="user", content="earlier question")
    memory.add_message("conv-1", role="assistant", content="earlier answer")

    agent = FakeAgent("answer_directly", ["ok"])
    service = AgentChatService(agent=agent, memory=memory, history_limit=10)

    _, events = service.stream_answer("follow up", conversation_id="conv-1")
    list(events)

    assert agent.received_history == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]


# ---------------------------------------------------------------------------
# Agent: routing and the branch that cannot stream
# ---------------------------------------------------------------------------


class RoutingLLM:
    """Returns a fixed routing decision, then streams a fixed answer."""

    def __init__(self, choice: str, chunks: list[str]) -> None:
        self.choice = choice
        self.chunks = chunks

    def generate_response(self, prompt: str, history=None) -> str:
        return self.choice

    def stream_response(self, message: str, history=None):
        return iter(self.chunks)


def test_agent_stream_direct_answer_streams_token_by_token():
    from agents.assistant_agent import AssistantAgent

    agent = AssistantAgent(
        llm_service=RoutingLLM("DIRECT_ANSWER", ["one", "two", "three"]),
        rag_service=None,
    )

    tool, chunks = agent.stream("hello")
    assert tool == "answer_directly"
    assert list(chunks) == ["one", "two", "three"]


def test_agent_stream_knowledge_base_delegates_to_rag_streaming():
    from agents.assistant_agent import AssistantAgent

    from rag.rag_service import RAGSource

    class FakeRag:
        def __init__(self, sources):
            self.sources = sources
            self.history_seen = None

        def stream_ask(self, question, top_k=None, history=None):
            self.history_seen = history
            return self.sources, iter(["from ", "the docs"])

    grounded = [RAGSource(chunk_id="c0", filename="d.pdf", page=1, score=0.9)]
    agent = AssistantAgent(
        llm_service=RoutingLLM("KNOWLEDGE_BASE", []), rag_service=FakeRag(grounded)
    )

    tool, chunks = agent.stream("what do my docs say?")
    assert tool == "search_knowledge_base"
    assert list(chunks) == ["from ", "the docs"]


def test_agent_stream_falls_back_when_the_knowledge_base_is_empty():
    """
    The streaming path does not run the compiled graph, so it has to make the
    same second-hop decision itself. If these two ever disagree, the UI (which
    uses the streaming endpoint) behaves differently from every test of the
    graph.
    """
    from agents.assistant_agent import AssistantAgent
    from rag.rag_service import RAGSource

    class EmptyRag:
        def stream_ask(self, question, top_k=None, history=None):
            return [], iter(["should not be used"])

    llm = RoutingLLM("KNOWLEDGE_BASE", ["general ", "knowledge"])
    agent = AssistantAgent(llm_service=llm, rag_service=EmptyRag())

    tool, chunks = agent.stream("something the docs do not cover")

    assert tool == "answer_directly_after_knowledge_base"
    assert list(chunks) == ["general ", "knowledge"]


def test_agent_stream_hands_history_to_retrieval():
    """Retrieval can only resolve a follow-up if it is given the turns before it."""
    from agents.assistant_agent import AssistantAgent
    from rag.rag_service import RAGSource

    class FakeRag:
        def __init__(self):
            self.history_seen = None

        def stream_ask(self, question, top_k=None, history=None):
            self.history_seen = history
            return [RAGSource(chunk_id="c", filename="d.pdf", page=1, score=0.9)], iter(["x"])

    rag = FakeRag()
    agent = AssistantAgent(llm_service=RoutingLLM("KNOWLEDGE_BASE", []), rag_service=rag)
    history = [{"role": "user", "content": "how much leave do I accrue?"}]

    list(agent.stream("how far ahead do I request it?", history=history)[1])

    assert rag.history_seen == history


def test_agent_stream_external_api_yields_once(monkeypatch):
    """
    The weather tool builds its sentence from a JSON response with no LLM
    involved, so there is nothing to arrive incrementally. One chunk is the
    honest result, not a bug.
    """
    from agents import assistant_agent
    from agents.assistant_agent import AssistantAgent

    monkeypatch.setattr(assistant_agent, "_fetch_weather", lambda city: f"It is sunny in {city}.")

    agent = AssistantAgent(
        llm_service=RoutingLLM("EXTERNAL_API", []), rag_service=None
    )

    tool, chunks = agent.stream("What is the weather in Paris?")
    assert tool == "call_external_api"
    assert list(chunks) == ["It is sunny in Paris."]


def test_agent_stream_rejects_empty_message():
    import pytest

    from agents.assistant_agent import AssistantAgent

    agent = AssistantAgent(llm_service=RoutingLLM("DIRECT_ANSWER", []), rag_service=None)

    with pytest.raises(ValueError):
        agent.stream("   ")
