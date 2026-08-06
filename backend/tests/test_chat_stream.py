"""
Tests for POST /chat/stream and ChatService.stream_answer.

Two layers are covered separately, because they can fail independently:
the endpoint's SSE framing, and the service's memory-write timing (which is
the thing streaming actually changed - the assistant turn is now persisted
after the last token rather than before the response is sent).
"""
import json

from fastapi.testclient import TestClient

from app.main import app
from app.services.chat_service import ChatService, get_chat_service
from memory.conversation_memory import ConversationMemory

client = TestClient(app)


def parse_events(body: str) -> list[dict]:
    """Pull the JSON payloads out of an SSE body, in order."""
    return [
        json.loads(line[len("data: ") :])
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]


class StubStreamingChatService(ChatService):
    """Yields a fixed set of chunks without touching Groq or SQLite."""

    def __init__(self, chunks: list[str], fail_after: int | None = None) -> None:
        self._chunks = chunks
        self._fail_after = fail_after

    def stream_answer(self, message: str, conversation_id: str | None = None):
        resolved_id = conversation_id or "generated-conv-id"

        def gen():
            for index, chunk in enumerate(self._chunks):
                if self._fail_after is not None and index == self._fail_after:
                    raise RuntimeError("LLM stream failed: connection reset")
                yield chunk

        return resolved_id, gen()


class FakeStreamingLLM:
    """Stands in for LLMService, yielding chunks and optionally failing."""

    def __init__(self, chunks: list[str], fail_after: int | None = None) -> None:
        self.chunks = chunks
        self.fail_after = fail_after
        self.received_history: list[dict[str, str]] | None = None

    def stream_response(self, user_message: str, history=None):
        self.received_history = history

        def gen():
            for index, chunk in enumerate(self.chunks):
                if self.fail_after is not None and index == self.fail_after:
                    raise RuntimeError("LLM stream failed: boom")
                yield chunk

        return gen()


# ---------------------------------------------------------------------------
# Endpoint: SSE framing
# ---------------------------------------------------------------------------


def test_stream_emits_start_then_deltas_then_done():
    app.dependency_overrides[get_chat_service] = lambda: StubStreamingChatService(
        ["Hello", " there", "!"]
    )
    try:
        response = client.post("/api/v1/chat/stream", json={"message": "hi"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_events(response.text)
    assert events[0] == {"type": "start", "conversation_id": "generated-conv-id"}
    assert [e["content"] for e in events if e["type"] == "delta"] == [
        "Hello",
        " there",
        "!",
    ]
    assert events[-1] == {"type": "done"}


def test_stream_reuses_provided_conversation_id():
    app.dependency_overrides[get_chat_service] = lambda: StubStreamingChatService(["ok"])
    try:
        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "follow up", "conversation_id": "abc-123"},
        )
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert parse_events(response.text)[0]["conversation_id"] == "abc-123"


def test_stream_reports_midstream_failure_as_an_error_event():
    """
    The status line is already 200 by the time the provider dies, so the
    failure has to arrive as an event. A `done` must not follow it.
    """
    app.dependency_overrides[get_chat_service] = lambda: StubStreamingChatService(
        ["partial answer", "never sent"], fail_after=1
    )
    try:
        response = client.post("/api/v1/chat/stream", json={"message": "hi"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 200
    events = parse_events(response.text)
    assert events[1] == {"type": "delta", "content": "partial answer"}
    assert events[-1]["type"] == "error"
    assert "connection reset" in events[-1]["detail"]
    assert not any(e["type"] == "done" for e in events)


def test_stream_escapes_newlines_so_one_token_stays_one_event():
    """A raw newline in a token would split the SSE frame into two."""
    app.dependency_overrides[get_chat_service] = lambda: StubStreamingChatService(
        ["line one\nline two"]
    )
    try:
        response = client.post("/api/v1/chat/stream", json={"message": "hi"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    deltas = [e for e in parse_events(response.text) if e["type"] == "delta"]
    assert len(deltas) == 1
    assert deltas[0]["content"] == "line one\nline two"


def test_stream_rejects_empty_message():
    app.dependency_overrides[get_chat_service] = lambda: StubStreamingChatService(["x"])
    try:
        response = client.post("/api/v1/chat/stream", json={"message": ""})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Service: memory-write timing
# ---------------------------------------------------------------------------


def test_stream_answer_persists_the_full_answer_after_the_last_chunk(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = ChatService(
        llm_service=FakeStreamingLLM(["Paris", " is", " the", " capital."]),
        memory=memory,
        history_limit=10,
    )

    conversation_id, chunks = service.stream_answer("capital of France?")

    # The user's turn is stored eagerly; the assistant's is not there yet.
    assert [m.role for m in memory.get_full_history(conversation_id)] == ["user"]

    assert "".join(chunks) == "Paris is the capital."

    stored = memory.get_full_history(conversation_id)
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[1].content == "Paris is the capital."


def test_stream_answer_persists_partial_text_when_the_client_disconnects(tmp_path):
    """
    Closing the generator is what a browser tab closing mid-answer looks
    like from here. The half the user actually saw has to survive, or the
    next turn's context disagrees with the screen.
    """
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = ChatService(
        llm_service=FakeStreamingLLM(["The answer", " is", " forty-two."]),
        memory=memory,
        history_limit=10,
    )

    conversation_id, chunks = service.stream_answer("meaning of life?")
    assert next(chunks) == "The answer"
    chunks.close()  # client goes away

    stored = memory.get_full_history(conversation_id)
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[1].content == "The answer"


def test_stream_answer_writes_nothing_when_the_stream_fails_before_any_token(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    service = ChatService(
        llm_service=FakeStreamingLLM(["never reached"], fail_after=0),
        memory=memory,
        history_limit=10,
    )

    conversation_id, chunks = service.stream_answer("hello")
    try:
        list(chunks)
    except RuntimeError:
        pass

    # Only the user's turn — no empty assistant row.
    assert [m.role for m in memory.get_full_history(conversation_id)] == ["user"]


def test_stream_answer_sends_prior_turns_but_not_the_current_message(tmp_path):
    """
    History is loaded before the user's message is stored, so the message
    being answered must not also appear in the history sent to the model.
    """
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    memory.add_message("conv-1", role="user", content="earlier question")
    memory.add_message("conv-1", role="assistant", content="earlier answer")

    llm = FakeStreamingLLM(["ok"])
    service = ChatService(llm_service=llm, memory=memory, history_limit=10)

    _, chunks = service.stream_answer("new question", conversation_id="conv-1")
    list(chunks)

    assert llm.received_history == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
