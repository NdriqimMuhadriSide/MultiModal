"""
Tests for the step-streaming path.

Two things need pinning. First, that `run` and `stream` cannot disagree -
`run` is implemented as a drain of `stream`, and these assert the two
really do produce the same trace and the same answer, because that
equivalence is the whole reason for the refactor.

Second, the SSE frame contract, which the frontend reads: order, shape,
and the rule that a streamed step is byte-identical to a step from the
non-streaming endpoint.
"""
import json

import pytest
from fastapi.testclient import TestClient

from agents.agent_loop import StepEvent
from agents.vision_agent import VisionResultEvent
from app.main import app
from app.services.vision_agent_service import get_vision_agent_chat_service
from rag import ocr
from tests.test_vision_agent import (  # reuse the fakes
    PNG,
    FakeLLMService,
    FakeRetriever,
    FakeVisionService,
    _agent,
    _chunk,
    _finish,
    _inspect,
    _read,
    _search,
)

client = TestClient(app)


@pytest.fixture
def ocr_returns(monkeypatch):
    def _set(text: str):
        monkeypatch.setattr(ocr, "is_available", lambda: True)
        monkeypatch.setattr(ocr, "ocr_image", lambda image_bytes: text)

    return _set


# ---- run and stream must agree ---------------------------------------------


def test_stream_emits_one_step_event_per_turn_then_one_result(ocr_returns):
    ocr_returns("TOTAL 84.50")
    llm = FakeLLMService([_read(), _search("meal limit"), _finish("84.50 [E1].")])
    retriever = FakeRetriever([[_chunk("c1")]])

    events = list(_agent(llm, retriever=retriever).stream("q", b"img", PNG))

    assert [type(event) for event in events] == [
        StepEvent,
        StepEvent,
        StepEvent,
        VisionResultEvent,
    ]
    assert [event.step.tool for event in events[:3]] == [
        "read_text",
        "search_knowledge_base",
        "finish",
    ]


def test_run_and_stream_produce_the_same_result(ocr_returns):
    """`run` is a drain of `stream`, so this is the invariant behind that."""
    ocr_returns("TOTAL 84.50")

    def build():
        return _agent(
            FakeLLMService([_read(), _finish("The total is 84.50.")]),
            retriever=FakeRetriever([]),
        )

    from_run = build().run("q", b"img", PNG)

    streamed = [
        event
        for event in build().stream("q", b"img", PNG)
        if isinstance(event, VisionResultEvent)
    ]
    from_stream = streamed[0].result

    assert from_run.answer == from_stream.answer
    assert from_run.stopped_because == from_stream.stopped_because
    assert [s.tool for s in from_run.steps] == [s.tool for s in from_stream.steps]
    assert from_run.unverified_values == from_stream.unverified_values


def test_the_result_event_carries_citations_and_the_numeric_check(ocr_returns):
    """Both are properties of the whole run, so they arrive with the result."""
    ocr_returns("TOTAL 84.50")
    llm = FakeLLMService([_read(), _search("cap"), _finish("84.50 vs the 50 cap [E1].")])

    events = list(
        _agent(llm, retriever=FakeRetriever([[_chunk("c1")]])).stream("q", b"img", PNG)
    )
    result = events[-1].result

    assert [source.chunk_id for source in result.sources] == ["c1"]
    assert result.unverified_values == ["50"]


def test_stream_validates_eagerly_rather_than_on_first_iteration():
    """A bad mime type must still be answerable as a 400."""
    agent = _agent(FakeLLMService([]))
    with pytest.raises(ValueError, match="Unsupported image type"):
        agent.stream("q", b"img", "image/tiff")


# ---- The SSE frame contract ------------------------------------------------


class StubStreamService:
    def __init__(self, events: list[dict]) -> None:
        self._events = events
        self.calls: list[tuple] = []

    def stream_analyze(
        self, question, image_bytes, mime_type, conversation_id=None
    ):
        self.calls.append((question, image_bytes, mime_type, conversation_id))
        return conversation_id or "generated-id", iter(self._events)


def _frames(stub, data=None):
    app.dependency_overrides[get_vision_agent_chat_service] = lambda: stub
    try:
        response = client.post(
            "/api/v1/vision/ask/stream",
            data=data or {"question": "what is the total?"},
            files={"image": ("receipt.png", b"PNGBYTES", "image/png")},
        )
    finally:
        app.dependency_overrides.pop(get_vision_agent_chat_service, None)

    assert response.status_code == 200
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_the_stream_opens_with_start_and_closes_with_done():
    stub = StubStreamService(
        [
            {"type": "step", "index": 1, "step": {"tool": "read_text"}},
            {"type": "answer", "content": "The total is 84.50."},
            {"type": "done", "stopped_because": "finished"},
        ]
    )
    frames = _frames(stub)

    assert frames[0] == {"type": "start", "conversation_id": "generated-id"}
    assert [frame["type"] for frame in frames] == [
        "start",
        "step",
        "answer",
        "done",
    ]


def test_a_provider_failure_mid_stream_becomes_an_error_frame():
    """By the time the model can fail, the response is already a 200."""

    class Failing(StubStreamService):
        def stream_analyze(self, question, image_bytes, mime_type, conversation_id=None):
            def events():
                yield {"type": "step", "index": 1, "step": {"tool": "read_text"}}
                raise RuntimeError("Vision request failed: upstream 500")

            return "c", events()

    frames = _frames(Failing([]))

    assert [frame["type"] for frame in frames] == ["start", "step", "error"]
    assert "upstream 500" in frames[-1]["detail"]


def test_an_oversized_image_is_still_a_real_http_error():
    from app.core.config import settings

    app.dependency_overrides[get_vision_agent_chat_service] = lambda: StubStreamService([])
    try:
        response = client.post(
            "/api/v1/vision/ask/stream",
            data={"question": "q"},
            files={
                "image": (
                    "big.png",
                    b"x" * (settings.max_upload_size_mb * 1024 * 1024 + 1),
                    "image/png",
                )
            },
        )
    finally:
        app.dependency_overrides.pop(get_vision_agent_chat_service, None)

    assert response.status_code == 413


# ---- The service layer's frame shapes --------------------------------------


def test_streamed_steps_match_the_non_streaming_step_shape(ocr_returns, tmp_path):
    """
    One step type for the frontend whether it streamed the run or not.
    """
    from memory.attachment_store import AttachmentStore
    from app.services.vision_agent_service import VisionAgentChatService

    ocr_returns("TOTAL 84.50")

    class FakeMemory:
        def new_conversation_id(self):
            return "conv-1"

        def get_history(self, conversation_id, limit=10):
            return []

        def get_last_attachment(self, conversation_id, modality):
            return None

        def add_message(self, conversation_id, role, content, **kwargs):
            pass

    service = VisionAgentChatService(
        agent=_agent(FakeLLMService([_read(), _finish("done")])),
        memory=FakeMemory(),
        attachments=AttachmentStore(root_dir=str(tmp_path / "attachments")),
    )
    _, events = service.stream_analyze("q", b"img", PNG)
    frames = list(events)

    step_frames = [frame for frame in frames if frame["type"] == "step"]
    assert set(step_frames[0]["step"]) == {
        "thought",
        "tool",
        "toolInput",
        "observation",
        "children",
    }
    # Arguments are JSON text, exactly as POST /vision/ask returns them.
    assert json.loads(step_frames[0]["step"]["toolInput"]) == {}
    assert [frame["type"] for frame in frames] == ["step", "step", "answer", "done"]


def test_the_assistant_turn_is_persisted_when_the_stream_ends(ocr_returns, tmp_path):
    from memory.attachment_store import AttachmentStore
    from app.services.vision_agent_service import VisionAgentChatService

    ocr_returns("TOTAL 84.50")
    stored: list[tuple[str, str, str, str]] = []

    class RecordingMemory:
        def new_conversation_id(self):
            return "conv-1"

        def get_history(self, conversation_id, limit=10):
            return []

        def get_last_attachment(self, conversation_id, modality):
            return None

        def add_message(
            self, conversation_id, role, content, modality="text", attachment_ref=None
        ):
            stored.append((role, content, modality, attachment_ref))

    service = VisionAgentChatService(
        agent=_agent(FakeLLMService([_read(), _finish("The total is 84.50.")])),
        memory=RecordingMemory(),
        attachments=AttachmentStore(root_dir=str(tmp_path / "attachments")),
    )
    _, events = service.stream_analyze("what is the total?", b"img", PNG)
    list(events)

    assert [(role, content) for role, content, _, _ in stored] == [
        ("user", "what is the total?"),
        ("assistant", "The total is 84.50."),
    ]
    # Both halves of the turn carry the same image, so a later turn can tell
    # what the answer was about.
    assert {modality for _, _, modality, _ in stored} == {"image"}
    refs = {ref for _, _, _, ref in stored}
    assert len(refs) == 1 and refs != {None}
