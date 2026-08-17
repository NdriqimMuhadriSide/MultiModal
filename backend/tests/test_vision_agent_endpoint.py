"""
Tests for POST /vision/ask - the HTTP layer only.

The agent is covered in test_vision_agent.py; these pin the multipart
contract and the response shape the frontend reads.
"""
from fastapi.testclient import TestClient

from agents.agent_loop import AgentStep
from app.main import app
from app.schemas.rag import RAGChatSource
from app.services.vision_agent_service import (
    VisionAgentChatResult,
    get_vision_agent_chat_service,
)

client = TestClient(app)


class StubVisionAgentChatService:
    def __init__(self, result: VisionAgentChatResult | None = None) -> None:
        self.calls: list[tuple[str, bytes, str, str | None]] = []
        self._result = result

    def analyze(
        self, question: str, image_bytes: bytes, mime_type: str, conversation_id=None
    ) -> VisionAgentChatResult:
        self.calls.append((question, image_bytes, mime_type, conversation_id))
        if self._result is not None:
            return self._result
        return VisionAgentChatResult(
            conversation_id=conversation_id or "generated-id",
            answer=f"stub answer for: {question}",
        )


def _post(stub, data: dict, files: dict | None = None):
    app.dependency_overrides[get_vision_agent_chat_service] = lambda: stub
    try:
        return client.post(
            "/api/v1/vision/ask",
            data=data,
            files=files or {"image": ("receipt.png", b"PNGBYTES", "image/png")},
        )
    finally:
        app.dependency_overrides.pop(get_vision_agent_chat_service, None)


def test_ask_returns_the_answer_and_passes_the_image_through():
    stub = StubVisionAgentChatService()
    response = _post(stub, {"question": "what is the total?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub answer for: what is the total?"
    assert body["conversationId"] == "generated-id"
    assert stub.calls == [("what is the total?", b"PNGBYTES", "image/png", None)]


def test_ask_serialises_the_trace_and_citations():
    stub = StubVisionAgentChatService(
        VisionAgentChatResult(
            conversation_id="conv-9",
            answer="84.50 exceeds the cap [E1].",
            steps=[
                AgentStep(
                    thought="I need the exact digits.",
                    action_json='{"tool": "read_text", "input": {}}',
                    tool="read_text",
                    tool_input={},
                    observation="Recognised text...",
                )
            ],
            sources=[
                RAGChatSource(
                    filename="expenses.pdf", page=4, chunk_id="c1", section="Limits"
                )
            ],
            stopped_because="step_limit",
        )
    )
    body = _post(stub, {"question": "within policy?", "conversation_id": "conv-9"}).json()

    assert body["stoppedBecause"] == "step_limit"
    assert body["steps"] == [
        {
            "thought": "I need the exact digits.",
            "tool": "read_text",
            "toolInput": "{}",
            "observation": "Recognised text...",
            # Empty because this agent calls tools, not other agents - but
            # present on every step, so one renderer handles this trace and a
            # delegating supervisor's alike.
            "children": [],
        }
    ]
    # Same citation shape as /rag/chat, /agent/ask and /research/ask.
    assert body["sources"] == [
        {"filename": "expenses.pdf", "page": 4, "chunkId": "c1", "section": "Limits"}
    ]


def test_ask_rejects_an_oversized_image():
    from app.core.config import settings

    oversized = b"x" * (settings.max_upload_size_mb * 1024 * 1024 + 1)
    response = _post(
        StubVisionAgentChatService(),
        {"question": "q"},
        files={"image": ("big.png", oversized, "image/png")},
    )
    assert response.status_code == 413


def test_ask_maps_an_unsupported_image_type_to_400():
    class Rejecting(StubVisionAgentChatService):
        def analyze(self, question, image_bytes, mime_type, conversation_id=None):
            raise ValueError(f"Unsupported image type '{mime_type}'.")

    response = _post(
        Rejecting(),
        {"question": "q"},
        files={"image": ("scan.tiff", b"bytes", "image/tiff")},
    )
    assert response.status_code == 400
    assert "Unsupported image type" in response.json()["detail"]


def test_ask_maps_a_provider_failure_to_502():
    class Failing(StubVisionAgentChatService):
        def analyze(self, question, image_bytes, mime_type, conversation_id=None):
            raise RuntimeError("Vision request failed: upstream timeout")

    response = _post(Failing(), {"question": "q"})
    assert response.status_code == 502
    assert "upstream timeout" in response.json()["detail"]


def test_unverified_values_reach_the_response():
    stub = StubVisionAgentChatService(
        VisionAgentChatResult(
            conversation_id="c",
            answer="The total is 92.00.",
            unverified_values=["92.00"],
        )
    )
    body = _post(stub, {"question": "what is the total?"}).json()

    assert body["unverifiedValues"] == ["92.00"]


def test_unverified_values_defaults_to_empty():
    body = _post(StubVisionAgentChatService(), {"question": "q"}).json()
    assert body["unverifiedValues"] == []


# ---- The image is optional on a follow-up ----------------------------------


def _post_without_a_file(stub, data: dict):
    app.dependency_overrides[get_vision_agent_chat_service] = lambda: stub
    try:
        return client.post("/api/v1/vision/ask", data=data)
    finally:
        app.dependency_overrides.pop(get_vision_agent_chat_service, None)


def test_ask_accepts_a_request_with_no_image():
    stub = StubVisionAgentChatService()

    response = _post_without_a_file(
        stub, {"question": "and the date?", "conversation_id": "conv-1"}
    )

    assert response.status_code == 200
    # None, not b"" - the service reads it as "use the image this
    # conversation already has", and an empty bytestring would look like a
    # zero-byte upload instead.
    assert stub.calls == [("and the date?", None, "", "conv-1")]


def test_ask_treats_an_empty_upload_as_no_image():
    stub = StubVisionAgentChatService()

    response = _post(
        stub,
        {"question": "and the date?", "conversation_id": "conv-1"},
        files={"image": ("", b"", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert stub.calls[0][1] is None


def test_ask_without_an_image_or_a_conversation_is_a_400():
    class Rejecting(StubVisionAgentChatService):
        def analyze(self, question, image_bytes, mime_type, conversation_id=None):
            raise ValueError("No image was provided and this conversation has none.")

    response = _post_without_a_file(Rejecting(), {"question": "what is this?"})

    assert response.status_code == 400
    assert "No image was provided" in response.json()["detail"]
