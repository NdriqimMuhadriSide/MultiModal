from fastapi.testclient import TestClient

from app.main import app
from app.services.stream_service import get_stream_processor
from processors.streaming.stream_processor import FrameProcessingResult, StreamProcessor
from processors.streaming.stream_validator import StreamValidationError
from processors.streaming.vision_stream import StreamObservation

client = TestClient(app)


def _tiny_jpeg_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(buffer, format="JPEG")
    return buffer.getvalue()


class StubStreamProcessor(StreamProcessor):
    """Bypasses the real validator/sampler/vision analyzer so tests don't call Groq."""

    def __init__(self, result: FrameProcessingResult | None = None, raise_error: Exception | None = None) -> None:
        self._result = result
        self._raise_error = raise_error
        self.last_call: dict | None = None
        self.ended_sessions: list[str] = []

    def process_frame(self, session_id, mime_type, frame_bytes, question):
        self.last_call = {
            "session_id": session_id,
            "mime_type": mime_type,
            "question": question,
        }
        if self._raise_error:
            raise self._raise_error
        return self._result

    def end_session(self, session_id):
        self.ended_sessions.append(session_id)


def test_analyze_frame_returns_observations_when_sampled():
    result = FrameProcessingResult(
        sampled=True,
        observation=StreamObservation(objects=["laptop"], activities=["typing"]),
        analysis="A laptop is visible; someone appears to be typing.",
    )
    app.dependency_overrides[get_stream_processor] = lambda: StubStreamProcessor(result=result)
    try:
        response = client.post(
            "/api/v1/stream/frame",
            files={"frame": ("frame.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
            data={"sessionId": "session-a", "question": "What's on the desk?"},
        )
    finally:
        app.dependency_overrides.pop(get_stream_processor, None)

    assert response.status_code == 200
    body = response.json()
    assert body["sampled"] is True
    assert body["observations"] == ["laptop", "typing"]
    assert body["analysis"] == "A laptop is visible; someone appears to be typing."


def test_analyze_frame_returns_empty_result_when_not_sampled():
    result = FrameProcessingResult(sampled=False, observation=None, analysis="")
    app.dependency_overrides[get_stream_processor] = lambda: StubStreamProcessor(result=result)
    try:
        response = client.post(
            "/api/v1/stream/frame",
            files={"frame": ("frame.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
            data={"sessionId": "session-a"},
        )
    finally:
        app.dependency_overrides.pop(get_stream_processor, None)

    assert response.status_code == 200
    body = response.json()
    assert body["sampled"] is False
    assert body["observations"] == []
    assert body["analysis"] == ""


def test_analyze_frame_requires_session_id():
    app.dependency_overrides[get_stream_processor] = lambda: StubStreamProcessor()
    try:
        response = client.post(
            "/api/v1/stream/frame",
            files={"frame": ("frame.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
        )
    finally:
        app.dependency_overrides.pop(get_stream_processor, None)

    assert response.status_code == 422


def test_analyze_frame_requires_frame_file():
    app.dependency_overrides[get_stream_processor] = lambda: StubStreamProcessor()
    try:
        response = client.post(
            "/api/v1/stream/frame",
            data={"sessionId": "session-a"},
        )
    finally:
        app.dependency_overrides.pop(get_stream_processor, None)

    assert response.status_code == 422


def test_analyze_frame_maps_validation_error_to_bad_request():
    app.dependency_overrides[get_stream_processor] = lambda: StubStreamProcessor(
        raise_error=StreamValidationError("sessionId must not be empty.")
    )
    try:
        response = client.post(
            "/api/v1/stream/frame",
            files={"frame": ("frame.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
            data={"sessionId": "session-a"},
        )
    finally:
        app.dependency_overrides.pop(get_stream_processor, None)

    assert response.status_code == 400


def test_analyze_frame_maps_runtime_error_to_bad_gateway():
    app.dependency_overrides[get_stream_processor] = lambda: StubStreamProcessor(
        raise_error=RuntimeError("Vision request failed: connection refused")
    )
    try:
        response = client.post(
            "/api/v1/stream/frame",
            files={"frame": ("frame.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
            data={"sessionId": "session-a"},
        )
    finally:
        app.dependency_overrides.pop(get_stream_processor, None)

    assert response.status_code == 502


def test_end_stream_session_calls_processor_and_returns_no_content():
    stub = StubStreamProcessor()
    app.dependency_overrides[get_stream_processor] = lambda: stub
    try:
        response = client.post("/api/v1/stream/session-a/end")
    finally:
        app.dependency_overrides.pop(get_stream_processor, None)

    assert response.status_code == 204
    assert stub.ended_sessions == ["session-a"]
