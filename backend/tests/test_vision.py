import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.vision_service import (
    VisionAnalysisResult,
    VisionAnalysisService,
    get_vision_analysis_service,
)

client = TestClient(app)


class StubVisionAnalysisService(VisionAnalysisService):
    """Bypasses the real vision service so tests don't call Groq."""

    def __init__(self) -> None:  # no super().__init__ - no real vision client needed
        pass

    def analyze(
        self,
        image_bytes: bytes,
        mime_type: str,
        question: str,
        conversation_id: str | None = None,
    ) -> VisionAnalysisResult:
        return VisionAnalysisResult(
            answer=f"stub analysis ({mime_type}, {len(image_bytes)} bytes) for: {question}",
            conversation_id=conversation_id or "generated-id",
        )


def _tiny_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_vision_analyze_returns_answer():
    app.dependency_overrides[get_vision_analysis_service] = lambda: StubVisionAnalysisService()
    try:
        response = client.post(
            "/api/v1/vision/analyze",
            files={"image": ("test.png", _tiny_png_bytes(), "image/png")},
            data={"question": "Analyze this error."},
        )
    finally:
        app.dependency_overrides.pop(get_vision_analysis_service, None)

    assert response.status_code == 200
    body = response.json()
    assert "Analyze this error." in body["answer"]
    assert "image/png" in body["answer"]


def test_vision_analyze_requires_question():
    app.dependency_overrides[get_vision_analysis_service] = lambda: StubVisionAnalysisService()
    try:
        response = client.post(
            "/api/v1/vision/analyze",
            files={"image": ("test.png", _tiny_png_bytes(), "image/png")},
        )
    finally:
        app.dependency_overrides.pop(get_vision_analysis_service, None)

    assert response.status_code == 422


def test_vision_analyze_requires_image():
    app.dependency_overrides[get_vision_analysis_service] = lambda: StubVisionAnalysisService()
    try:
        response = client.post(
            "/api/v1/vision/analyze",
            data={"question": "Analyze this error."},
        )
    finally:
        app.dependency_overrides.pop(get_vision_analysis_service, None)

    assert response.status_code == 422
