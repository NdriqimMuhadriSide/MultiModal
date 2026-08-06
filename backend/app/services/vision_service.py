"""
Vision business logic.

Sits between the API route and the AI vision service, mirroring
app/services/chat_service.py. Currently just delegates to the vision
service, but this is the place to add things like image resizing/validation
rules, logging of analysis requests, or combining vision output with RAG
context in later tasks - without changing the route or the vision client code.
"""
from fastapi import HTTPException, status

from ai.vision_service import VisionService, get_vision_service


class VisionAnalysisService:
    def __init__(self, vision_service: VisionService) -> None:
        self._vision_service = vision_service

    def analyze(self, image_bytes: bytes, mime_type: str, question: str) -> str:
        return self._vision_service.analyze_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            question=question,
        )


def get_vision_analysis_service() -> VisionAnalysisService:
    """
    FastAPI dependency that builds a VisionAnalysisService.

    Configuration errors (e.g. missing GROQ_API_KEY) happen here, during
    dependency resolution - outside the route handler's try/except - so
    they're translated into a clean HTTPException instead of leaking a raw
    stack trace to the client as an unhandled 500.
    """
    try:
        return VisionAnalysisService(vision_service=get_vision_service())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vision service is not configured: {exc}",
        ) from exc
