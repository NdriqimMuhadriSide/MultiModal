"""
Vision endpoint.

Pure HTTP layer: parses the multipart/form-data request (image file +
question field), enforces the upload size limit, calls the vision service,
maps domain errors to HTTP status codes, and returns the response model.
No OpenAI/Groq-specific code lives here.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.vision import VisionAnalyzeResponse
from app.services.vision_service import VisionAnalysisService, get_vision_analysis_service

router = APIRouter(tags=["vision"])


@router.post("/vision/analyze", response_model=VisionAnalyzeResponse)
async def analyze_image(
    image: UploadFile = File(..., description="Image file to analyze."),
    question: str = Form(..., description="Question about the image."),
    conversation_id: str | None = Form(
        default=None,
        description=(
            "Conversation to file this analysis under. Omit to start a new "
            "one - the backend generates an id and returns it. It does not "
            "change the answer; this endpoint asks one question about one "
            "image."
        ),
    ),
    vision_service: VisionAnalysisService = Depends(get_vision_analysis_service),
) -> VisionAnalyzeResponse:
    image_bytes = await image.read()

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {settings.max_upload_size_mb}MB upload limit.",
        )

    try:
        result = vision_service.analyze(
            image_bytes=image_bytes,
            mime_type=image.content_type or "",
            question=question,
            conversation_id=conversation_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return VisionAnalyzeResponse(
        answer=result.answer, conversation_id=result.conversation_id
    )
