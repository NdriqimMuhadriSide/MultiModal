"""
Real-time streaming endpoint.

Pure HTTP layer: parses the multipart/form-data request (one sampled
frame + sessionId + optional question + timestamp), maps domain errors to
HTTP status codes, and returns the response model. No Whisper/vision/LLM-
specific code lives here - that lives in processors/streaming/ (validation,
sampling, vision analysis orchestration) and app/services/stream_service.py
(builds the process-wide StreamProcessor). This mirrors
app/api/v1/endpoints/audio.py and app/api/v1/endpoints/vision.py exactly.

Each call to this endpoint represents ONE frame the client has already
captured and decided to send - the client (frontend hooks/useStreaming.ts)
is responsible for capturing frames off a MediaStream and posting them
here one at a time; this endpoint has no concept of "a stream" beyond the
sessionId tying a sequence of individual frame requests together.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.schemas.stream import StreamFrameResponse
from processors.streaming.stream_processor import StreamProcessor
from app.services.stream_service import get_stream_processor
from processors.streaming.stream_validator import StreamValidationError

router = APIRouter(tags=["streaming"])


@router.post("/stream/frame", response_model=StreamFrameResponse)
async def analyze_frame(
    frame: UploadFile = File(..., description="One sampled video frame (image)."),
    sessionId: str = Form(..., description="Identifies which streaming session this frame belongs to."),
    question: str | None = Form(
        default=None,
        description="Optional question about the current frame (e.g. 'Explain this error message.').",
    ),
    timestamp: str | None = Form(
        default=None,
        description="Client-side capture timestamp (ISO 8601 or epoch ms), for logging/future use.",
    ),
    stream_processor: StreamProcessor = Depends(get_stream_processor),
) -> StreamFrameResponse:
    """
    Frame -> validate -> sampling decision -> (if due) vision analysis ->
    update session context -> {observations, analysis, sampled}.

    Most calls to this endpoint return `sampled: false` with empty
    observations/analysis - that's the frame sampler intentionally
    dropping a frame that arrived between sampling points, not an error.
    """
    frame_bytes = await frame.read()

    try:
        result = stream_processor.process_frame(
            session_id=sessionId,
            mime_type=frame.content_type or "",
            frame_bytes=frame_bytes,
            question=question,
        )
    except StreamValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    observations = result.observation.as_flat_list() if result.observation else []

    return StreamFrameResponse(
        observations=observations,
        analysis=result.analysis,
        sampled=result.sampled,
    )


@router.post("/stream/{session_id}/end", status_code=status.HTTP_204_NO_CONTENT)
async def end_stream_session(
    session_id: str,
    stream_processor: StreamProcessor = Depends(get_stream_processor),
) -> None:
    """
    Explicitly end a streaming session, freeing its context buffer and
    sampling state immediately rather than waiting for TTL-based cleanup.
    Called by the frontend when the user clicks "Stop" (see useStreaming.ts).
    """
    stream_processor.end_session(session_id)
