"""
Audio understanding endpoint.

Pure HTTP layer: parses the multipart/form-data request (audio file +
optional question field), maps domain errors to HTTP status codes, and
returns the response model. No Whisper/LLM-specific code lives here - that
lives in processors/audio/ (validation, metadata, transcription
orchestration) and app/services/audio_service.py (ties processing +
analysis together). This mirrors app/api/v1/endpoints/vision.py exactly.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.schemas.audio import AudioAnalyzeResponse, AudioMetadataResponse
from app.services.audio_service import AudioAnalysisService, get_audio_analysis_service
from processors.audio.audio_validator import AudioValidationError

router = APIRouter(tags=["audio"])


@router.post("/audio/analyze", response_model=AudioAnalyzeResponse)
async def analyze_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe and analyze."),
    question: str | None = Form(
        default=None,
        description="Optional question about the audio (e.g. 'Summarize the key decisions.'). "
        "Defaults to a general summary if omitted.",
    ),
    audio_service: AudioAnalysisService = Depends(get_audio_analysis_service),
) -> AudioAnalyzeResponse:
    """
    Audio -> Speech-to-Text -> Transcript -> GPT Analysis -> Summary.

        audio bytes -> validate (type/size) -> extract metadata
        (duration/sample rate/channels) -> Whisper transcription ->
        LLM analysis of the transcript (default summary, or the user's
        custom question) -> {transcript, analysis, metadata}
    """
    audio_bytes = await audio.read()

    try:
        result = audio_service.analyze(
            filename=audio.filename or "unknown",
            mime_type=audio.content_type or "",
            audio_bytes=audio_bytes,
            question=question,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return AudioAnalyzeResponse(
        transcript=result.transcript.text,
        analysis=result.analysis,
        metadata=AudioMetadataResponse(
            filename=result.metadata.filename,
            duration=result.metadata.duration_seconds,
            size=result.metadata.size_bytes,
            sample_rate=result.metadata.sample_rate,
            channels=result.metadata.channels,
        ),
    )
