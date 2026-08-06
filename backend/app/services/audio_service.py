"""
Audio business logic (FastAPI-facing wrapper).

Thin adapter around processors/audio/audio_processor.py's AudioProcessor
plus the LLM analysis step, mirroring the shape of
app/services/vision_service.py and app/services/rag_service.py: this is
the dependency-injection seam where configuration errors (missing
GROQ_API_KEY, etc.) get turned into clean HTTPExceptions instead of raw
500s. No audio processing or prompt-building logic lives here - that's in
processors/audio/ and prompts/audio_prompts.py respectively; this class
only sequences "process the audio, then analyze the transcript."

Layering: this is the Service layer between the API layer
(app/api/v1/endpoints/audio.py) and the Audio Processing layer
(processors/audio/audio_processor.py) + AI layer (ai/llm_service.py -
reused unchanged from chat, no new LLM client). It never talks to Whisper
directly - that's AudioProcessor's job via its Transcriber.
"""
from dataclasses import dataclass

from fastapi import HTTPException, status

from ai.llm_service import LLMService, get_llm_service
from ai.transcription_service import get_transcription_service
from app.core.config import settings
from prompts.audio_prompts import format_audio_analysis_prompt
from processors.audio.audio_metadata import AudioMetadata
from processors.audio.audio_processor import AudioProcessor
from processors.audio.audio_validator import AudioValidator
from processors.audio.transcriber import Transcriber, Transcript


@dataclass
class AudioAnalysisResult:
    transcript: Transcript
    analysis: str
    metadata: AudioMetadata


class AudioAnalysisService:
    """Runs the audio processing pipeline, then analyzes the resulting transcript with the LLM."""

    def __init__(self, audio_processor: AudioProcessor, llm_service: LLMService) -> None:
        self._audio_processor = audio_processor
        self._llm_service = llm_service

    def analyze(
        self, filename: str, mime_type: str, audio_bytes: bytes, question: str | None
    ) -> AudioAnalysisResult:
        """
        Process an uploaded audio file end to end: validate -> extract
        metadata -> transcribe -> analyze the transcript with the LLM.

        Raises:
            processors.audio.audio_validator.AudioValidationError: if the
                file fails validation.
            RuntimeError: if transcription or the LLM analysis call fails.
        """
        result = self._audio_processor.process(
            filename=filename, mime_type=mime_type, audio_bytes=audio_bytes
        )

        if not result.transcript.text:
            # A structurally valid audio file with nothing intelligible in
            # it (silence, pure noise) - don't send an empty transcript to
            # the LLM and risk it inventing content; return an honest,
            # deterministic result instead.
            analysis = (
                "No speech was detected in this audio file, so no analysis "
                "could be generated."
            )
        else:
            prompt = format_audio_analysis_prompt(
                transcript=result.transcript.text, question=question
            )
            analysis = self._llm_service.generate_response(prompt)

        return AudioAnalysisResult(
            transcript=result.transcript,
            analysis=analysis,
            metadata=result.metadata,
        )


def get_audio_analysis_service() -> AudioAnalysisService:
    """
    FastAPI dependency that builds an AudioAnalysisService.

    Configuration errors (e.g. missing GROQ_API_KEY) happen here, during
    dependency resolution, so they're translated into a clean
    HTTPException instead of leaking a raw stack trace to the client.
    """
    try:
        validator = AudioValidator(max_size_mb=settings.max_audio_size_mb)
        transcriber = Transcriber(transcription_service=get_transcription_service())
        audio_processor = AudioProcessor(validator=validator, transcriber=transcriber)
        return AudioAnalysisService(
            audio_processor=audio_processor,
            llm_service=get_llm_service(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Audio analysis service is not configured: {exc}",
        ) from exc
