"""
Transcription service - the only file in the codebase that sends audio to
the speech-to-text (Whisper) model.

Responsibilities:
1. Load the API key / Whisper model name / base URL from environment
   variables (via app settings) - the exact same GROQ_API_KEY and base URL
   ai/llm_service.py and ai/vision_service.py already use. No new client,
   no new credentials: Groq's OpenAI-compatible API exposes Whisper
   through the same `openai` SDK this project already depends on, just via
   `client.audio.transcriptions.create(...)` instead of
   `chat.completions.create(...)`.
2. Accept raw audio bytes + filename.
3. Send them to the Whisper model.
4. Return the transcript text, detected language, and (if the model
   provides them) per-segment timestamps.

This module has no knowledge of FastAPI, HTTP, or file upload handling -
just bytes in, transcript out - mirroring ai/vision_service.py exactly.
Business rules (validation, metadata, orchestration) live in
processors/audio/, one layer up.

Whisper model notes (see prompts you asked to be explained, echoed here
since this is where the model is actually invoked):
    - Whisper is trained on large-scale multilingual audio+transcript pairs
      and performs speech-to-text only - it has no reasoning ability and
      does not "understand" what it transcribes.
    - Language detection is automatic unless a language hint is passed;
      Groq's Whisper endpoint returns the detected language in its verbose
      response format.
    - Timestamp support: requesting the "verbose_json" response format
      returns per-segment start/end timestamps alongside the text, which
      is what this service requests by default - useful today for display,
      and exactly the granularity a future audio-RAG chunking step would
      key off of instead of re-deriving offsets from scratch.
    - Accuracy considerations: transcription quality depends heavily on
      audio quality (sample rate, background noise, overlapping speakers,
      accents) - see the "common challenges with noisy audio" explanation
      after implementation.
"""
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

from openai import OpenAI

from app.core.config import settings


@dataclass
class TranscriptSegment:
    """A single timestamped segment of the transcript, if the model provides one."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[TranscriptSegment]


def _segment_field(segment: object, field: str, default: object) -> object:
    """
    Read a field off a Whisper response segment, regardless of whether the
    SDK represents it as a dict (e.g. `{"start": 0.0, ...}`) or a Pydantic
    model/object with attributes (e.g. Groq's `TranscriptionSegment`).
    Different OpenAI-compatible providers have been observed returning
    either shape for verbose_json segments, so this keeps the parsing
    logic here - one place - instead of assuming one shape and breaking
    when a provider returns the other.
    """
    if isinstance(segment, dict):
        return segment.get(field, default)
    return getattr(segment, field, default)


class TranscriptionService:
    """Wraps an OpenAI-compatible client for audio (speech-to-text) transcription."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file before "
                "calling the transcription service."
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def transcribe(self, audio_bytes: bytes, filename: str) -> TranscriptionResult:
        """
        Send audio to the Whisper model and return its transcript.

        Raises:
            ValueError: if `audio_bytes` is empty.
            RuntimeError: if the API call fails.
        """
        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty.")

        audio_file = BytesIO(audio_bytes)
        audio_file.name = filename  # the SDK reads .name to set the multipart filename

        try:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=audio_file,
                response_format="verbose_json",
            )
        except Exception as exc:  # noqa: BLE001 - surface as a domain error
            raise RuntimeError(f"Transcription request failed: {exc}") from exc

        segments = [
            TranscriptSegment(
                start=float(_segment_field(segment, "start", 0.0)),
                end=float(_segment_field(segment, "end", 0.0)),
                text=str(_segment_field(segment, "text", "")).strip(),
            )
            for segment in (getattr(response, "segments", None) or [])
        ]

        return TranscriptionResult(
            text=(response.text or "").strip(),
            language=getattr(response, "language", None),
            segments=segments,
        )


@lru_cache
def get_transcription_service() -> TranscriptionService:
    """Return a cached TranscriptionService instance built from app settings (Groq Whisper)."""
    return TranscriptionService(
        api_key=settings.groq_api_key,
        model=settings.groq_whisper_model,
        base_url=settings.groq_base_url,
    )
