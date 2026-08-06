"""
Transcriber - the audio processing layer's view of "turn audio into text."

Responsibility: receive uploaded audio and return a transcript, delegating
the actual Whisper call to ai/transcription_service.py (the AI layer).

This is a deliberately thin wrapper, not a reimplementation - its purpose
is to keep `processors/audio/audio_processor.py` depending on a narrow,
audio-domain-shaped interface (`Transcriber.transcribe(audio_bytes,
filename) -> Transcript`) instead of importing `ai/transcription_service.py`
directly. That indirection is what makes it possible to:
    - swap the underlying speech-to-text provider (e.g. move off Groq to
      OpenAI's own Whisper endpoint, or a self-hosted faster-whisper model)
      by changing only this file's constructor, with zero changes to
      audio_processor.py or anything above it.
    - unit test audio_processor.py against a fake Transcriber without
      touching the real AI client.

Per the architecture rules: this module is the boundary between the Audio
Processing layer and the Transcription/AI layer. Nothing above this file
(audio_processor.py, the service layer, the API layer) is allowed to import
ai/transcription_service.py directly.
"""
from dataclasses import dataclass

from ai.transcription_service import TranscriptionService


@dataclass
class TranscriptSegment:
    """A single timestamped segment of the transcript, if available."""

    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """The full transcription result for one audio file."""

    text: str
    language: str | None
    segments: list[TranscriptSegment]


class Transcriber:
    """Turns audio bytes into a Transcript by delegating to a TranscriptionService."""

    def __init__(self, transcription_service: TranscriptionService) -> None:
        self._transcription_service = transcription_service

    def transcribe(self, audio_bytes: bytes, filename: str) -> Transcript:
        """
        Receive uploaded audio and return its transcript.

        Raises:
            ValueError: if `audio_bytes` is empty (propagated from
                ai.transcription_service).
            RuntimeError: if the transcription call fails (propagated from
                ai.transcription_service).
        """
        result = self._transcription_service.transcribe(audio_bytes, filename)

        return Transcript(
            text=result.text,
            language=result.language,
            segments=[
                TranscriptSegment(start=s.start, end=s.end, text=s.text)
                for s in result.segments
            ],
        )
