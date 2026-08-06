"""
Audio processor - orchestrates the Audio Processing layer's pipeline:

    Receive Audio
        -> Validate                 (audio_validator.AudioValidator)
        -> Extract Metadata          (audio_metadata.extract_metadata)
        -> Generate Transcript       (transcriber.Transcriber - delegates
                                       to the Transcription/AI layer)
        -> Return Transcript + Metadata

This is the single entry point the service layer (app/services/audio_service.py)
calls - it never talks to AudioValidator, extract_metadata, or Transcriber
directly. That keeps the orchestration order (validate before spending time
on metadata/transcription; metadata extraction before the expensive
network call to Whisper) defined in exactly one place.

Dependency injection: AudioProcessor receives its Transcriber (and,
implicitly, the validator config) through its constructor rather than
constructing them itself - this is what let audio_processor be unit tested
against a fake Transcriber (see tests/test_audio_processor.py) without a
real Whisper call, and follows the Single Responsibility Principle: this
class's only job is sequencing, not building its own collaborators.
"""
from dataclasses import dataclass

from processors.audio.audio_metadata import AudioMetadata, extract_metadata
from processors.audio.audio_validator import AudioValidationError, AudioValidator
from processors.audio.transcriber import Transcriber, Transcript


@dataclass
class AudioProcessingResult:
    transcript: Transcript
    metadata: AudioMetadata


class AudioProcessor:
    """Runs the validate -> extract metadata -> transcribe pipeline for one audio file."""

    def __init__(self, validator: AudioValidator, transcriber: Transcriber) -> None:
        self._validator = validator
        self._transcriber = transcriber

    def process(self, filename: str, mime_type: str, audio_bytes: bytes) -> AudioProcessingResult:
        """
        Validate, extract metadata from, and transcribe an uploaded audio file.

        Raises:
            AudioValidationError: if the file fails MIME/extension/size
                validation (propagated from audio_validator).
            RuntimeError: if transcription fails (propagated from the
                Transcription/AI layer).
        """
        self._validator.validate(filename=filename, mime_type=mime_type, file_bytes=audio_bytes)

        metadata = extract_metadata(filename=filename, file_bytes=audio_bytes)

        transcript = self._transcriber.transcribe(audio_bytes=audio_bytes, filename=filename)

        return AudioProcessingResult(transcript=transcript, metadata=metadata)


__all__ = ["AudioProcessor", "AudioProcessingResult", "AudioValidationError"]
