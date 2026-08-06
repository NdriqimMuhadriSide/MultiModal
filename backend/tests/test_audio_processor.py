import pytest

from processors.audio.audio_processor import AudioProcessor
from processors.audio.audio_validator import AudioValidationError, AudioValidator
from processors.audio.transcriber import Transcript, TranscriptSegment


class FakeTranscriber:
    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript
        self.last_call: tuple[bytes, str] | None = None

    def transcribe(self, audio_bytes: bytes, filename: str) -> Transcript:
        self.last_call = (audio_bytes, filename)
        return self._transcript


def _wav_bytes() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 8000)
    return buffer.getvalue()


def test_process_runs_validate_then_metadata_then_transcribe():
    fake_transcript = Transcript(
        text="We decided to ship on Friday.",
        language="en",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="We decided to ship on Friday.")],
    )
    transcriber = FakeTranscriber(fake_transcript)
    processor = AudioProcessor(
        validator=AudioValidator(max_size_mb=25),
        transcriber=transcriber,
    )
    audio_bytes = _wav_bytes()

    result = processor.process(filename="meeting.wav", mime_type="audio/wav", audio_bytes=audio_bytes)

    assert result.transcript.text == "We decided to ship on Friday."
    assert result.metadata.filename == "meeting.wav"
    assert result.metadata.size_bytes == len(audio_bytes)
    assert result.metadata.sample_rate == 16000
    assert transcriber.last_call == (audio_bytes, "meeting.wav")


def test_process_raises_before_transcribing_when_validation_fails():
    transcriber = FakeTranscriber(Transcript(text="", language=None, segments=[]))
    processor = AudioProcessor(
        validator=AudioValidator(max_size_mb=25),
        transcriber=transcriber,
    )

    with pytest.raises(AudioValidationError):
        processor.process(filename="notes.txt", mime_type="text/plain", audio_bytes=b"fake")

    # Transcriber should never have been called for a file that failed validation.
    assert transcriber.last_call is None


def test_process_rejects_oversized_file_before_transcribing():
    transcriber = FakeTranscriber(Transcript(text="", language=None, segments=[]))
    processor = AudioProcessor(
        validator=AudioValidator(max_size_mb=1),
        transcriber=transcriber,
    )
    oversized = b"a" * (2 * 1024 * 1024)

    with pytest.raises(AudioValidationError):
        processor.process(filename="big.mp3", mime_type="audio/mpeg", audio_bytes=oversized)

    assert transcriber.last_call is None
