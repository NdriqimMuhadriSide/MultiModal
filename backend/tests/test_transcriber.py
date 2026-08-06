import pytest

from ai.transcription_service import TranscriptionResult, TranscriptSegment as AITranscriptSegment
from processors.audio.transcriber import Transcriber


class FakeTranscriptionService:
    def __init__(self, result: TranscriptionResult) -> None:
        self._result = result
        self.last_call: tuple[bytes, str] | None = None

    def transcribe(self, audio_bytes: bytes, filename: str) -> TranscriptionResult:
        self.last_call = (audio_bytes, filename)
        return self._result


def test_transcribe_maps_ai_result_to_transcript():
    fake_result = TranscriptionResult(
        text="Let's ship the new feature by Friday.",
        language="en",
        segments=[AITranscriptSegment(start=0.0, end=2.5, text="Let's ship the new feature by Friday.")],
    )
    transcriber = Transcriber(transcription_service=FakeTranscriptionService(fake_result))

    transcript = transcriber.transcribe(audio_bytes=b"fake audio", filename="meeting.mp3")

    assert transcript.text == "Let's ship the new feature by Friday."
    assert transcript.language == "en"
    assert len(transcript.segments) == 1
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[0].end == 2.5
    assert transcript.segments[0].text == "Let's ship the new feature by Friday."


def test_transcribe_passes_audio_bytes_and_filename_through():
    fake_service = FakeTranscriptionService(TranscriptionResult(text="", language=None, segments=[]))
    transcriber = Transcriber(transcription_service=fake_service)

    transcriber.transcribe(audio_bytes=b"raw bytes here", filename="voice.wav")

    assert fake_service.last_call == (b"raw bytes here", "voice.wav")


def test_transcribe_handles_no_segments():
    fake_result = TranscriptionResult(text="hello", language="en", segments=[])
    transcriber = Transcriber(transcription_service=FakeTranscriptionService(fake_result))

    transcript = transcriber.transcribe(audio_bytes=b"fake", filename="a.mp3")

    assert transcript.segments == []
