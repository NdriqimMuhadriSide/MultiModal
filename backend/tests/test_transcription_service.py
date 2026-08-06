"""
Tests for ai/transcription_service.py's response parsing.

Regression coverage for a real bug found via a live smoke test: Groq's SDK
returns verbose_json `segments` as Pydantic model objects (attribute
access via `.start`/`.end`/`.text`), not plain dicts - the original
implementation assumed dicts (`segment.get(...)`) and raised
AttributeError against a real Groq response. _segment_field() must handle
both a dict-shaped segment and an object-shaped segment.
"""
from dataclasses import dataclass

import pytest

from ai.transcription_service import TranscriptionService


@dataclass
class FakeSegmentObject:
    """Stands in for Groq SDK's TranscriptionSegment - attribute access, not dict.get()."""

    start: float
    end: float
    text: str


class FakeTranscriptionResponse:
    def __init__(self, text: str, language: str, segments: list) -> None:
        self.text = text
        self.language = language
        self.segments = segments


class FakeAudioTranscriptions:
    def __init__(self, response) -> None:
        self._response = response

    def create(self, model, file, response_format):
        return self._response


class FakeAudioNamespace:
    def __init__(self, response) -> None:
        self.transcriptions = FakeAudioTranscriptions(response)


class FakeOpenAIClient:
    def __init__(self, response) -> None:
        self.audio = FakeAudioNamespace(response)


def _service_with_fake_client(response) -> TranscriptionService:
    service = TranscriptionService(api_key="fake-key", model="whisper-large-v3")
    service._client = FakeOpenAIClient(response)  # bypass the real OpenAI client
    return service


def test_transcribe_parses_object_shaped_segments():
    response = FakeTranscriptionResponse(
        text="hello world",
        language="en",
        segments=[FakeSegmentObject(start=0.0, end=1.0, text="hello world")],
    )
    service = _service_with_fake_client(response)

    result = service.transcribe(audio_bytes=b"fake audio", filename="a.mp3")

    assert result.text == "hello world"
    assert result.language == "en"
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.0
    assert result.segments[0].text == "hello world"


def test_transcribe_parses_dict_shaped_segments():
    response = FakeTranscriptionResponse(
        text="hello world",
        language="en",
        segments=[{"start": 0.0, "end": 1.0, "text": "hello world"}],
    )
    service = _service_with_fake_client(response)

    result = service.transcribe(audio_bytes=b"fake audio", filename="a.mp3")

    assert result.segments[0].text == "hello world"


def test_transcribe_handles_no_segments():
    response = FakeTranscriptionResponse(text="hello", language="en", segments=[])
    service = _service_with_fake_client(response)

    result = service.transcribe(audio_bytes=b"fake", filename="a.mp3")

    assert result.segments == []


def test_transcribe_rejects_empty_audio_bytes():
    service = _service_with_fake_client(FakeTranscriptionResponse("", "en", []))

    with pytest.raises(ValueError):
        service.transcribe(audio_bytes=b"", filename="a.mp3")
