import io
import wave

from fastapi.testclient import TestClient

from app.main import app
from app.services.audio_service import (
    AudioAnalysisResult,
    AudioAnalysisService,
    get_audio_analysis_service,
)
from processors.audio.audio_metadata import AudioMetadata
from processors.audio.audio_validator import AudioValidationError
from processors.audio.transcriber import Transcript, TranscriptSegment

client = TestClient(app)


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 8000)
    return buffer.getvalue()


class StubAudioAnalysisService(AudioAnalysisService):
    """Bypasses the real Whisper/LLM calls so tests don't hit Groq."""

    def __init__(self) -> None:  # no super().__init__ - no real collaborators needed
        pass

    def analyze(self, filename, mime_type, audio_bytes, question):
        return AudioAnalysisResult(
            transcript=Transcript(
                text="We decided to ship the feature on Friday.",
                language="en",
                segments=[TranscriptSegment(start=0.0, end=2.0, text="We decided to ship the feature on Friday.")],
            ),
            analysis=f"stub analysis for question: {question!r}",
            metadata=AudioMetadata(
                filename=filename,
                duration_seconds=2.0,
                sample_rate=16000,
                channels=1,
                size_bytes=len(audio_bytes),
            ),
        )


class FailingValidationAudioService(AudioAnalysisService):
    def __init__(self) -> None:
        pass

    def analyze(self, filename, mime_type, audio_bytes, question):
        raise AudioValidationError(f"Unsupported audio file '{filename}'.")


def test_audio_analyze_returns_transcript_analysis_and_metadata():
    app.dependency_overrides[get_audio_analysis_service] = lambda: StubAudioAnalysisService()
    try:
        response = client.post(
            "/api/v1/audio/analyze",
            files={"audio": ("meeting.wav", _wav_bytes(), "audio/wav")},
            data={"question": "Summarize the key decisions."},
        )
    finally:
        app.dependency_overrides.pop(get_audio_analysis_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "We decided to ship the feature on Friday."
    assert "Summarize the key decisions." in body["analysis"]
    assert body["metadata"]["filename"] == "meeting.wav"
    assert body["metadata"]["duration"] == 2.0
    assert body["metadata"]["sample_rate"] == 16000
    assert body["metadata"]["channels"] == 1
    assert body["metadata"]["size"] == len(_wav_bytes())


def test_audio_analyze_question_is_optional():
    app.dependency_overrides[get_audio_analysis_service] = lambda: StubAudioAnalysisService()
    try:
        response = client.post(
            "/api/v1/audio/analyze",
            files={"audio": ("meeting.wav", _wav_bytes(), "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_audio_analysis_service, None)

    assert response.status_code == 200
    assert "None" in response.json()["analysis"]


def test_audio_analyze_requires_audio_file():
    app.dependency_overrides[get_audio_analysis_service] = lambda: StubAudioAnalysisService()
    try:
        response = client.post(
            "/api/v1/audio/analyze",
            data={"question": "Summarize this."},
        )
    finally:
        app.dependency_overrides.pop(get_audio_analysis_service, None)

    assert response.status_code == 422


def test_audio_analyze_maps_validation_error_to_bad_request():
    app.dependency_overrides[get_audio_analysis_service] = lambda: FailingValidationAudioService()
    try:
        response = client.post(
            "/api/v1/audio/analyze",
            files={"audio": ("notes.txt", b"not audio", "text/plain")},
        )
    finally:
        app.dependency_overrides.pop(get_audio_analysis_service, None)

    assert response.status_code == 400
    assert "notes.txt" in response.json()["detail"]


def test_audio_analyze_maps_runtime_error_to_bad_gateway():
    class FailingTranscriptionService(AudioAnalysisService):
        def __init__(self) -> None:
            pass

        def analyze(self, filename, mime_type, audio_bytes, question):
            raise RuntimeError("Transcription request failed: connection refused")

    app.dependency_overrides[get_audio_analysis_service] = lambda: FailingTranscriptionService()
    try:
        response = client.post(
            "/api/v1/audio/analyze",
            files={"audio": ("meeting.wav", _wav_bytes(), "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_audio_analysis_service, None)

    assert response.status_code == 502
