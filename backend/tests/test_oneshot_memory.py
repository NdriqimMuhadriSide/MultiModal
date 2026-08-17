"""
What POST /vision/analyze and POST /audio/analyze leave behind.

Both used to be pure functions - bytes in, string out, nothing kept - which
made them the two modalities whose output vanished with the response. These
pin the record they now write.
"""
import pytest

from app.services.audio_service import AudioAnalysisService
from app.services.vision_service import VisionAnalysisService
from memory.attachment_store import AttachmentStore
from memory.conversation_memory import ConversationMemory
from processors.audio.audio_metadata import AudioMetadata
from processors.audio.transcriber import Transcript


@pytest.fixture
def memory(tmp_path):
    return ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))


@pytest.fixture
def attachments(tmp_path):
    return AttachmentStore(root_dir=str(tmp_path / "attachments"))


# ---- Vision ----------------------------------------------------------------


class FakeVisionService:
    def __init__(self, answer: str = "A red car.") -> None:
        self.answer = answer
        self.failure: Exception | None = None

    def analyze_image(self, image_bytes, mime_type, question):
        if self.failure is not None:
            raise self.failure
        return self.answer


def test_a_vision_analysis_is_recorded(memory, attachments):
    service = VisionAnalysisService(
        vision_service=FakeVisionService(), memory=memory, attachments=attachments
    )

    result = service.analyze(b"png bytes", "image/png", "What is this?")

    stored = memory.get_full_history(result.conversation_id)
    assert [(msg.role, msg.content) for msg in stored] == [
        ("user", "What is this?"),
        ("assistant", "A red car."),
    ]
    assert {msg.modality for msg in stored} == {"image"}
    assert all(msg.attachment_ref for msg in stored)


def test_a_vision_analysis_can_join_an_existing_conversation(memory, attachments):
    service = VisionAnalysisService(
        vision_service=FakeVisionService(), memory=memory, attachments=attachments
    )
    memory.add_message("conv-1", role="user", content="Hello")

    result = service.analyze(b"png bytes", "image/png", "What is this?", "conv-1")

    assert result.conversation_id == "conv-1"
    assert len(memory.get_full_history("conv-1")) == 3


def test_the_image_survives_a_failed_vision_call(memory, attachments, tmp_path):
    vision = FakeVisionService()
    vision.failure = RuntimeError("Vision request failed: upstream timeout")
    service = VisionAnalysisService(
        vision_service=vision, memory=memory, attachments=attachments
    )

    with pytest.raises(RuntimeError):
        service.analyze(b"png bytes", "image/png", "What is this?")

    # No turn was written - there is no answer to record - but the upload is
    # kept, so asking again costs no second upload.
    assert list((tmp_path / "attachments").iterdir())


def test_a_vision_turn_can_be_inherited_by_the_agent(memory, attachments):
    """
    The point of storing the ref: a follow-up to POST /vision/ask can pick
    up the image POST /vision/analyze was given.
    """
    service = VisionAnalysisService(
        vision_service=FakeVisionService(), memory=memory, attachments=attachments
    )
    result = service.analyze(b"png bytes", "image/png", "What is this?")

    ref = memory.get_last_attachment(result.conversation_id, modality="image")
    assert attachments.load(ref).data == b"png bytes"


# ---- Audio -----------------------------------------------------------------


class FakeAudioProcessor:
    def __init__(self, transcript_text: str) -> None:
        self._transcript_text = transcript_text

    def process(self, filename, mime_type, audio_bytes):
        class _Result:
            transcript = Transcript(text=self._transcript_text, language="en", segments=[])
            metadata = AudioMetadata(
                filename=filename,
                duration_seconds=2.0,
                sample_rate=16000,
                channels=1,
                size_bytes=len(audio_bytes),
            )

        return _Result()


class FakeLLMService:
    def generate_response(self, prompt, history=None, system_prompt=None, temperature=None):
        return "They decided to ship on Friday."


def _audio_service(memory, attachments, transcript: str, chars: int = 4000):
    return AudioAnalysisService(
        audio_processor=FakeAudioProcessor(transcript),
        llm_service=FakeLLMService(),
        memory=memory,
        attachments=attachments,
        transcript_memory_chars=chars,
    )


def test_an_audio_analysis_records_the_transcript(memory, attachments):
    service = _audio_service(memory, attachments, "We decided to ship on Friday.")

    result = service.analyze(
        "meeting.wav", "audio/wav", b"wav bytes", "What was decided?"
    )

    stored = memory.get_full_history(result.conversation_id)
    assert stored[0].role == "user"
    assert "What was decided?" in stored[0].content
    # The transcript is the reusable artefact - minutes of speech turned
    # into text at real cost. Losing it means paying that cost again.
    assert "We decided to ship on Friday." in stored[0].content
    assert stored[1].content == "They decided to ship on Friday."
    assert {msg.modality for msg in stored} == {"audio"}


def test_an_audio_turn_without_a_question_still_reads_as_something_said(
    memory, attachments
):
    service = _audio_service(memory, attachments, "Some speech.")

    result = service.analyze("meeting.wav", "audio/wav", b"wav bytes", None)

    assert memory.get_full_history(result.conversation_id)[0].content.startswith(
        "Summarise this audio."
    )


def test_a_long_transcript_is_clipped_and_says_so(memory, attachments):
    service = _audio_service(memory, attachments, "word " * 2000, chars=100)

    result = service.analyze("meeting.wav", "audio/wav", b"wav bytes", "Summarise")

    content = memory.get_full_history(result.conversation_id)[0].content
    assert "transcript truncated" in content
    # Announced rather than silent: a transcript that just stops invites the
    # model to answer as though that were the whole recording.
    assert "10000 characters in full" in content


def test_silence_records_the_turn_without_a_transcript_block(memory, attachments):
    service = _audio_service(memory, attachments, "")

    result = service.analyze("silence.wav", "audio/wav", b"wav bytes", "What is said?")

    content = memory.get_full_history(result.conversation_id)[0].content
    assert content == "What is said?"
    assert "Transcript" not in content


def test_the_audio_file_is_stored_and_retrievable(memory, attachments):
    service = _audio_service(memory, attachments, "Some speech.")

    result = service.analyze("meeting.wav", "audio/wav", b"wav bytes", None)

    ref = memory.get_last_attachment(result.conversation_id, modality="audio")
    assert attachments.load(ref).data == b"wav bytes"


def test_an_unstorable_container_still_records_the_analysis(memory, attachments):
    service = _audio_service(memory, attachments, "Some speech.")

    # A format the validator let through but the attachment store has no
    # extension for. Losing the user's analysis over that would be a poor
    # trade for keeping the file.
    result = service.analyze("odd.aiff", "audio/aiff", b"aiff bytes", "Summarise")

    stored = memory.get_full_history(result.conversation_id)
    assert stored[1].content == "They decided to ship on Friday."
    assert all(msg.attachment_ref is None for msg in stored)
