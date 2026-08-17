"""
What a vision turn remembers.

These exercise the service against a real ConversationMemory and a real
AttachmentStore, with only the agent faked - the behaviour under test is
entirely about what gets stored and what comes back out, so a fake memory
would be testing the fake.
"""
from dataclasses import dataclass, field

import pytest

from app.services.vision_agent_service import VisionAgentChatService
from memory.attachment_store import AttachmentStore
from memory.conversation_memory import ConversationMemory

PNG = "image/png"
IMAGE_A = b"\x89PNG image a"
IMAGE_B = b"\x89PNG image b"


@dataclass
class FakeResult:
    answer: str
    steps: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    stopped_because: str = "finished"
    unverified_values: list = field(default_factory=list)


class FakeAgent:
    """Records what it was asked, answers with a canned string."""

    def __init__(self, answer: str = "It is red.") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def run(self, question, image_bytes, mime_type, history=None):
        self.calls.append(
            {
                "question": question,
                "image_bytes": image_bytes,
                "mime_type": mime_type,
                "history": history or [],
            }
        )
        return FakeResult(answer=self.answer)

    @property
    def last_history(self) -> list[dict]:
        return self.calls[-1]["history"]


@pytest.fixture
def service(tmp_path):
    def build(agent: FakeAgent) -> VisionAgentChatService:
        return VisionAgentChatService(
            agent=agent,
            memory=ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3")),
            attachments=AttachmentStore(root_dir=str(tmp_path / "attachments")),
        )

    return build


def test_a_turn_records_the_image_it_was_about(service, tmp_path):
    agent = FakeAgent()
    built = service(agent)

    result = built.analyze("What colour is the car?", IMAGE_A, PNG)

    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    stored = memory.get_full_history(result.conversation_id)

    assert [msg.role for msg in stored] == ["user", "assistant"]
    assert {msg.modality for msg in stored} == {"image"}
    # Both halves point at the same picture - the answer is the half a later
    # turn reads back, so it is the half that most needs the ref.
    refs = {msg.attachment_ref for msg in stored}
    assert len(refs) == 1 and None not in refs


def test_history_marks_a_turn_about_a_different_image(service):
    agent = FakeAgent()
    built = service(agent)

    first = built.analyze("What colour is the car?", IMAGE_A, PNG)
    built.analyze("Is it still the same colour?", IMAGE_B, PNG, conversation_id=first.conversation_id)

    # This is the bug the ref exists for: without the label, "It is red."
    # arrives as plain history and reads as a statement about IMAGE_B.
    history = agent.last_history
    assert len(history) == 2
    assert all("DIFFERENT image" in turn["content"] for turn in history)
    assert "It is red." in history[1]["content"]


def test_history_marks_a_turn_about_the_same_image(service):
    agent = FakeAgent()
    built = service(agent)

    first = built.analyze("What colour is the car?", IMAGE_A, PNG)
    built.analyze("And the wheels?", IMAGE_A, PNG, conversation_id=first.conversation_id)

    history = agent.last_history
    assert all("DIFFERENT" not in turn["content"] for turn in history)
    assert all("about the image in this message" in turn["content"] for turn in history)


def test_a_follow_up_can_omit_the_image(service):
    agent = FakeAgent()
    built = service(agent)

    first = built.analyze("What colour is the car?", IMAGE_A, PNG)
    built.analyze("And the wheels?", None, "", conversation_id=first.conversation_id)

    # The agent still got a picture - the one the conversation is about -
    # with its mime type recovered from the store.
    assert agent.calls[-1]["image_bytes"] == IMAGE_A
    assert agent.calls[-1]["mime_type"] == PNG
    # And it counts as the same image, not a different one.
    assert all("DIFFERENT" not in turn["content"] for turn in agent.last_history)


def test_an_inherited_image_stays_available_past_the_history_window(service, tmp_path):
    agent = FakeAgent()
    built = VisionAgentChatService(
        agent=agent,
        memory=ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3")),
        attachments=AttachmentStore(root_dir=str(tmp_path / "attachments")),
        history_limit=2,
    )

    first = built.analyze("What colour is the car?", IMAGE_A, PNG)
    for _ in range(5):
        built.analyze("And again?", None, "", conversation_id=first.conversation_id)

    # Long past a 2-message window, the conversation is still about it.
    assert agent.calls[-1]["image_bytes"] == IMAGE_A


def test_asking_with_no_image_and_no_history_is_a_value_error(service):
    built = service(FakeAgent())

    with pytest.raises(ValueError, match="No image was provided"):
        built.analyze("What colour is the car?", None, "")


def test_asking_with_no_image_when_the_file_is_gone_is_a_value_error(service, tmp_path):
    agent = FakeAgent()
    built = service(agent)
    first = built.analyze("What colour is the car?", IMAGE_A, PNG)

    # A database moved without its attachment directory: the ref is still on
    # the turn, the bytes are not there. Re-uploading is the fix, so the
    # caller has to be told rather than handed a turn with no image.
    for path in (tmp_path / "attachments").iterdir():
        path.unlink()

    with pytest.raises(ValueError, match="No image was provided"):
        built.analyze("And the wheels?", None, "", conversation_id=first.conversation_id)


def test_the_same_image_is_stored_once_across_turns(service, tmp_path):
    built = service(FakeAgent())

    first = built.analyze("What colour is the car?", IMAGE_A, PNG)
    built.analyze("And the wheels?", IMAGE_A, PNG, conversation_id=first.conversation_id)

    assert len(list((tmp_path / "attachments").iterdir())) == 1


def test_plain_text_turns_are_passed_through_unlabelled(service, tmp_path):
    agent = FakeAgent()
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    built = VisionAgentChatService(
        agent=agent,
        memory=memory,
        attachments=AttachmentStore(root_dir=str(tmp_path / "attachments")),
    )

    # A conversation that started in /chat and moved to the vision agent.
    memory.add_message("conv-1", role="user", content="Hello")
    memory.add_message("conv-1", role="assistant", content="Hi - how can I help?")

    built.analyze("What colour is the car?", IMAGE_A, PNG, conversation_id="conv-1")

    assert [turn["content"] for turn in agent.last_history] == [
        "Hello",
        "Hi - how can I help?",
    ]
