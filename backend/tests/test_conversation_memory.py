import sqlite3

import pytest

from memory.conversation_memory import ConversationMemory


@pytest.fixture
def memory(tmp_path):
    return ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))


def _legacy_database(path) -> None:
    """
    Create a messages table in the shape this module wrote before modality
    and attachment_ref existed, with one row in it.

    Hand-written rather than produced by an older ConversationMemory,
    because the point is to pin the *old* schema: if this helper were kept
    in sync with the class it would stop testing the migration the moment
    the class changed.
    """
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES ('conv-1', 'user', 'written before the migration', '2026-01-01T00:00:00+00:00')"
        )


def test_add_message_rejects_empty_content(memory):
    with pytest.raises(ValueError):
        memory.add_message("conv-1", role="user", content="")


def test_get_history_rejects_non_positive_limit(memory):
    with pytest.raises(ValueError):
        memory.get_history("conv-1", limit=0)


def test_get_history_returns_empty_list_for_unknown_conversation(memory):
    assert memory.get_history("does-not-exist", limit=5) == []


def test_add_and_get_history_preserves_order(memory):
    memory.add_message("conv-1", role="user", content="What is RAG?")
    memory.add_message("conv-1", role="assistant", content="Retrieval-Augmented Generation.")
    memory.add_message("conv-1", role="user", content="And vector DBs?")

    history = memory.get_history("conv-1", limit=10)

    assert [msg.content for msg in history] == [
        "What is RAG?",
        "Retrieval-Augmented Generation.",
        "And vector DBs?",
    ]
    assert [msg.role for msg in history] == ["user", "assistant", "user"]
    assert all(msg.conversation_id == "conv-1" for msg in history)


def test_get_history_respects_limit_and_keeps_most_recent(memory):
    for i in range(5):
        memory.add_message("conv-1", role="user", content=f"message {i}")

    history = memory.get_history("conv-1", limit=2)

    # Should return the 2 most recent, still in chronological order.
    assert [msg.content for msg in history] == ["message 3", "message 4"]


def test_conversations_are_isolated_by_conversation_id(memory):
    memory.add_message("conv-1", role="user", content="conv 1 message")
    memory.add_message("conv-2", role="user", content="conv 2 message")

    history_1 = memory.get_history("conv-1", limit=10)
    history_2 = memory.get_history("conv-2", limit=10)

    assert [msg.content for msg in history_1] == ["conv 1 message"]
    assert [msg.content for msg in history_2] == ["conv 2 message"]


def test_get_full_history_returns_everything_regardless_of_history_limit(memory):
    for i in range(15):
        memory.add_message("conv-1", role="user", content=f"message {i}")

    full_history = memory.get_full_history("conv-1")

    assert len(full_history) == 15
    assert full_history[0].content == "message 0"
    assert full_history[-1].content == "message 14"


def test_messages_default_to_text_with_no_attachment(memory):
    memory.add_message("conv-1", role="user", content="plain text turn")

    message = memory.get_history("conv-1", limit=1)[0]

    assert message.modality == "text"
    assert message.attachment_ref is None


def test_modality_and_attachment_ref_round_trip(memory):
    memory.add_message(
        "conv-1",
        role="user",
        content="What colour is the car?",
        modality="image",
        attachment_ref="sha-abc",
    )

    message = memory.get_history("conv-1", limit=1)[0]

    assert message.modality == "image"
    assert message.attachment_ref == "sha-abc"
    # Both read paths carry the same fields - the window and the full
    # transcript disagreeing about a turn would be worse than either being
    # wrong on its own.
    assert memory.get_full_history("conv-1")[0].attachment_ref == "sha-abc"


def test_get_last_attachment_returns_the_most_recent_of_that_modality(memory):
    memory.add_message(
        "conv-1", role="user", content="first photo", modality="image", attachment_ref="img-1"
    )
    memory.add_message("conv-1", role="user", content="a text turn in between")
    memory.add_message(
        "conv-1", role="user", content="second photo", modality="image", attachment_ref="img-2"
    )

    assert memory.get_last_attachment("conv-1", modality="image") == "img-2"


def test_get_last_attachment_is_scoped_by_modality_and_conversation(memory):
    memory.add_message(
        "conv-1", role="user", content="a recording", modality="audio", attachment_ref="aud-1"
    )
    memory.add_message(
        "conv-2", role="user", content="a photo", modality="image", attachment_ref="img-1"
    )

    # An audio turn must not answer for an image one...
    assert memory.get_last_attachment("conv-1", modality="image") is None
    assert memory.get_last_attachment("conv-1", modality="audio") == "aud-1"
    # ...and neither may another conversation's.
    assert memory.get_last_attachment("conv-3", modality="image") is None


def test_get_last_attachment_ignores_the_history_window(memory):
    memory.add_message(
        "conv-1", role="user", content="the photo", modality="image", attachment_ref="img-1"
    )
    for i in range(30):
        memory.add_message("conv-1", role="user", content=f"later turn {i}")

    # Long past the 10-message window, but the conversation is still about
    # that picture.
    assert memory.get_last_attachment("conv-1", modality="image") == "img-1"


def test_opening_a_pre_migration_database_adds_the_new_columns(tmp_path):
    db_path = tmp_path / "conversations.sqlite3"
    _legacy_database(db_path)

    memory = ConversationMemory(db_path=str(db_path))

    # The old row is readable, and reads as what it actually was.
    existing = memory.get_history("conv-1", limit=10)
    assert [msg.content for msg in existing] == ["written before the migration"]
    assert existing[0].modality == "text"
    assert existing[0].attachment_ref is None

    # And the migrated table accepts the new columns.
    memory.add_message(
        "conv-1", role="user", content="after", modality="image", attachment_ref="img-1"
    )
    assert memory.get_last_attachment("conv-1", modality="image") == "img-1"


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "conversations.sqlite3"
    _legacy_database(db_path)

    ConversationMemory(db_path=str(db_path))
    # Every startup runs _init_schema; a second one must not fail on
    # "duplicate column name".
    memory = ConversationMemory(db_path=str(db_path))

    assert len(memory.get_full_history("conv-1")) == 1


def test_new_conversation_id_generates_unique_ids():
    first = ConversationMemory.new_conversation_id()
    second = ConversationMemory.new_conversation_id()

    assert first != second
    assert isinstance(first, str) and len(first) > 0
