import pytest

from memory.conversation_memory import ConversationMemory


@pytest.fixture
def memory(tmp_path):
    return ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))


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


def test_new_conversation_id_generates_unique_ids():
    first = ConversationMemory.new_conversation_id()
    second = ConversationMemory.new_conversation_id()

    assert first != second
    assert isinstance(first, str) and len(first) > 0
