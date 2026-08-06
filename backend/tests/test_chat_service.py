from app.services.chat_service import ChatService
from memory.conversation_memory import ConversationMemory


class FakeLLMService:
    def __init__(self, response: str = "a response") -> None:
        self.response = response
        self.last_message: str | None = None
        self.last_history: list[dict] | None = None

    def generate_response(self, user_message: str, history: list[dict] | None = None) -> str:
        self.last_message = user_message
        self.last_history = history
        return self.response


def test_get_answer_generates_new_conversation_id_when_none_given(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    llm = FakeLLMService()
    service = ChatService(llm_service=llm, memory=memory)

    result = service.get_answer("hello")

    assert result.conversation_id  # non-empty, generated
    assert result.answer == "a response"


def test_get_answer_stores_user_message_and_assistant_response(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    llm = FakeLLMService(response="hi there")
    service = ChatService(llm_service=llm, memory=memory)

    result = service.get_answer("hello", conversation_id="conv-1")

    stored = memory.get_full_history("conv-1")
    assert [(m.role, m.content) for m in stored] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]
    assert result.conversation_id == "conv-1"


def test_get_answer_passes_prior_history_to_llm(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    memory.add_message("conv-1", role="user", content="What is RAG?")
    memory.add_message("conv-1", role="assistant", content="Retrieval-Augmented Generation.")

    llm = FakeLLMService(response="It combines search and generation.")
    service = ChatService(llm_service=llm, memory=memory)

    service.get_answer("Can you say more?", conversation_id="conv-1")

    assert llm.last_message == "Can you say more?"
    assert llm.last_history == [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "Retrieval-Augmented Generation."},
    ]


def test_get_answer_respects_history_limit(tmp_path):
    memory = ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))
    for i in range(5):
        memory.add_message("conv-1", role="user", content=f"message {i}")

    llm = FakeLLMService()
    service = ChatService(llm_service=llm, memory=memory, history_limit=2)

    service.get_answer("new message", conversation_id="conv-1")

    # Only the 2 most recent prior messages should be sent as history.
    assert llm.last_history == [
        {"role": "user", "content": "message 3"},
        {"role": "user", "content": "message 4"},
    ]
