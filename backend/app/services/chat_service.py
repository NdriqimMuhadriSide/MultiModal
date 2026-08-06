"""
Chat business logic.

Sits between the API route and the AI service, now wired through the
conversation memory layer:

    User -> ChatService.get_answer -> Memory Layer (load history,
            store the new user message) -> LLM (history + message) ->
            Memory Layer (store the assistant's response) -> User

Every call is scoped to a conversation_id, so multiple independent
conversations don't bleed into each other's history. If no conversation_id
is given, a new one is generated - see get_or_create_conversation_id.
"""
from dataclasses import dataclass

from fastapi import HTTPException, status

from ai.llm_service import LLMService, get_llm_service
from memory.conversation_memory import ConversationMemory, get_conversation_memory


@dataclass
class ChatResult:
    conversation_id: str
    answer: str


class ChatService:
    def __init__(
        self,
        llm_service: LLMService,
        memory: ConversationMemory,
        history_limit: int = 10,
    ) -> None:
        self._llm_service = llm_service
        self._memory = memory
        self._history_limit = history_limit

    def get_answer(self, message: str, conversation_id: str | None = None) -> ChatResult:
        """
        Answer `message`, using and updating the conversation identified by
        `conversation_id` (a new one is generated if not provided).

        Memory layer flow:
          1. Load recent history for this conversation_id (short-term
             memory window, bounded by history_limit).
          2. Store the incoming user message.
          3. Send (history + message) to the LLM.
          4. Store the assistant's response.
          5. Return the answer along with the conversation_id, so the
             caller can keep using it for follow-up turns.
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()

        history = self._memory.get_history(resolved_id, limit=self._history_limit)
        history_messages = [{"role": msg.role, "content": msg.content} for msg in history]

        self._memory.add_message(resolved_id, role="user", content=message)

        answer = self._llm_service.generate_response(message, history=history_messages)

        self._memory.add_message(resolved_id, role="assistant", content=answer)

        return ChatResult(conversation_id=resolved_id, answer=answer)


def get_chat_service() -> ChatService:
    """
    FastAPI dependency that builds a ChatService.

    Configuration errors (e.g. missing GROQ_API_KEY) happen here, during
    dependency resolution - outside the route handler's try/except - so
    they're translated into a clean HTTPException instead of leaking a raw
    stack trace to the client as an unhandled 500.
    """
    from app.core.config import settings

    try:
        return ChatService(
            llm_service=get_llm_service(),
            memory=get_conversation_memory(),
            history_limit=settings.conversation_history_limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chat service is not configured: {exc}",
        ) from exc
