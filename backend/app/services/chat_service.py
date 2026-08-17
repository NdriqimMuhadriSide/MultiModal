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
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import HTTPException, status

from ai.llm_service import LLMService, get_llm_service
from memory.compaction import ConversationCompactor, get_conversation_compactor
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
        compactor: ConversationCompactor | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._memory = memory
        self._history_limit = history_limit
        self._compactor = compactor

    def _load_history(self, conversation_id: str) -> list[dict[str, str]]:
        """
        The window, with the summary of everything before it in front.

        Without a compactor this is exactly what it always was - the last N
        messages - so a conversation shorter than the window, or a
        deployment with compaction switched off, behaves identically.
        """
        prefix = (
            self._compactor.summary_prefix(conversation_id) if self._compactor else []
        )
        window = self._memory.get_history(conversation_id, limit=self._history_limit)
        return prefix + [{"role": msg.role, "content": msg.content} for msg in window]

    def _compact(self, conversation_id: str) -> None:
        """Summarise anything that has just fallen out of the window."""
        if self._compactor:
            self._compactor.compact(conversation_id)

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

        history_messages = self._load_history(resolved_id)

        self._memory.add_message(resolved_id, role="user", content=message)

        answer = self._llm_service.generate_response(message, history=history_messages)

        self._memory.add_message(resolved_id, role="assistant", content=answer)
        self._compact(resolved_id)

        return ChatResult(conversation_id=resolved_id, answer=answer)

    def stream_answer(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[str, Iterator[str]]:
        """
        The streaming counterpart to `get_answer`: same memory flow, but the
        answer is yielded in pieces as the model produces it.

        Returns `(conversation_id, chunks)` rather than just the generator,
        because the caller needs the id *before* the first token in order to
        send it as the opening event - and resolving it lazily inside the
        generator would mean the client can't associate the stream with a
        conversation until the model has already started answering.

        Steps 1-3 (load history, store the user message) run eagerly, at call
        time. Only the LLM call is deferred to iteration.

        Where this genuinely differs from `get_answer`:

          - The assistant's turn is written once the stream finishes, not
            before, because there is nothing to write until then.
          - `finally` covers client disconnects. When a browser tab closes
            mid-answer the generator is closed and GeneratorExit is raised at
            the yield, so the partial text is still persisted. A user who saw
            half an answer has that half in their history, and the next turn's
            context matches what they actually read - the alternative is a
            conversation whose history silently disagrees with the screen.
          - Nothing is written when zero chunks arrived (a request that failed
            before the first token), which would otherwise store an empty
            assistant turn that `add_message` rejects anyway.
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()

        history_messages = self._load_history(resolved_id)

        self._memory.add_message(resolved_id, role="user", content=message)

        def chunks() -> Iterator[str]:
            collected: list[str] = []
            try:
                for delta in self._llm_service.stream_response(
                    message, history=history_messages
                ):
                    collected.append(delta)
                    yield delta
            finally:
                answer = "".join(collected)
                if answer.strip():
                    self._memory.add_message(
                        resolved_id, role="assistant", content=answer
                    )
                    # After the last token has been sent, so the summarising
                    # call is not something the reader waits on - the stream
                    # simply stays open a moment longer than it used to.
                    self._compact(resolved_id)

        return resolved_id, chunks()


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
            compactor=get_conversation_compactor(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chat service is not configured: {exc}",
        ) from exc
