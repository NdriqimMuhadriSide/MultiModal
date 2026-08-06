"""
Agent business logic (FastAPI-facing wrapper).

Two things live here:

1. `get_assistant_agent` - the dependency-injection seam where configuration
   errors turn into clean HTTPExceptions instead of raw 500s. No agent/graph
   logic lives here; that's all in agents/assistant_agent.py.

2. `AgentChatService` - the memory-aware orchestration around the agent,
   mirroring app/services/chat_service.py's ChatService exactly:

       User -> AgentChatService.get_answer -> Memory Layer (load history,
               store the user message) -> Agent (route + run one tool) ->
               Memory Layer (store the answer) -> User

   Without this, agent turns would be invisible to every later message -
   the agent itself is stateless, and conversation history is read from the
   memory layer, not from the graph.
"""
from dataclasses import dataclass

from fastapi import HTTPException, status

from agents.assistant_agent import AssistantAgent
from ai.llm_service import get_llm_service
from app.services.rag_service import get_rag_service
from memory.conversation_memory import ConversationMemory, get_conversation_memory


@dataclass
class AgentChatResult:
    conversation_id: str
    answer: str
    tool_used: str


class AgentChatService:
    """Wraps AssistantAgent with the conversation memory layer."""

    def __init__(
        self,
        agent: AssistantAgent,
        memory: ConversationMemory,
        history_limit: int = 10,
    ) -> None:
        self._agent = agent
        self._memory = memory
        self._history_limit = history_limit

    def get_answer(
        self, message: str, conversation_id: str | None = None
    ) -> AgentChatResult:
        """
        Answer `message` via the agent, using and updating the conversation
        identified by `conversation_id` (a new one is generated if absent).

        Same flow as ChatService.get_answer - load history, store the user
        turn, generate, store the answer - so agent conversations land in
        the same conversations table as plain chat ones and read back
        identically via GET /chat/{id}/history.

        Raises:
            ValueError: if `message` is empty (propagated from the agent).
            RuntimeError: if an LLM call fails (propagated from llm_service).
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()

        history = self._memory.get_history(resolved_id, limit=self._history_limit)
        history_messages = [{"role": msg.role, "content": msg.content} for msg in history]

        self._memory.add_message(resolved_id, role="user", content=message)

        result = self._agent.run(message, history=history_messages)

        self._memory.add_message(resolved_id, role="assistant", content=result.answer)

        return AgentChatResult(
            conversation_id=resolved_id,
            answer=result.answer,
            tool_used=result.tool_used,
        )


def get_assistant_agent() -> AssistantAgent:
    """FastAPI dependency that builds an AssistantAgent from shared singletons."""
    try:
        return AssistantAgent(
            llm_service=get_llm_service(),
            rag_service=get_rag_service(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent is not configured: {exc}",
        ) from exc


def get_agent_chat_service() -> AgentChatService:
    """FastAPI dependency that builds a memory-aware AgentChatService."""
    from app.core.config import settings

    return AgentChatService(
        agent=get_assistant_agent(),
        memory=get_conversation_memory(),
        history_limit=settings.conversation_history_limit,
    )
