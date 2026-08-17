"""
Research agent business logic (FastAPI-facing wrapper).

Same two responsibilities as app/services/agent_service.py, for the other
agent:

1. `get_research_agent` - the dependency-injection seam where configuration
   errors become clean HTTPExceptions instead of raw 500s. No loop logic
   lives here; that is all in agents/research_agent.py.

2. `ResearchChatService` - the memory-aware orchestration around the agent:

       User -> ResearchChatService.research -> Memory Layer (load history,
               store the question) -> ResearchAgent (multi-hop loop) ->
               Memory Layer (store the answer) -> User

   The agent itself is stateless, so without this a research turn would be
   invisible to every later message - and follow-ups are exactly where this
   agent needs history most, since "and what about the other one?" has to
   resolve to a subject before its first search can be any good.

Only the final answer is written to memory, not the intermediate steps. The
scratchpad is working memory for one question: replaying "I searched for
refund window and found three passages" into the *next* question's history
would spend the whole history budget on the last question's plumbing.
"""
from dataclasses import dataclass, field

from fastapi import HTTPException, status

from agents.research_agent import ResearchAgent, ResearchStep
from ai.llm_service import get_llm_service
from app.core.config import settings
from app.schemas.rag import RAGChatSource
from app.services.rag_service import build_retriever, to_chat_sources
from memory.compaction import ConversationCompactor, get_conversation_compactor
from memory.conversation_memory import ConversationMemory, get_conversation_memory
from rag.document_registry import get_document_registry


@dataclass
class ResearchChatResult:
    conversation_id: str
    answer: str
    steps: list[ResearchStep] = field(default_factory=list)
    sources: list[RAGChatSource] = field(default_factory=list)
    stopped_because: str = "finished"


class ResearchChatService:
    """Wraps ResearchAgent with the conversation memory layer."""

    def __init__(
        self,
        agent: ResearchAgent,
        memory: ConversationMemory,
        history_limit: int = 10,
        compactor: ConversationCompactor | None = None,
    ) -> None:
        self._agent = agent
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
        return prefix + [
            {"role": message.role, "content": message.content} for message in window
        ]

    def _compact(self, conversation_id: str) -> None:
        """Summarise anything that has just fallen out of the window."""
        if self._compactor:
            self._compactor.compact(conversation_id)

    def research(
        self, question: str, conversation_id: str | None = None
    ) -> ResearchChatResult:
        """
        Research `question`, recording the turn in the conversation
        identified by `conversation_id` (a new one is generated if absent).

        Mirrors AgentChatService.get_answer, so research turns land in the
        same conversations table as chat and agent turns and read back
        identically via GET /chat/{id}/history.

        Raises:
            ValueError: if `question` is empty (propagated from the agent).
            RuntimeError: if an LLM call fails (propagated from llm_service).
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()

        history_messages = self._load_history(resolved_id)

        self._memory.add_message(resolved_id, role="user", content=question)

        result = self._agent.run(question, history=history_messages)

        self._memory.add_message(
            resolved_id, role="assistant", content=result.answer
        )
        self._compact(resolved_id)

        return ResearchChatResult(
            conversation_id=resolved_id,
            answer=result.answer,
            steps=result.steps,
            sources=to_chat_sources(result.sources),
            stopped_because=result.stopped_because,
        )


def get_research_agent() -> ResearchAgent:
    """
    FastAPI dependency that builds a ResearchAgent from shared singletons.

    Takes a Retriever rather than a RAGService - see the module docstring in
    agents/research_agent.py for why evidence, not answers, is what a
    multi-hop loop needs from retrieval. Every agent now takes retrieval this
    way; RAGService is left to /rag/chat, which genuinely does want one
    retrieve-and-answer pass.
    """
    try:
        return ResearchAgent(
            llm_service=get_llm_service(),
            retriever=build_retriever(),
            document_registry=get_document_registry(),
            max_steps=settings.research_max_steps,
            search_top_k=settings.research_search_top_k,
            temperature=settings.agent_temperature,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Research agent is not configured: {exc}",
        ) from exc


def get_research_chat_service() -> ResearchChatService:
    """FastAPI dependency that builds a memory-aware ResearchChatService."""
    return ResearchChatService(
        agent=get_research_agent(),
        memory=get_conversation_memory(),
        history_limit=settings.conversation_history_limit,
        compactor=get_conversation_compactor(),
    )
