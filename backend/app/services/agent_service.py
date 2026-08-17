"""
Supervisor agent business logic (FastAPI-facing wrapper).

Two things live here, unchanged in shape from when this wrapped the old
routing agent:

1. `get_supervisor_agent` - the dependency-injection seam where configuration
   errors turn into clean HTTPExceptions instead of raw 500s. No agent logic
   lives here; that is all in agents/supervisor_agent.py.

2. `AgentChatService` - the memory-aware orchestration around it:

       User -> AgentChatService.get_answer -> Memory Layer (load history,
               store the user message) -> SupervisorAgent (decide, delegate,
               answer) -> Memory Layer (store the answer) -> User

   Without this, agent turns would be invisible to every later message - the
   agent is stateless, and conversation history is read from the memory
   layer, not from the loop.

WHAT DELEGATION ADDED HERE

One thing: resolving the conversation's image. `/agent/ask` is a JSON
endpoint with no upload, but a conversation that has been through
`/vision/ask` is already carrying a picture in the attachment store, and
`get_last_attachment` has always been able to find it. Passing it in is what
lets the supervisor's `read_image` tool work at all from a text endpoint -
and it is why "does this receipt comply with our expense policy?" is now a
question this endpoint can answer.

The agent is handed the bytes rather than the memory layer, exactly as
VisionAgent is, so it stays ignorant of where pictures are kept.

WHAT IS STORED

Only the final answer, not the trace. The supervisor's steps are working
memory for one question: replaying "I asked the document specialist about
the refund window" into the *next* question's history would spend the
history budget on the last question's plumbing. This is the same call
app/services/research_service.py makes, for the same reason.
"""
from collections.abc import Iterator
from dataclasses import dataclass, field

from fastapi import HTTPException, status

from agents.agent_loop import AgentStep, StepEvent
from agents.supervisor_agent import (
    BoundImage,
    SupervisorAgent,
    SupervisorResultEvent,
)
from ai.llm_service import get_llm_service
from ai.vision_service import get_vision_service
from app.schemas.agent_trace import to_step_payload
from app.schemas.rag import RAGChatSource
from app.services.rag_service import build_retriever, to_chat_sources
from memory.attachment_store import AttachmentStore, get_attachment_store
from memory.compaction import ConversationCompactor, get_conversation_compactor
from memory.conversation_memory import ConversationMemory, get_conversation_memory
from rag.document_registry import get_document_registry


@dataclass
class AgentChatResult:
    conversation_id: str
    answer: str
    tool_used: str
    # Already projected onto the shape the chat UI renders, so the endpoint
    # stays a pure HTTP layer.
    sources: list[RAGChatSource] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    stopped_because: str = "finished"


class AgentChatService:
    """Wraps SupervisorAgent with the conversation memory layer."""

    def __init__(
        self,
        agent: SupervisorAgent,
        memory: ConversationMemory,
        attachments: AttachmentStore,
        history_limit: int = 10,
        compactor: ConversationCompactor | None = None,
    ) -> None:
        self._agent = agent
        self._memory = memory
        self._attachments = attachments
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

    def _bound_image(self, conversation_id: str) -> BoundImage | None:
        """
        The picture this conversation is carrying, if any.

        None covers every ordinary text conversation and, deliberately, the
        case where a conversation *had* an image whose file has since gone
        (a database moved without its attachment directory). Neither is an
        error here: the supervisor's `read_image` tool reports "there is no
        image" as an observation and carries on, which is the right outcome
        for a question that turned out not to need one.

        Not bounded by the history window, matching
        `get_last_attachment`'s own reasoning: a picture does not stop being
        the subject because ten messages have gone by.
        """
        ref = self._memory.get_last_attachment(conversation_id, modality="image")
        if not ref:
            return None

        attachment = self._attachments.load(ref)
        if attachment is None:
            return None

        return BoundImage(data=attachment.data, mime_type=attachment.mime_type)

    def get_answer(
        self, message: str, conversation_id: str | None = None
    ) -> AgentChatResult:
        """
        Answer `message` via the supervisor, using and updating the
        conversation identified by `conversation_id` (a new one is generated
        if absent).

        Raises:
            ValueError: if `message` is empty (propagated from the agent).
            RuntimeError: if an LLM call fails (propagated from llm_service).
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()

        history_messages = self._load_history(resolved_id)
        image = self._bound_image(resolved_id)

        self._memory.add_message(resolved_id, role="user", content=message)

        result = self._agent.run(message, history=history_messages, image=image)

        self._memory.add_message(resolved_id, role="assistant", content=result.answer)
        self._compact(resolved_id)

        return AgentChatResult(
            conversation_id=resolved_id,
            answer=result.answer,
            tool_used=result.tool_used,
            sources=to_chat_sources(result.sources),
            steps=result.steps,
            stopped_because=result.stopped_because,
        )

    def stream_answer(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[str, Iterator[dict]]:
        """
        The streaming counterpart to `get_answer`.

        Returns `(conversation_id, events)`. The events are dicts because
        several different things reach the client in order:

            {"type": "step", "index": 1, "depth": 0, "step": {...}}
            {"type": "tool", "tool": "research_documents"}
            {"type": "sources", "sources": [...]}
            {"type": "answer", "content": "..."}
            {"type": "done", "stopped_because": "finished"}

        STEPS, NOT TOKENS - AND WHY THAT CHANGED

        This endpoint used to stream token deltas, because the old routing
        agent picked one tool and then streamed that tool's generation
        straight through. A supervisor cannot: its answer is already whole
        inside the `finish` action by the time the loop sees it, so there is
        no token stream left to forward.

        What replaces it is better suited to the work anyway. A delegating
        run is several seconds of thinking with no prose at all, and a step
        frame - "asking the document specialist about the refund window" -
        arrives the moment each tool returns. `depth` distinguishes the
        supervisor's own steps from a specialist's, so the client can indent
        rather than presenting a specialist's search as something the
        supervisor did.

        `tool` still arrives before the answer, as it always did, so the
        bubble can be labelled while the run is still going. It is sent once
        the run resolves rather than eagerly, because with delegation the
        honest label depends on what the whole turn ended up using.

        The assistant's turn is persisted when the stream ends, including
        when it ends early: `finally` runs on GeneratorExit, so a browser tab
        closed mid-run still records whatever was produced.
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()

        history_messages = self._load_history(resolved_id)
        image = self._bound_image(resolved_id)

        self._memory.add_message(resolved_id, role="user", content=message)

        def events() -> Iterator[dict]:
            answer = ""
            try:
                stream = self._agent.stream(
                    message, history=history_messages, image=image
                )
                for index, event in enumerate(stream, start=1):
                    if isinstance(event, StepEvent):
                        yield {
                            "type": "step",
                            "index": index,
                            "depth": event.depth,
                            "step": to_step_payload(event.step),
                        }
                        continue

                    result = event.result
                    answer = result.answer
                    yield {"type": "tool", "tool": result.tool_used}
                    if result.sources:
                        # by_alias so a streamed citation is byte-for-byte
                        # the one POST /agent/ask and POST /rag/chat return
                        # (chunkId, not chunk_id) - the frontend has a single
                        # type for all three.
                        yield {
                            "type": "sources",
                            "sources": [
                                source.model_dump(by_alias=True)
                                for source in to_chat_sources(result.sources)
                            ],
                        }
                    yield {"type": "answer", "content": result.answer}
                    yield {
                        "type": "done",
                        "stopped_because": result.stopped_because,
                    }
            finally:
                if answer.strip():
                    self._memory.add_message(
                        resolved_id, role="assistant", content=answer
                    )
                    # After the last frame, so the summarising call is not
                    # something the reader waits on.
                    self._compact(resolved_id)

        return resolved_id, events()


def get_supervisor_agent() -> SupervisorAgent:
    """
    FastAPI dependency that builds a SupervisorAgent from shared singletons.

    Built per request rather than cached, like every other agent factory
    here. The specialists it constructs are cheap wrappers around the shared
    retriever, vision and LLM singletons - and the budget and evidence ledger
    it owns are per-run state, which a process-wide instance would have to be
    trusted to reset rather than simply not having to share.
    """
    from app.core.config import settings

    try:
        return SupervisorAgent(
            llm_service=get_llm_service(),
            vision_service=get_vision_service(),
            retriever=build_retriever(),
            document_registry=get_document_registry(),
            max_steps=settings.supervisor_max_steps,
            tree_budget=settings.supervisor_tree_budget,
            research_max_steps=settings.research_max_steps,
            vision_max_steps=settings.vision_agent_max_steps,
            search_top_k=settings.research_search_top_k,
            temperature=settings.agent_temperature,
            critic_enabled=settings.supervisor_critic_enabled,
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
        agent=get_supervisor_agent(),
        memory=get_conversation_memory(),
        attachments=get_attachment_store(),
        history_limit=settings.conversation_history_limit,
        compactor=get_conversation_compactor(),
    )
