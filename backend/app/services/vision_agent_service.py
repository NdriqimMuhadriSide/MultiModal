"""
Vision agent business logic (FastAPI-facing wrapper).

Same two responsibilities as app/services/research_service.py, for the
image-reading agent:

1. `get_vision_agent` - the dependency-injection seam where configuration
   errors become clean HTTPExceptions instead of raw 500s.

2. `VisionAgentChatService` - the memory-aware orchestration around it.

WHAT A TURN REMEMBERS

The question and the answer go to conversation memory as before, but both
now carry the image they were about: modality "image", and the
memory/attachment_store.py ref of the picture itself. That single addition
is what closes a hole this service used to have.

Before it, a stored turn was text with nothing tying it to a photograph.
Ask "what colour is the car?" about one image, then upload a different one
and ask "is it still the same colour?", and the earlier "It's red" arrived
in the prompt as ordinary conversation history - indistinguishable from
something said about the picture now in front of the model. The answer
that came back looked like a hallucination and wasn't: the context really
did say red, about a car the model could no longer see.

So history is labelled on the way into the prompt. A turn about the
current image says so; a turn about a different one says that instead. The
model is told which of its own previous statements apply, rather than
being left to assume they all do.

FOLLOW-UPS WITHOUT RE-UPLOADING

Because the bytes are kept, the image is now optional on a request that
names a conversation: omit it and the last image in that conversation is
loaded back and used. "And what about the date?" is finally a question
this endpoint can answer, where before it needed the file attached again
every single time.

What that costs, stated plainly: attachments are never deleted, and a
conversation holds onto its last image indefinitely - `get_last_attachment`
deliberately ignores the history window, since a picture doesn't stop
being the subject because ten messages went by.
"""
from collections.abc import Iterator
from dataclasses import dataclass, field

from fastapi import HTTPException, status

from agents.agent_loop import AgentStep, StepEvent
from agents.vision_agent import VisionAgent, VisionResultEvent
from ai.llm_service import get_llm_service
from ai.vision_service import get_vision_service
from app.core.config import settings
from app.schemas.agent_trace import to_step_payload
from app.schemas.rag import RAGChatSource
from app.services.rag_service import build_retriever, to_chat_sources
from memory.attachment_store import AttachmentStore, get_attachment_store
from memory.compaction import ConversationCompactor, get_conversation_compactor
from memory.conversation_memory import ConversationMemory, get_conversation_memory


@dataclass
class VisionAgentChatResult:
    conversation_id: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[RAGChatSource] = field(default_factory=list)
    stopped_because: str = "finished"
    unverified_values: list[str] = field(default_factory=list)


@dataclass
class _ResolvedImage:
    """The image a turn is about, however the request supplied it."""

    data: bytes
    mime_type: str
    ref: str


class VisionAgentChatService:
    """Wraps VisionAgent with the conversation memory layer."""

    def __init__(
        self,
        agent: VisionAgent,
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

    def _compact(self, conversation_id: str) -> None:
        """Summarise anything that has just fallen out of the window."""
        if self._compactor:
            self._compactor.compact(conversation_id)

    def _resolve_image(
        self, conversation_id: str, image_bytes: bytes | None, mime_type: str
    ) -> _ResolvedImage:
        """
        Work out which image this turn is about, and make sure it is stored.

        An uploaded image is stored and used. No upload means the caller is
        following up on a picture already in this conversation, so the last
        one is loaded back out of the attachment store.

        Raises:
            ValueError: if there is no image here and none to inherit -
                including the case where the conversation *had* one but its
                file is gone (a database moved without its attachment
                directory). Re-uploading is the fix for both, and it is
                what every request had to do before this existed.
        """
        if image_bytes:
            return _ResolvedImage(
                data=image_bytes,
                mime_type=mime_type,
                ref=self._attachments.store(image_bytes, mime_type=mime_type),
            )

        previous_ref = self._memory.get_last_attachment(conversation_id, modality="image")
        attachment = self._attachments.load(previous_ref) if previous_ref else None
        if attachment is None:
            raise ValueError(
                "No image was provided and this conversation has none to fall "
                "back on. Attach an image to ask about it."
            )

        return _ResolvedImage(
            data=attachment.data, mime_type=attachment.mime_type, ref=attachment.ref
        )

    def _history_for(self, conversation_id: str, current_ref: str) -> list[dict[str, str]]:
        """
        Load the history window, labelling each turn with which image it was
        about.

        The label is the whole point. Without it the model reads every past
        statement as applying to the picture in front of it - see the module
        docstring for what that does to a conversation with two images in
        it. Text turns are passed through untouched, so a conversation that
        never carried an image looks exactly as it did before.

        The compaction summary goes in front, unlabelled: it covers turns
        that are no longer individually present, so there is no single image
        it could honestly be attributed to.
        """
        history: list[dict[str, str]] = (
            self._compactor.summary_prefix(conversation_id) if self._compactor else []
        )
        for message in self._memory.get_history(conversation_id, limit=self._history_limit):
            content = message.content
            if message.modality == "image" and message.attachment_ref:
                label = (
                    "about the image in this message"
                    if message.attachment_ref == current_ref
                    else "about a DIFFERENT image, not the one attached now"
                )
                content = f"[{label}] {content}"
            history.append({"role": message.role, "content": content})
        return history

    def analyze(
        self,
        question: str,
        image_bytes: bytes | None,
        mime_type: str,
        conversation_id: str | None = None,
    ) -> VisionAgentChatResult:
        """
        Answer `question` about `image_bytes`, recording the turn in the
        conversation identified by `conversation_id` (a new one is generated
        if absent).

        `image_bytes` may be None, which means "the image this conversation
        is already about" - see `_resolve_image`.

        Raises:
            ValueError: if the question is empty, the mime type is
                unsupported, or there is no image here and none to inherit.
            RuntimeError: if an LLM or vision call fails.
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()
        image = self._resolve_image(resolved_id, image_bytes, mime_type)

        history_messages = self._history_for(resolved_id, current_ref=image.ref)

        self._memory.add_message(
            resolved_id,
            role="user",
            content=question,
            modality="image",
            attachment_ref=image.ref,
        )

        result = self._agent.run(
            question,
            image_bytes=image.data,
            mime_type=image.mime_type,
            history=history_messages,
        )

        # Tagged identically to the question: the answer is the half a later
        # turn reads back as context, so it is the half that most needs to
        # say which picture it was about.
        self._memory.add_message(
            resolved_id,
            role="assistant",
            content=result.answer,
            modality="image",
            attachment_ref=image.ref,
        )
        self._compact(resolved_id)

        return VisionAgentChatResult(
            conversation_id=resolved_id,
            answer=result.answer,
            steps=result.steps,
            sources=to_chat_sources(result.sources),
            stopped_because=result.stopped_because,
            unverified_values=result.unverified_values,
        )


    def stream_analyze(
        self,
        question: str,
        image_bytes: bytes | None,
        mime_type: str,
        conversation_id: str | None = None,
    ) -> tuple[str, Iterator[dict]]:
        """
        The streaming counterpart to `analyze`, mirroring
        AgentChatService.stream_answer's shape.

        Returns `(conversation_id, events)`. The events are dicts because
        several different things reach the client in order:

            {"type": "step", "index": 1, "step": {...}}
            {"type": "sources", "sources": [...]}
            {"type": "answer", "content": "..."}
            {"type": "unverified", "values": ["£42.25"]}
            {"type": "done", "stopped_because": "finished"}

        Steps rather than tokens, because a step is the unit worth
        watching: "reading the text", then "checking the expense policy" is
        a progress report a person can follow, where prose appearing one
        word at a time after twenty seconds of silence is not.

        `sources` and `unverified` are emitted only when non-empty, and
        after the answer they describe rather than before it - unlike the
        chat agent's citations, which are known before generation starts.
        Here the numeric check reads the finished answer, so it cannot
        exist any earlier.

        The assistant turn is persisted when the stream ends, including
        when it ends early: `finally` runs on GeneratorExit, so a browser
        tab closed mid-run still records whatever was produced.
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()
        # Also eager, and for the same reason as the stream below: a
        # follow-up whose image has gone missing has to be a 400 while a 400
        # is still possible.
        image = self._resolve_image(resolved_id, image_bytes, mime_type)

        history_messages = self._history_for(resolved_id, current_ref=image.ref)

        self._memory.add_message(
            resolved_id,
            role="user",
            content=question,
            modality="image",
            attachment_ref=image.ref,
        )

        # Eager, so an unsupported mime type is still a 400 rather than an
        # error frame inside a 200 that has already been committed.
        stream = self._agent.stream(
            question, image_bytes=image.data, mime_type=image.mime_type,
            history=history_messages,
        )

        def events() -> Iterator[dict]:
            answer = ""
            try:
                for index, event in enumerate(stream, start=1):
                    if isinstance(event, StepEvent):
                        yield {
                            "type": "step",
                            "index": index,
                            "step": to_step_payload(event.step),
                        }
                        continue

                    result = event.result
                    answer = result.answer
                    if result.sources:
                        yield {
                            "type": "sources",
                            "sources": [
                                source.model_dump(by_alias=True)
                                for source in to_chat_sources(result.sources)
                            ],
                        }
                    yield {"type": "answer", "content": result.answer}
                    if result.unverified_values:
                        yield {
                            "type": "unverified",
                            "values": result.unverified_values,
                        }
                    yield {
                        "type": "done",
                        "stopped_because": result.stopped_because,
                    }
            finally:
                if answer.strip():
                    self._memory.add_message(
                        resolved_id,
                        role="assistant",
                        content=answer,
                        modality="image",
                        attachment_ref=image.ref,
                    )
                    # After the last frame, so the summarising call is not
                    # something the reader waits on.
                    self._compact(resolved_id)

        return resolved_id, events()


def get_vision_agent() -> VisionAgent:
    """FastAPI dependency that builds a VisionAgent from shared singletons."""
    try:
        return VisionAgent(
            llm_service=get_llm_service(),
            vision_service=get_vision_service(),
            retriever=build_retriever(),
            max_steps=settings.vision_agent_max_steps,
            search_top_k=settings.vision_agent_search_top_k,
            temperature=settings.agent_temperature,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vision agent is not configured: {exc}",
        ) from exc


def get_vision_agent_chat_service() -> VisionAgentChatService:
    """FastAPI dependency that builds a memory-aware VisionAgentChatService."""
    return VisionAgentChatService(
        agent=get_vision_agent(),
        memory=get_conversation_memory(),
        attachments=get_attachment_store(),
        history_limit=settings.conversation_history_limit,
        compactor=get_conversation_compactor(),
    )
