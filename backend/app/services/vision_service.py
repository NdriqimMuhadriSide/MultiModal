"""
Vision business logic.

Sits between the API route and the AI vision service, mirroring
app/services/chat_service.py. One vision call, one answer - the deciding
and re-checking that app/services/vision_agent_service.py does is
deliberately not here.

WHY THIS WRITES TO MEMORY

It used to be a pure function: bytes in, string out, nothing kept. That
made it the one modality whose output was unrecoverable the moment the
response was sent - the image was gone, the answer was gone, and a user
who wanted it again had to re-upload and re-ask.

So the turn is now recorded like any other, with the image stored and
referenced. What that buys is not a smarter /vision/analyze; it is that
the analysis exists afterwards - in GET /chat/{id}/history, as context for
a later /chat or /agent turn in the same conversation, and as an image a
follow-up to POST /vision/ask can inherit rather than demand again.

It writes but does not read. This endpoint asks the vision model one
question about one picture, and prepending conversation history would
change what it is - the agent endpoint is the one that reasons over
context. A conversation_id here means "file this under that conversation",
not "answer in light of it", and the docstring on `analyze` says so.
"""
from dataclasses import dataclass

from fastapi import HTTPException, status

from ai.vision_service import VisionService, get_vision_service
from memory.attachment_store import AttachmentStore, get_attachment_store
from memory.conversation_memory import ConversationMemory, get_conversation_memory


@dataclass
class VisionAnalysisResult:
    answer: str
    conversation_id: str


class VisionAnalysisService:
    def __init__(
        self,
        vision_service: VisionService,
        memory: ConversationMemory,
        attachments: AttachmentStore,
    ) -> None:
        self._vision_service = vision_service
        self._memory = memory
        self._attachments = attachments

    def analyze(
        self,
        image_bytes: bytes,
        mime_type: str,
        question: str,
        conversation_id: str | None = None,
    ) -> VisionAnalysisResult:
        """
        Answer `question` about `image_bytes` and record the exchange.

        `conversation_id` files the turn under an existing conversation; a
        new one is generated when it is absent, and returned either way so
        the caller can keep using it. It does not affect the answer - see
        the module docstring.

        Raises:
            ValueError: if the mime type is unsupported (propagated from
                the vision service).
            RuntimeError: if the vision call fails.
        """
        resolved_id = conversation_id or self._memory.new_conversation_id()

        # Stored before the model call, so a provider failure still leaves
        # the image behind - the user can ask again without re-uploading,
        # which is the more useful half of the record when a call fails.
        # Unsupported types raise here rather than there, and both are the
        # 400 the endpoint already maps.
        ref = self._attachments.store(image_bytes, mime_type=mime_type)

        answer = self._vision_service.analyze_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            question=question,
        )

        self._memory.add_message(
            resolved_id,
            role="user",
            content=question,
            modality="image",
            attachment_ref=ref,
        )
        self._memory.add_message(
            resolved_id,
            role="assistant",
            content=answer,
            modality="image",
            attachment_ref=ref,
        )

        return VisionAnalysisResult(answer=answer, conversation_id=resolved_id)


def get_vision_analysis_service() -> VisionAnalysisService:
    """
    FastAPI dependency that builds a VisionAnalysisService.

    Configuration errors (e.g. missing GROQ_API_KEY) happen here, during
    dependency resolution - outside the route handler's try/except - so
    they're translated into a clean HTTPException instead of leaking a raw
    stack trace to the client as an unhandled 500.
    """
    try:
        return VisionAnalysisService(
            vision_service=get_vision_service(),
            memory=get_conversation_memory(),
            attachments=get_attachment_store(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vision service is not configured: {exc}",
        ) from exc
