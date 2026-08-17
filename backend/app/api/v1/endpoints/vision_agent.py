"""
Vision agent endpoint.

Pure HTTP layer: parses the multipart request (image file + question),
enforces the upload size limit, runs the agent loop, maps domain errors to
status codes, and returns the answer with the trace that produced it.

Distinct from POST /vision/analyze, which sends the image to the vision
model once and returns what it says. This one decides *how* to read the
image - vision model, character recognition, or both - and can check what
it finds against the ingested documents. The questions it exists for are
the ones where a single vision call gives a confidently wrong answer:
anything turning on an exact number, and anything that has to be compared
against a policy.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.v1.sse import SSE_HEADERS, sse_event
from app.core.config import settings
from app.schemas.agent_trace import to_step_models
from app.schemas.vision_agent import VisionAgentResponse
from app.services.vision_agent_service import (
    VisionAgentChatService,
    get_vision_agent_chat_service,
)

router = APIRouter(tags=["vision-agent"])


def _is_absent(image: UploadFile | str | None) -> bool:
    """
    True when the request carried no usable file in the image field.

    Written as "is it None or a string" rather than the obvious "is it an
    UploadFile", because that obvious version is quietly wrong here:
    `fastapi.UploadFile` is a *subclass* of `starlette.datastructures.
    UploadFile`, and what the multipart parser hands back is the base
    class. `isinstance(image, fastapi.UploadFile)` is therefore False for
    every real upload - and the failure is silent, since the request still
    succeeds and merely behaves as though no image was attached.

    Testing for the two things that mean "absent" avoids the question
    entirely, and keeps working whichever class a future FastAPI hands us.
    """
    return image is None or isinstance(image, str)


async def _read_upload(image: UploadFile | str | None) -> bytes | None:
    """
    Read an optional image upload, enforcing the size limit.

    Returns None when there is nothing to read, which is the service's
    signal to fall back to the image the conversation already carries.

    Three different things mean "no image", and they all have to end up
    here rather than as three different errors:

        - the field is absent entirely
        - the part is present but empty
        - the part has an empty filename, which Starlette parses as a
          *string* form field rather than an upload. This is what a browser
          sends for a file input nobody picked a file with, so the
          alternative is a 422 complaining about a type the caller never
          knowingly chose - which is why the annotation admits `str` at
          all.

    Raises:
        HTTPException: 413, if the upload is over the limit.
    """
    if _is_absent(image):
        return None

    image_bytes = await image.read()
    if not image_bytes:
        return None

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {settings.max_upload_size_mb}MB upload limit.",
        )

    return image_bytes


def _mime_type_of(image: UploadFile | str | None) -> str:
    """
    The declared type of an upload, or "" when there isn't one.

    Paired with `_read_upload` so the two answers cannot disagree: an empty
    string here is only ever read alongside a None there, and the service
    ignores it entirely in that case.
    """
    return "" if _is_absent(image) else (image.content_type or "")


@router.post("/vision/ask", response_model=VisionAgentResponse)
async def ask_about_image(
    image: UploadFile | str | None = File(
        default=None,
        description=(
            "Image to analyse. Optional when `conversation_id` names a "
            "conversation that already carries one - omit it to ask a "
            "follow-up about the same picture."
        ),
    ),
    question: str = Form(..., description="Question about the image."),
    conversation_id: str | None = Form(
        default=None,
        description=(
            "Conversation to continue. Omit on the first message of a new "
            "conversation - the backend generates one and returns it."
        ),
    ),
    service: VisionAgentChatService = Depends(get_vision_agent_chat_service),
) -> VisionAgentResponse:
    image_bytes = await _read_upload(image)

    try:
        result = service.analyze(
            question=question,
            image_bytes=image_bytes,
            mime_type=_mime_type_of(image),
            conversation_id=conversation_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return VisionAgentResponse(
        question=question,
        answer=result.answer,
        conversation_id=result.conversation_id,
        steps=to_step_models(result.steps),
        sources=result.sources,
        stopped_because=result.stopped_because,
        unverified_values=result.unverified_values,
    )


@router.post("/vision/ask/stream")
async def ask_about_image_stream(
    image: UploadFile | str | None = File(
        default=None,
        description=(
            "Image to analyse. Optional when `conversation_id` names a "
            "conversation that already carries one."
        ),
    ),
    question: str = Form(..., description="Question about the image."),
    conversation_id: str | None = Form(default=None),
    service: VisionAgentChatService = Depends(get_vision_agent_chat_service),
) -> StreamingResponse:
    """
    Same contract as POST /vision/ask, delivered as Server-Sent Events.

    Event shapes, all sent as `data: {...}`:

        {"type": "start", "conversation_id": "..."}   exactly one, first
        {"type": "step", "index": 1, "step": {...}}   one per completed turn
        {"type": "sources", "sources": [...]}         at most one, if any
        {"type": "answer", "content": "..."}          exactly one
        {"type": "unverified", "values": [...]}       at most one, if any
        {"type": "done", "stopped_because": "..."}    terminates a good stream
        {"type": "error", "detail": "..."}            terminates a bad one

    Steps, not tokens. A run takes several seconds per step and produces no
    prose at all until the end, so token streaming would deliver nothing
    for twenty seconds and then everything. A step frame arrives the moment
    each tool returns, which is both sooner and more informative: "reading
    the text" then "checking the expense policy" tells the reader what the
    agent is doing, and the trace it builds is the same one the
    non-streaming endpoint returns in `steps`.

    `unverified` arrives after `answer` rather than before it, unlike the
    chat agent's citations. It cannot come earlier - the check reads the
    finished answer.

    The non-streaming POST /vision/ask stays: the service worker's offline
    outbox replays queued requests with no page attached to read a stream.

    Errors are frames rather than status codes once the stream has opened,
    for the usual reason - the response has already committed to 200 by the
    time a provider can fail. Input problems are still real HTTP errors,
    because validation happens before the first frame.
    """
    image_bytes = await _read_upload(image)

    try:
        conversation_id_out, events = service.stream_analyze(
            question=question,
            image_bytes=image_bytes,
            mime_type=_mime_type_of(image),
            conversation_id=conversation_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    def frames():
        yield sse_event({"type": "start", "conversation_id": conversation_id_out})
        try:
            for event in events:
                yield sse_event(event)
        except (RuntimeError, ValueError) as exc:
            yield sse_event({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
