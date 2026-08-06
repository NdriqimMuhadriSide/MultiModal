"""
Chat endpoint.

Pure HTTP layer: parses the request, calls the chat service (which reads
from and writes to the conversation memory layer internally), maps domain
errors to HTTP status codes, and returns the response model. No OpenAI-
specific code or memory/storage logic lives here.
"""
import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    ConversationMessageResponse,
)
from app.services.chat_service import ChatService, get_chat_service
from memory.conversation_memory import ConversationMemory, get_conversation_memory

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        result = chat_service.get_answer(
            message=request.message, conversation_id=request.conversation_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return ChatResponse(conversation_id=result.conversation_id, answer=result.answer)


def _sse(event: dict) -> str:
    """
    Frame one payload as a Server-Sent Event.

    The blank line is the message terminator - without it the browser buffers
    the frame forever waiting for the rest. `ensure_ascii=True` (the default)
    also matters more than it looks: SSE is line-delimited, so a raw newline
    inside a token would split one event into two malformed ones. JSON
    escaping it to \\n is what makes arbitrary model output safe to send.
    """
    return f"data: {json.dumps(event)}\n\n"


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Same contract as POST /chat, delivered as a Server-Sent Event stream.

    Event shapes, all sent as `data: {...}`:

        {"type": "start", "conversation_id": "..."}   exactly one, first
        {"type": "delta", "content": "..."}           zero or more
        {"type": "done"}                              terminates a good stream
        {"type": "error", "detail": "..."}            terminates a bad one

    POST rather than GET, so this cannot be consumed with `EventSource` -
    that API only issues GET requests with no body, and the request carries
    a JSON body. Clients read it with fetch() and a stream reader instead;
    see frontend/services/chat-service.ts.

    Why errors are events rather than status codes: the response status is
    committed the moment the first byte is written, which happens before the
    model has produced anything. A mid-stream provider failure therefore
    cannot be a 502 - the client is already reading a 200. Configuration
    errors are the exception and still surface as real HTTP errors, because
    those are raised by `get_chat_service` during dependency resolution,
    before this function is entered.

    The non-streaming POST /chat is deliberately kept: the service worker's
    offline outbox replays queued messages with no page attached to consume
    a stream, and the agent router needs the whole answer at once.
    """
    try:
        conversation_id, chunks = chat_service.stream_answer(
            message=request.message, conversation_id=request.conversation_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    def events():
        yield _sse({"type": "start", "conversation_id": conversation_id})
        try:
            for delta in chunks:
                yield _sse({"type": "delta", "content": delta})
        except RuntimeError as exc:
            yield _sse({"type": "error", "detail": str(exc)})
            return
        yield _sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            # Without these a proxy or the browser will happily hold the
            # whole stream and hand it over in one piece, which reintroduces
            # exactly the latency this endpoint exists to remove.
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/{conversation_id}/history", response_model=ConversationHistoryResponse)
def get_conversation_history(
    conversation_id: str,
    memory: ConversationMemory = Depends(get_conversation_memory),
) -> ConversationHistoryResponse:
    """
    Return every stored message (user + assistant) for a conversation_id,
    in chronological order - a direct view into what the memory layer has
    persisted, independent of the short-term window used when calling the LLM.
    """
    messages = memory.get_full_history(conversation_id)
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=[
            ConversationMessageResponse(
                role=msg.role, content=msg.content, created_at=msg.created_at
            )
            for msg in messages
        ],
    )
