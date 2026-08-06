"""
Agent endpoint.

Pure HTTP layer: parses the request, invokes the agent's reasoning loop,
maps domain errors to HTTP status codes, and returns the response model
(including which tool the agent chose, for transparency/debugging).
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.v1.sse import SSE_HEADERS, sse_event
from app.schemas.agent import AgentAskRequest, AgentAskResponse
from app.services.agent_service import AgentChatService, get_agent_chat_service

router = APIRouter(tags=["agent"])


@router.post("/agent/ask", response_model=AgentAskResponse)
def ask_agent(
    request: AgentAskRequest,
    agent_chat_service: AgentChatService = Depends(get_agent_chat_service),
) -> AgentAskResponse:
    try:
        result = agent_chat_service.get_answer(
            message=request.message, conversation_id=request.conversation_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return AgentAskResponse(
        message=request.message,
        answer=result.answer,
        tool_used=result.tool_used,
        conversation_id=result.conversation_id,
    )


@router.post("/agent/ask/stream")
def ask_agent_stream(
    request: AgentAskRequest,
    agent_chat_service: AgentChatService = Depends(get_agent_chat_service),
) -> StreamingResponse:
    """
    Same contract as POST /agent/ask, delivered as a Server-Sent Event stream.

    Event shapes, all sent as `data: {...}`:

        {"type": "start", "conversation_id": "..."}   exactly one, first
        {"type": "tool", "tool": "answer_directly"}   once routing resolves
        {"type": "delta", "content": "..."}           zero or more
        {"type": "done"}                              terminates a good stream
        {"type": "error", "detail": "..."}            terminates a bad one

    The `tool` event is what this has and /chat/stream doesn't: routing costs
    an LLM round trip before any answer text exists, so it arrives as its own
    event rather than holding up `start`. The client can label the bubble
    with the chosen tool while the answer is still being written.

    Errors are events rather than status codes for the same reason as
    /chat/stream: by the time the model can fail, the response has already
    committed as a 200. Configuration failures still surface as real HTTP
    errors, since those are raised during dependency resolution.

    The non-streaming POST /agent/ask stays: the service worker's offline
    outbox replays queued messages with no page attached to read a stream.
    """
    try:
        conversation_id, events = agent_chat_service.stream_answer(
            message=request.message, conversation_id=request.conversation_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    def frames():
        yield sse_event({"type": "start", "conversation_id": conversation_id})
        try:
            for event in events:
                yield sse_event(event)
        except (RuntimeError, ValueError) as exc:
            yield sse_event({"type": "error", "detail": str(exc)})
            return
        yield sse_event({"type": "done"})

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
