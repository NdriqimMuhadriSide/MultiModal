"""
Agent endpoint.

Pure HTTP layer: parses the request, invokes the supervisor, maps domain
errors to HTTP status codes, and returns the answer together with the trace
that produced it.

This is the project's general entry point: rather than the caller choosing
between /chat, /rag/chat, /research/ask and /vision/ask, it sends the message
here and the supervisor decides whether it can answer directly or needs a
specialist - and, when a question spans both, uses more than one.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.v1.sse import SSE_HEADERS, sse_event
from app.schemas.agent import AgentAskRequest, AgentAskResponse
from app.schemas.agent_trace import to_step_models
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
        sources=result.sources,
        steps=to_step_models(result.steps),
        stopped_because=result.stopped_because,
    )


@router.post("/agent/ask/stream")
def ask_agent_stream(
    request: AgentAskRequest,
    agent_chat_service: AgentChatService = Depends(get_agent_chat_service),
) -> StreamingResponse:
    """
    Same contract as POST /agent/ask, delivered as a Server-Sent Event stream.

    Event shapes, all sent as `data: {...}`:

        {"type": "start", "conversation_id": "..."}    exactly one, first
        {"type": "step", "index": 1, "depth": 0, ...}  one per completed turn
        {"type": "tool", "tool": "research_documents"} exactly one
        {"type": "sources", "sources": [...]}          at most one, if any
        {"type": "answer", "content": "..."}           exactly one
        {"type": "done", "stopped_because": "..."}     terminates a good stream
        {"type": "error", "detail": "..."}             terminates a bad one

    Steps rather than token deltas, which is a change from what this endpoint
    used to send. The old routing agent picked one tool and forwarded that
    tool's generation token by token; a supervisor's answer is already whole
    inside its `finish` action by the time the loop sees it, so there is no
    token stream to forward. See AgentChatService.stream_answer for why the
    replacement suits the work better.

    `depth` is 0 for the supervisor's own steps and 1 for a specialist's, so
    the client can indent a delegated step rather than presenting a
    specialist's search as something the supervisor did itself.

    `tool` arrives once the run resolves rather than up front: with
    delegation, the honest label for a turn depends on what the whole turn
    ended up using, and a turn that read the image *and* checked a policy
    should not be badged as either alone.

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

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
