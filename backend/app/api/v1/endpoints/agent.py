"""
Agent endpoint.

Pure HTTP layer: parses the request, invokes the agent's reasoning loop,
maps domain errors to HTTP status codes, and returns the response model
(including which tool the agent chose, for transparency/debugging).
"""
from fastapi import APIRouter, Depends, HTTPException, status

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
