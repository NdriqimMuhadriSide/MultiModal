"""
Chat endpoint.

Pure HTTP layer: parses the request, calls the chat service (which reads
from and writes to the conversation memory layer internally), maps domain
errors to HTTP status codes, and returns the response model. No OpenAI-
specific code or memory/storage logic lives here.
"""
from fastapi import APIRouter, Depends, HTTPException, status

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
