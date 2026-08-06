"""
Pydantic request/response models for the /chat endpoint.
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's input text.")
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation to continue. Omit to start a new conversation.",
    )


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str


class ConversationMessageResponse(BaseModel):
    role: str
    content: str
    created_at: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessageResponse]
