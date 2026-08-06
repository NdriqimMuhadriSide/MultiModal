"""
Pydantic request/response models for the agent endpoint.
"""
from pydantic import BaseModel, Field


class AgentAskRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Message for the agent to handle.")
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Conversation to continue. Omit on the first message of a new "
            "conversation - the backend generates one and returns it."
        ),
    )


class AgentAskResponse(BaseModel):
    message: str
    answer: str
    tool_used: str
    conversation_id: str
