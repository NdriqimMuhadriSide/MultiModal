"""
Pydantic request/response models for the research agent endpoint.
"""
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent_trace import AgentStepModel
from app.schemas.rag import RAGChatSource

# The trace model moved to app/schemas/agent_trace.py when the vision agent
# became a second producer of the same shape. Aliased rather than
# re-declared so existing imports keep working and the two endpoints cannot
# drift apart.
ResearchStepModel = AgentStepModel


class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The question to research across the ingested documents.",
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Conversation to continue. Omit on the first message of a new "
            "conversation - the backend generates one and returns it."
        ),
    )


class ResearchResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str
    answer: str
    conversation_id: str = Field(..., alias="conversationId")
    steps: list[ResearchStepModel] = Field(default_factory=list)
    # Same citation model as /rag/chat and /agent/ask - a passage retrieved
    # by this agent *is* a passage retrieved by that pipeline, and a parallel
    # model would let the two drift apart field by field.
    #
    # Ordered so that index i is the passage the answer cites as [E(i+1)].
    sources: list[RAGChatSource] = Field(default_factory=list)
    # "finished" | "step_limit" | "parse_failures" - why the loop ended.
    # Surfaced because the three are not equally trustworthy: an answer
    # written after the budget ran out was synthesised from partial research,
    # and a client that shows it identically to a completed one is hiding the
    # difference from the reader.
    stopped_because: str = Field(..., alias="stoppedBecause")
