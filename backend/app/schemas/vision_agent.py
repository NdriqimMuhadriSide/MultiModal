"""
Response model for the vision agent endpoint.

There is no request model: the request is multipart/form-data (an image
file plus form fields), which FastAPI takes as File/Form parameters rather
than a parsed body - the same shape POST /vision/analyze uses.
"""
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent_trace import AgentStepModel
from app.schemas.rag import RAGChatSource


class VisionAgentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str
    answer: str
    conversation_id: str = Field(..., alias="conversationId")
    steps: list[AgentStepModel] = Field(default_factory=list)
    # Knowledge-base passages only, never the image. An image citation would
    # need a chunk id, a filename and a page, and the upload has none of
    # them - so the list stays empty on a question answered purely from the
    # picture, which is the honest result rather than a gap.
    sources: list[RAGChatSource] = Field(default_factory=list)
    # "finished" | "step_limit" | "parse_failures". Surfaced because the
    # three are not equally trustworthy: an answer written after the budget
    # ran out was synthesised from partial work.
    stopped_because: str = Field(..., alias="stoppedBecause")
    # Figures in the answer that character recognition did not confirm.
    # Empty is the good case, and a non-empty list is a caution rather than
    # a verdict: a correctly derived per-head amount and a limit quoted from
    # a retrieved policy both appear here legitimately. It exists so a
    # reader can tell which numbers came off the image, which is the one
    # distinction the answer text alone cannot be trusted to make.
    unverified_values: list[str] = Field(
        default_factory=list, alias="unverifiedValues"
    )
