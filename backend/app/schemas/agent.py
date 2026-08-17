"""
Pydantic request/response models for the agent endpoint.
"""
from pydantic import BaseModel, Field

from app.schemas.agent_trace import AgentStepModel
from app.schemas.rag import RAGChatSource


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
    # Which specialist the answer rests on, in one word, for the badge on the
    # bubble. "multiple_specialists" when more than one contributed - naming
    # either alone would tell the reader the answer rests on less than it
    # does. The detail is in `steps`.
    tool_used: str
    conversation_id: str
    # Reuses /rag/chat's citation model rather than declaring a parallel one:
    # a citation from a specialist's knowledge-base tool *is* a citation from
    # /rag/chat - the same chunk, retrieved by the same pipeline - and two
    # models would let the two drift apart field by field.
    #
    # Empty for the tools with nothing to cite (live weather, an answer from
    # the model's own knowledge). Index i is the passage labelled [E(i+1)] in
    # the answer, and stays so across every specialist a turn delegated to,
    # because they all number from one shared ledger.
    sources: list[RAGChatSource] = Field(default_factory=list)
    # The supervisor's trace, with each delegation's sub-steps nested under
    # the step that caused them.
    #
    # New with the supervisor, and not a debug extra. This endpoint used to
    # make one routing decision and run one tool; it now reaches an answer
    # through work the user never asked for and cannot see, and "trust me, I
    # checked" is not a reasonable thing to ask of a reader.
    steps: list[AgentStepModel] = Field(default_factory=list)
    # "finished" when the supervisor answered, "step_limit" when the tree's
    # shared budget ran out first, "parse_failures" when the model stopped
    # producing usable actions. A reader deserves to know an answer was
    # written under a budget rather than because the work was done.
    stopped_because: str = "finished"
