"""
Pydantic response model for the /vision/analyze endpoint.

The request itself is multipart/form-data (image file + question field),
so there's no request schema here - FastAPI parses those directly as
endpoint parameters (UploadFile + Form).
"""
from pydantic import BaseModel, ConfigDict, Field


class VisionAnalyzeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    answer: str
    # Where this analysis was filed. Returned even when the caller didn't
    # ask for a conversation - the turn is recorded either way, and an id
    # the client never sees would be a record nobody can reach.
    conversation_id: str = Field(..., alias="conversationId")
