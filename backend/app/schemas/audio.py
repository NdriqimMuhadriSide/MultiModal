"""
Pydantic response model for the /audio/analyze endpoint.

The request itself is multipart/form-data (an `audio` file part + an
optional `question` form field), so there's no request schema here -
FastAPI parses those directly as endpoint parameters (UploadFile + Form),
matching the same pattern app/schemas/vision.py uses for /vision/analyze.
"""
from pydantic import BaseModel, ConfigDict, Field


class AudioMetadataResponse(BaseModel):
    filename: str
    duration: float | None = None
    size: int
    sample_rate: int | None = None
    channels: int | None = None


class AudioAnalyzeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    transcript: str
    analysis: str
    metadata: AudioMetadataResponse
    # Where this analysis was filed. Returned even when the caller didn't
    # ask for a conversation - the turn is recorded either way, and an id
    # the client never sees would be a record nobody can reach.
    conversation_id: str = Field(..., alias="conversationId")
