"""
Pydantic response model for the /audio/analyze endpoint.

The request itself is multipart/form-data (an `audio` file part + an
optional `question` form field), so there's no request schema here -
FastAPI parses those directly as endpoint parameters (UploadFile + Form),
matching the same pattern app/schemas/vision.py uses for /vision/analyze.
"""
from pydantic import BaseModel


class AudioMetadataResponse(BaseModel):
    filename: str
    duration: float | None = None
    size: int
    sample_rate: int | None = None
    channels: int | None = None


class AudioAnalyzeResponse(BaseModel):
    transcript: str
    analysis: str
    metadata: AudioMetadataResponse
