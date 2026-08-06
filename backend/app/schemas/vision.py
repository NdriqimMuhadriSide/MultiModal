"""
Pydantic response model for the /vision/analyze endpoint.

The request itself is multipart/form-data (image file + question field),
so there's no request schema here - FastAPI parses those directly as
endpoint parameters (UploadFile + Form).
"""
from pydantic import BaseModel


class VisionAnalyzeResponse(BaseModel):
    answer: str
