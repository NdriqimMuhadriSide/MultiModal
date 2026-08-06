"""
Pydantic response model for the /stream/frame endpoint.

The request is multipart/form-data (a `frame` image part + `sessionId`,
`question` (optional), and `timestamp` form fields), so there's no request
schema here - FastAPI parses those directly as endpoint parameters
(UploadFile + Form), matching the pattern in app/schemas/vision.py and
app/schemas/audio.py.
"""
from pydantic import BaseModel


class StreamFrameResponse(BaseModel):
    observations: list[str]
    analysis: str
    # True if this frame was actually sent for vision analysis (i.e. the
    # sampling strategy decided "now" was a sampling point); False if the
    # frame was received and validated but intentionally skipped. The
    # frontend uses this to know whether `observations`/`analysis` reflect
    # a fresh result or should be ignored in favor of the last sampled one.
    sampled: bool
