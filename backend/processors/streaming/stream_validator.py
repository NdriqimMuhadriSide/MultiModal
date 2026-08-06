"""
Stream frame validator.

Responsibility: decide whether a single sampled frame (and the session
identifier it's attached to) is acceptable to process, before any
expensive work (vision analysis) happens. Mirrors
processors/audio/audio_validator.py's shape exactly: a pure,
dependency-free module - no AI client, no FastAPI - so it's unit testable
with plain bytes and reusable outside the /stream/frame endpoint if a
future ingestion path (e.g. a WebSocket stream) needs the same checks.

Validates three things:
    1. sessionId    - must be present and non-empty; it's the key every
                       downstream layer (frame_sampler, the context ring
                       buffer in stream_processor) uses to keep one
                       browser tab's stream independent from another's.
    2. MIME type    - what the browser reports for the captured frame
                       (a canvas.toBlob() output - typically image/jpeg or
                       image/png).
    3. File size     - a sampled frame is a screenshot-sized image; a
                       generously-sized limit still catches a misbehaving
                       or malicious client sending something far larger.

Raises StreamValidationError (not a generic ValueError) so the API layer
can distinguish "this validator rejected the input" from any other
ValueError a lower layer might raise, and always return a clear, specific
message - never a generic "invalid input."
"""
# Frame MIME types accepted from the browser's canvas.toBlob() output.
# Deliberately a small set - narrower than processors' image/vision
# validators elsewhere, since frames are always produced by a canvas
# capture, never picked by a user, so there's no need to support every
# image format a user might upload.
SUPPORTED_FRAME_MIME_TYPES: set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


class StreamValidationError(Exception):
    """Raised when a sampled frame or its session fails validation. The message is safe to show to the end user."""


class StreamValidator:
    """Validates an incoming frame's session id, MIME type, and size."""

    def __init__(self, max_frame_size_mb: int) -> None:
        self._max_frame_size_bytes = max_frame_size_mb * 1024 * 1024
        self._max_frame_size_mb = max_frame_size_mb

    def validate(self, session_id: str, mime_type: str, frame_bytes: bytes) -> None:
        """
        Validate a single sampled frame.

        Raises:
            StreamValidationError: with a clear, user-facing message if the
                session id is missing, the MIME type is unsupported, the
                frame is empty, or the frame exceeds the size limit.
        """
        if not session_id or not session_id.strip():
            raise StreamValidationError("sessionId must not be empty.")

        if mime_type not in SUPPORTED_FRAME_MIME_TYPES:
            raise StreamValidationError(
                f"Unsupported frame type '{mime_type or 'unknown'}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_FRAME_MIME_TYPES))}."
            )

        if not frame_bytes:
            raise StreamValidationError("frame must not be empty.")

        if len(frame_bytes) > self._max_frame_size_bytes:
            actual_mb = len(frame_bytes) / (1024 * 1024)
            raise StreamValidationError(
                f"Frame is {actual_mb:.1f}MB, which exceeds the "
                f"{self._max_frame_size_mb}MB limit for a single sampled frame."
            )
