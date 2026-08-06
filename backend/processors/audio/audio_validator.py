"""
Audio validator.

Responsibility: decide whether an uploaded file is acceptable to process,
before any expensive work (metadata extraction, transcription) happens.
This is intentionally a pure, dependency-free module - no AI client, no
FastAPI, no filesystem access - so it can be unit tested with plain bytes
and reused unchanged if audio validation is ever needed outside the
/audio/analyze endpoint (e.g. a future bulk-import job).

Validates three things, in order, matching the Phase 5 spec:
    1. MIME type   - what the browser/client claims the file is
    2. Extension    - a fallback/cross-check, since browsers sometimes send
                       an empty or generic MIME type (e.g. some Safari/
                       mobile uploads report "" or "application/octet-stream"
                       for audio)
    3. File size    - checked last since it's the cheapest to explain and
                       the most likely to be hit by a real user (a large
                       recording), so the two content checks take priority
                       for a wrong-file-type error message.

Raises AudioValidationError (not a generic ValueError) so the API layer can
distinguish "this specific validator rejected the file" from any other
ValueError a lower layer might raise, and always return a clear, specific
message to the client - never a generic "invalid input."
"""
# Extensions accepted by this pipeline, per the Phase 5 spec. Keys are the
# canonical MIME type FastAPI/browsers report for each format; a given
# extension may map from more than one commonly-seen MIME type (e.g. some
# browsers send "audio/mp3" while others send the more correct "audio/mpeg").
SUPPORTED_AUDIO_MIME_TYPES: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
}

SUPPORTED_AUDIO_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a", ".webm"}


class AudioValidationError(Exception):
    """Raised when an uploaded file fails audio validation. The message is safe to show to the end user."""





class AudioValidator:
    """Validates an uploaded audio file's MIME type, extension, and size."""

    def __init__(self, max_size_mb: int) -> None:
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_size_mb = max_size_mb

    def validate(self, filename: str, mime_type: str, file_bytes: bytes) -> None:
        """
        Validate an uploaded audio file.

        Raises:
            AudioValidationError: with a clear, user-facing message if the
                MIME type, extension, or size is not acceptable.
        """
        extension = _extract_extension(filename)

        mime_ok = mime_type in SUPPORTED_AUDIO_MIME_TYPES
        extension_ok = extension in SUPPORTED_AUDIO_EXTENSIONS

        if not mime_ok and not extension_ok:
            raise AudioValidationError(
                f"Unsupported audio file '{filename}' (type '{mime_type or 'unknown'}'). "
                f"Supported formats: mp3, wav, m4a, webm."
            )

        if not file_bytes:
            raise AudioValidationError(f"'{filename}' is empty.")

        if len(file_bytes) > self._max_size_bytes:
            actual_mb = len(file_bytes) / (1024 * 1024)
            raise AudioValidationError(
                f"'{filename}' is {actual_mb:.1f}MB, which exceeds the "
                f"{self._max_size_mb}MB limit for audio uploads."
            )


def _extract_extension(filename: str) -> str:
    """Return the lowercased extension (including the dot), or '' if none."""
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()
