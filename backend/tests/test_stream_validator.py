import pytest

from processors.streaming.stream_validator import StreamValidationError, StreamValidator


def test_validate_accepts_supported_frame():
    validator = StreamValidator(max_frame_size_mb=5)
    validator.validate(session_id="abc-123", mime_type="image/jpeg", frame_bytes=b"fake frame bytes")


@pytest.mark.parametrize("mime_type", ["image/png", "image/webp", "image/jpg"])
def test_validate_accepts_every_supported_mime_type(mime_type):
    validator = StreamValidator(max_frame_size_mb=5)
    validator.validate(session_id="abc-123", mime_type=mime_type, frame_bytes=b"fake")


def test_validate_rejects_missing_session_id():
    validator = StreamValidator(max_frame_size_mb=5)
    with pytest.raises(StreamValidationError):
        validator.validate(session_id="", mime_type="image/jpeg", frame_bytes=b"fake")


def test_validate_rejects_unsupported_mime_type():
    validator = StreamValidator(max_frame_size_mb=5)
    with pytest.raises(StreamValidationError):
        validator.validate(session_id="abc-123", mime_type="video/mp4", frame_bytes=b"fake")


def test_validate_rejects_empty_frame():
    validator = StreamValidator(max_frame_size_mb=5)
    with pytest.raises(StreamValidationError):
        validator.validate(session_id="abc-123", mime_type="image/jpeg", frame_bytes=b"")


def test_validate_rejects_frame_over_size_limit():
    validator = StreamValidator(max_frame_size_mb=1)
    oversized = b"a" * (2 * 1024 * 1024)
    with pytest.raises(StreamValidationError) as exc_info:
        validator.validate(session_id="abc-123", mime_type="image/jpeg", frame_bytes=oversized)

    assert "1MB" in str(exc_info.value)
