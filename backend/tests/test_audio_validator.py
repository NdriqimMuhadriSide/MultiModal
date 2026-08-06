import pytest

from processors.audio.audio_validator import AudioValidationError, AudioValidator


def test_validate_accepts_supported_mime_type():
    validator = AudioValidator(max_size_mb=25)
    validator.validate(filename="meeting.mp3", mime_type="audio/mpeg", file_bytes=b"fake audio bytes")


def test_validate_accepts_supported_extension_when_mime_type_is_generic():
    # Some browsers send a generic/empty MIME type for audio uploads -
    # the extension should still let a genuinely-supported file through.
    validator = AudioValidator(max_size_mb=25)
    validator.validate(
        filename="voice_note.m4a", mime_type="application/octet-stream", file_bytes=b"fake"
    )


@pytest.mark.parametrize(
    "filename,mime_type",
    [
        ("song.flac", "audio/flac"),
        ("clip.ogg", "audio/ogg"),
        ("notes.txt", "text/plain"),
    ],
)
def test_validate_rejects_unsupported_type(filename, mime_type):
    validator = AudioValidator(max_size_mb=25)
    with pytest.raises(AudioValidationError):
        validator.validate(filename=filename, mime_type=mime_type, file_bytes=b"fake")


def test_validate_rejects_empty_file():
    validator = AudioValidator(max_size_mb=25)
    with pytest.raises(AudioValidationError):
        validator.validate(filename="meeting.mp3", mime_type="audio/mpeg", file_bytes=b"")


def test_validate_rejects_file_over_size_limit():
    validator = AudioValidator(max_size_mb=1)
    oversized = b"a" * (2 * 1024 * 1024)
    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate(filename="meeting.mp3", mime_type="audio/mpeg", file_bytes=oversized)

    assert "1MB" in str(exc_info.value)


def test_validate_error_message_names_the_file():
    validator = AudioValidator(max_size_mb=25)
    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate(filename="clip.ogg", mime_type="audio/ogg", file_bytes=b"fake")

    assert "clip.ogg" in str(exc_info.value)
