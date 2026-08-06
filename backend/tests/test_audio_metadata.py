import wave
import io

from processors.audio.audio_metadata import extract_metadata


def _tiny_wav_bytes(duration_seconds: float = 0.5, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Build a minimal, real WAV file so mutagen can parse real metadata."""
    buffer = io.BytesIO()
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_frames * channels)
    return buffer.getvalue()


def test_extract_metadata_returns_filename_and_size():
    wav_bytes = _tiny_wav_bytes()

    metadata = extract_metadata(filename="test.wav", file_bytes=wav_bytes)

    assert metadata.filename == "test.wav"
    assert metadata.size_bytes == len(wav_bytes)


def test_extract_metadata_parses_duration_sample_rate_and_channels_from_real_wav():
    wav_bytes = _tiny_wav_bytes(duration_seconds=1.0, sample_rate=16000, channels=1)

    metadata = extract_metadata(filename="test.wav", file_bytes=wav_bytes)

    assert metadata.duration_seconds is not None
    assert abs(metadata.duration_seconds - 1.0) < 0.1
    assert metadata.sample_rate == 16000
    assert metadata.channels == 1


def test_extract_metadata_degrades_gracefully_for_unparseable_bytes():
    metadata = extract_metadata(filename="broken.mp3", file_bytes=b"not actually audio")

    assert metadata.filename == "broken.mp3"
    assert metadata.size_bytes == len(b"not actually audio")
    assert metadata.duration_seconds is None
    assert metadata.sample_rate is None
    assert metadata.channels is None


def test_extract_metadata_never_raises_for_empty_bytes():
    metadata = extract_metadata(filename="empty.mp3", file_bytes=b"")

    assert metadata.size_bytes == 0
    assert metadata.duration_seconds is None
