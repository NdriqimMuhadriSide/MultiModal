"""
Audio metadata extraction.

Responsibility: pull descriptive facts out of an audio file - filename,
duration, sample rate, channel count, and file size - without decoding or
transcribing the audio itself. Uses `mutagen`, a pure-Python audio tag/
metadata library (no ffmpeg or native compilation required), so this stays
a lightweight, fast step that runs before the expensive transcription call.

Why metadata is useful:
    - Duration lets the UI show "3:42" next to a file and lets the backend
      reject absurdly long recordings before spending time/cost on
      transcription.
    - Sample rate / channel count help diagnose audio quality issues (e.g.
      an 8kHz mono recording will transcribe far less accurately than a
      44.1kHz stereo one) and are useful diagnostics when a user reports a
      bad transcript.
    - File size is already known post-upload, but bundling it with the
      rest of the metadata gives the frontend one place to read all of it
      instead of tracking it separately.
    - All four fields together form the audit trail attached to every
      transcript/analysis - useful today for the response payload, and
      exactly the kind of provenance metadata Phase 4A attaches to document
      chunks (filename, page) - so audio can plug into the same "chunk +
      metadata" pattern once it's chunked for RAG.

Metadata extraction failures (a corrupt file, or a format mutagen can't
parse) are treated as non-fatal: the pipeline can still transcribe some
files mutagen can't introspect, so partial metadata degrades gracefully
rather than blocking the whole request.
"""
import io
from dataclasses import dataclass

from mutagen import File as MutagenFile


@dataclass
class AudioMetadata:
    """Descriptive facts about an audio file, independent of its content."""

    filename: str
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    size_bytes: int


def extract_metadata(filename: str, file_bytes: bytes) -> AudioMetadata:
    """
    Extract metadata from raw audio bytes.

    Never raises for a file that fails metadata parsing - duration/
    sample_rate/channels simply come back as None in that case, since a
    malformed metadata header doesn't necessarily mean the audio itself is
    unusable for transcription.
    """
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None

    try:
        parsed = MutagenFile(io.BytesIO(file_bytes))
        if parsed is not None and parsed.info is not None:
            duration_seconds = getattr(parsed.info, "length", None)
            sample_rate = getattr(parsed.info, "sample_rate", None)
            channels = getattr(parsed.info, "channels", None)
    except Exception:  # noqa: BLE001 - metadata parsing is best-effort, never fatal
        pass

    return AudioMetadata(
        filename=filename,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        size_bytes=len(file_bytes),
    )
