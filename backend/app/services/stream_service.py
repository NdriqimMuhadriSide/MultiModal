"""
Streaming business logic (FastAPI-facing wrapper).

Thin adapter around processors/streaming/stream_processor.py's
StreamProcessor, mirroring the shape of app/services/audio_service.py and
app/services/vision_service.py: this is the dependency-injection seam
where configuration errors (missing GROQ_API_KEY, an invalid sampling
strategy, etc.) get turned into clean HTTPExceptions instead of raw 500s.
No frame processing, sampling, or vision-prompt logic lives here - that's
all in processors/streaming/ and prompts/streaming_prompts.py; this module
only builds and caches the StreamProcessor singleton the endpoint depends on.

Singleton, not per-request: unlike most other *_service.py DI functions in
this codebase, get_stream_processor() is cached (@lru_cache) because the
whole point of StreamProcessor is to hold state *across* requests (the
per-session context ring buffer and sampling clocks) - a fresh instance
per request would defeat sampling and context entirely, since every frame
would look like session's first.
"""
from functools import lru_cache

from fastapi import HTTPException, status

from ai.vision_service import get_vision_service
from app.core.config import settings
from processors.streaming.frame_sampler import FrameSampler
from processors.streaming.stream_processor import StreamProcessor
from processors.streaming.stream_validator import StreamValidator
from processors.streaming.vision_stream import VisionStreamAnalyzer


@lru_cache
def get_stream_processor() -> StreamProcessor:
    """
    FastAPI dependency that builds (and caches) the process-wide
    StreamProcessor.

    Configuration errors (e.g. missing GROQ_API_KEY, or an invalid
    STREAM_SAMPLING_STRATEGY value) happen here, during dependency
    resolution, so they're translated into a clean HTTPException instead
    of leaking a raw stack trace to the client.
    """
    try:
        validator = StreamValidator(max_frame_size_mb=settings.max_stream_frame_size_mb)
        sampler = FrameSampler(
            strategy=settings.stream_sampling_strategy,
            interval_seconds=settings.stream_sampling_interval_seconds,
            frame_count=settings.stream_sampling_frame_count,
        )
        analyzer = VisionStreamAnalyzer(vision_service=get_vision_service())
        return StreamProcessor(
            validator=validator,
            sampler=sampler,
            analyzer=analyzer,
            context_window_size=settings.stream_context_window_size,
            session_ttl_seconds=settings.stream_session_ttl_seconds,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Streaming service is not configured: {exc}",
        ) from exc
