"""
Stream processor - orchestrates the Frame Processing layer's pipeline:

    Receive frame
        -> Validate                  (stream_validator.StreamValidator)
        -> Sampling decision          (frame_sampler.FrameSampler - is this
                                        frame due for analysis right now?)
        -> Vision analysis            (vision_stream.VisionStreamAnalyzer -
                                        delegates to the Vision/LLM layer)
        -> Update session context     (in-memory ring buffer, this module)
        -> Return observations

This is the single entry point the service layer
(app/services/stream_service.py) calls - it never talks to the validator,
sampler, or analyzer directly, matching the same orchestration pattern as
processors/audio/audio_processor.py.

Session context ring buffer: a short-lived, in-memory, per-session_id
deque of recent observation summaries (bounded by
settings.stream_context_window_size), used to give the vision model
continuity across frames - "Frame 1: person enters room. Frame 2: person
sits at desk." - so it can reason about *change*, not just describe one
frame in isolation, per the Phase 7 spec's "Context Management" section.

Deliberately in-memory / single-process (a plain dict keyed by
session_id), not persisted to a database: this phase's spec explicitly
scopes context to "a short-lived context of recent observations," and says
to *prepare for* (not implement) future persistence, multimodal RAG, and
recording. This buffer's shape - a list of (timestamp, summary) pairs -
is exactly what a future persistence layer would durably store; swapping
the in-memory dict for e.g. a Redis-backed store or a SQLite table later
would not require changing this class's public interface
(process_frame/get_recent_context), only its internals - the same
"swap the internals, keep the interface" pattern rag/vector_store.py's
docstring uses for its own future Postgres migration.

Session expiry: sessions with no frames for `settings.stream_session_ttl_seconds`
are dropped on the next call to `cleanup_expired_sessions()` - bounding
memory growth from abandoned sessions (e.g. a browser tab closed without
ever calling a stop endpoint), since this process never otherwise learns a
session ended.
"""
import time
from collections import deque
from dataclasses import dataclass, field

from processors.streaming.frame_sampler import FrameSampler
from processors.streaming.stream_validator import StreamValidator
from processors.streaming.vision_stream import StreamObservation, VisionStreamAnalyzer


@dataclass
class FrameProcessingResult:
    """Result of processing one incoming frame."""

    sampled: bool
    observation: StreamObservation | None
    analysis: str


@dataclass
class _SessionContext:
    observations: deque[str] = field(default_factory=deque)
    last_activity_at: float = 0.0


class StreamProcessor:
    """Runs the validate -> sample -> analyze -> update-context pipeline for streamed frames."""

    def __init__(
        self,
        validator: StreamValidator,
        sampler: FrameSampler,
        analyzer: VisionStreamAnalyzer,
        context_window_size: int = 10,
        session_ttl_seconds: int = 300,
    ) -> None:
        self._validator = validator
        self._sampler = sampler
        self._analyzer = analyzer
        self._context_window_size = context_window_size
        self._session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, _SessionContext] = {}

    def process_frame(
        self,
        session_id: str,
        mime_type: str,
        frame_bytes: bytes,
        question: str | None,
    ) -> FrameProcessingResult:
        """
        Validate and (if this frame is due per the sampling strategy)
        analyze a single frame, updating the session's recent-observations
        context either way.

        A frame that is validated but *not* sampled (most frames, by
        design) returns `sampled=False` with no observation and no vision
        API call - this is the efficiency mechanism the whole pipeline
        exists for, not a fallback path.

        Raises:
            processors.streaming.stream_validator.StreamValidationError:
                if the frame fails validation.
            RuntimeError: if vision analysis fails (propagated from the
                Vision/LLM layer).
        """
        self._validator.validate(
            session_id=session_id, mime_type=mime_type, frame_bytes=frame_bytes
        )

        session = self._sessions.setdefault(session_id, _SessionContext())
        session.last_activity_at = time.monotonic()

        if not self._sampler.should_sample(session_id):
            return FrameProcessingResult(sampled=False, observation=None, analysis="")

        recent_context = self.get_recent_context(session_id)
        observation = self._analyzer.analyze_frame(
            frame_bytes=frame_bytes,
            mime_type=mime_type,
            recent_context=recent_context,
            question=question,
        )

        session.observations.append(observation.summary_line())
        while len(session.observations) > self._context_window_size:
            session.observations.popleft()

        return FrameProcessingResult(
            sampled=True, observation=observation, analysis=observation.raw_text
        )

    def get_recent_context(self, session_id: str) -> str:
        """
        Return this session's recent observation summaries as a single
        newline-joined, "Frame N: ..." labeled string, oldest first -
        ready to be embedded directly in the vision prompt.
        """
        session = self._sessions.get(session_id)
        if not session or not session.observations:
            return ""
        return "\n".join(
            f"Frame {i + 1}: {summary}" for i, summary in enumerate(session.observations)
        )

    def end_session(self, session_id: str) -> None:
        """Clear all state for a session (context buffer + sampling clock/counter)."""
        self._sessions.pop(session_id, None)
        self._sampler.reset_session(session_id)

    def cleanup_expired_sessions(self, now: float | None = None) -> int:
        """
        Drop sessions with no activity for longer than session_ttl_seconds.

        Returns the number of sessions removed. Intended to be called
        periodically (e.g. from a background task, or opportunistically on
        each request) rather than relying on every session receiving an
        explicit "stop" signal, since a client can disappear (closed tab,
        lost connection) without ever calling one.
        """
        current_time = now if now is not None else time.monotonic()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if current_time - session.last_activity_at > self._session_ttl_seconds
        ]
        for session_id in expired:
            self.end_session(session_id)
        return len(expired)
