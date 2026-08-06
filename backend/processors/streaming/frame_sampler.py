"""
Frame sampler.

Responsibility: decide whether an incoming, already-validated frame should
actually be sent to the (expensive, slow) vision model, or dropped.

This is the single most important efficiency decision in the whole
streaming pipeline. A camera/screen MediaStream can produce 24-60 frames
per second; sending every one of them to a vision LLM would be both
prohibitively expensive (vision API calls are priced per image) and
pointless (consecutive frames 30ms apart carry essentially the same
information). See the "why sending every frame is inefficient" and "frame
sampling strategies" explanations for the full accuracy/latency/cost
trade-off - this module is where that policy is enforced.

Two configurable strategies (Settings.stream_sampling_strategy):

    "interval"     - sample once every N seconds, regardless of how many
                      frames arrived in between. Decoupled from the
                      underlying stream's actual fps, which can vary by
                      device/browser - this is the default (2 seconds),
                      matching the Phase 7 spec.
    "frame_count"   - sample every Nth frame received, regardless of
                      elapsed time. Simpler to reason about per-frame, but
                      its effective real-world time interval shifts if the
                      client's capture rate changes.

State is tracked per session_id (one browser tab's camera or screen share
is one session) so two concurrent sessions don't interfere with each
other's sampling clocks/counters. This is intentionally in-memory,
single-process state (a plain dict) - see stream_processor.py's docstring
for why that's a deliberate, documented limitation for this phase, not an
oversight.
"""
import time
from dataclasses import dataclass, field


@dataclass
class _SessionSamplingState:
    # None means "never sampled yet" - distinct from 0.0, which would
    # incorrectly make the very first frame's elapsed time depend on how
    # large `now` happens to be (e.g. a real time.monotonic() value, which
    # is arbitrary and can be large) rather than always sampling
    # immediately on a session's first frame.
    last_sampled_at: float | None = None
    frames_seen: int = 0


class FrameSampler:
    """
    Decides, per session, whether "now" is a sampling point according to
    the configured strategy.
    """

    def __init__(
        self,
        strategy: str = "interval",
        interval_seconds: float = 2.0,
        frame_count: int = 10,
    ) -> None:
        if strategy not in ("interval", "frame_count"):
            raise ValueError(
                f"Unknown sampling strategy '{strategy}'. Expected 'interval' or 'frame_count'."
            )
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")
        if frame_count <= 0:
            raise ValueError("frame_count must be positive.")

        self._strategy = strategy
        self._interval_seconds = interval_seconds
        self._frame_count = frame_count
        self._session_state: dict[str, _SessionSamplingState] = {}

    def should_sample(self, session_id: str, now: float | None = None) -> bool:
        """
        Return True if this frame (for `session_id`) should be sent for
        vision analysis, False if it should be dropped.

        Every call counts as "a frame arrived" for that session, whether
        or not it's sampled - `frame_count` strategy needs every frame
        counted to know when the Nth one arrives, and `interval` strategy's
        clock is unaffected by frame count either way.

        `now` is injectable (defaults to `time.monotonic()`) purely so
        tests can control elapsed time deterministically instead of
        sleeping in real time.
        """
        current_time = now if now is not None else time.monotonic()
        state = self._session_state.setdefault(session_id, _SessionSamplingState())
        state.frames_seen += 1

        if self._strategy == "interval":
            if state.last_sampled_at is None or (
                current_time - state.last_sampled_at >= self._interval_seconds
            ):
                state.last_sampled_at = current_time
                return True
            return False

        # strategy == "frame_count": sample the 1st, (N+1)th, (2N+1)th, ...
        # frame - using (frames_seen - 1) % N rather than frames_seen % N
        # so frame_count=1 correctly samples every frame instead of never
        # matching (frames_seen % 1 is always 0, never 1).
        if (state.frames_seen - 1) % self._frame_count == 0:
            return True
        return False

    def reset_session(self, session_id: str) -> None:
        """Clear sampling state for a session (e.g. when a stream stops)."""
        self._session_state.pop(session_id, None)
