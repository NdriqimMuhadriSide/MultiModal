import pytest

from processors.streaming.frame_sampler import FrameSampler
from processors.streaming.stream_processor import StreamProcessor
from processors.streaming.stream_validator import StreamValidationError, StreamValidator
from processors.streaming.vision_stream import StreamObservation


class FakeAnalyzer:
    def __init__(self, observation: StreamObservation) -> None:
        self._observation = observation
        self.calls: list[dict] = []

    def analyze_frame(self, frame_bytes, mime_type, recent_context, question):
        self.calls.append(
            {
                "frame_bytes": frame_bytes,
                "mime_type": mime_type,
                "recent_context": recent_context,
                "question": question,
            }
        )
        return self._observation


def _processor(analyzer=None, sampler=None, **kwargs) -> StreamProcessor:
    return StreamProcessor(
        validator=StreamValidator(max_frame_size_mb=5),
        sampler=sampler or FrameSampler(strategy="frame_count", frame_count=1),
        analyzer=analyzer or FakeAnalyzer(StreamObservation(objects=["laptop"])),
        context_window_size=kwargs.get("context_window_size", 10),
        session_ttl_seconds=kwargs.get("session_ttl_seconds", 300),
    )


def test_process_frame_analyzes_when_sampled():
    analyzer = FakeAnalyzer(StreamObservation(objects=["laptop"], raw_text="A laptop is visible."))
    processor = _processor(analyzer=analyzer)

    result = processor.process_frame(
        session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question="what's here?"
    )

    assert result.sampled is True
    assert result.observation.objects == ["laptop"]
    assert result.analysis == "A laptop is visible."
    assert len(analyzer.calls) == 1


def test_process_frame_skips_analysis_when_not_sampled():
    sampler = FrameSampler(strategy="frame_count", frame_count=2)
    analyzer = FakeAnalyzer(StreamObservation(objects=["laptop"]))
    processor = _processor(analyzer=analyzer, sampler=sampler)

    first = processor.process_frame(session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question=None)
    second = processor.process_frame(session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question=None)

    assert first.sampled is True
    assert second.sampled is False
    assert second.observation is None
    assert second.analysis == ""
    assert len(analyzer.calls) == 1


def test_process_frame_raises_before_analyzing_on_invalid_frame():
    analyzer = FakeAnalyzer(StreamObservation())
    processor = _processor(analyzer=analyzer)

    with pytest.raises(StreamValidationError):
        processor.process_frame(session_id="", mime_type="image/jpeg", frame_bytes=b"fake", question=None)

    assert len(analyzer.calls) == 0


def test_process_frame_builds_up_session_context_across_frames():
    analyzer = FakeAnalyzer(StreamObservation())
    processor = _processor(analyzer=analyzer)
    # Override the fake to return different observations per call.
    observations = [
        StreamObservation(activities=["person enters room"]),
        StreamObservation(activities=["person sits at desk"]),
    ]

    def analyze_frame(frame_bytes, mime_type, recent_context, question):
        analyzer.calls.append({"recent_context": recent_context})
        return observations[len(analyzer.calls) - 1]

    analyzer.analyze_frame = analyze_frame

    processor.process_frame(session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question=None)
    processor.process_frame(session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question=None)

    # Second call should have seen the first frame's summary as context.
    assert "person enters room" in analyzer.calls[1]["recent_context"]

    context = processor.get_recent_context("session-a")
    assert "Frame 1: person enters room" in context
    assert "Frame 2: person sits at desk" in context


def test_context_window_size_bounds_stored_observations():
    processor = _processor(context_window_size=2)

    for _ in range(5):
        processor.process_frame(session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question=None)

    context = processor.get_recent_context("session-a")
    # Only the last 2 observations should remain.
    assert context.count("Frame") == 2


def test_get_recent_context_is_empty_for_unknown_session():
    processor = _processor()

    assert processor.get_recent_context("never-seen") == ""


def test_end_session_clears_context_and_sampling_state():
    processor = _processor()
    processor.process_frame(session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question=None)
    assert processor.get_recent_context("session-a") != ""

    processor.end_session("session-a")

    assert processor.get_recent_context("session-a") == ""


def test_cleanup_expired_sessions_removes_stale_sessions():
    import time

    processor = _processor(session_ttl_seconds=10)
    processor.process_frame(session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question=None)

    # Well past session_ttl_seconds relative to the real time.monotonic()
    # clock process_frame recorded activity against (monotonic() is
    # process-uptime-based, so a hardcoded absolute value like 1000.0
    # isn't a reliable "far in the future" reference).
    removed = processor.cleanup_expired_sessions(now=time.monotonic() + 3600)

    assert removed == 1
    assert processor.get_recent_context("session-a") == ""


def test_cleanup_expired_sessions_keeps_active_sessions():
    import time

    processor = _processor(session_ttl_seconds=300)
    processor.process_frame(session_id="session-a", mime_type="image/jpeg", frame_bytes=b"fake", question=None)

    removed = processor.cleanup_expired_sessions(now=time.monotonic())

    assert removed == 0
