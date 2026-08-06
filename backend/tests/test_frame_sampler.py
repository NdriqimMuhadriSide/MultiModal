import pytest

from processors.streaming.frame_sampler import FrameSampler


def test_interval_strategy_samples_first_frame_immediately():
    sampler = FrameSampler(strategy="interval", interval_seconds=2.0)

    assert sampler.should_sample("session-a", now=0.0) is True


def test_interval_strategy_drops_frames_before_interval_elapses():
    sampler = FrameSampler(strategy="interval", interval_seconds=2.0)

    assert sampler.should_sample("session-a", now=0.0) is True
    assert sampler.should_sample("session-a", now=0.5) is False
    assert sampler.should_sample("session-a", now=1.9) is False


def test_interval_strategy_samples_again_once_interval_elapses():
    sampler = FrameSampler(strategy="interval", interval_seconds=2.0)

    assert sampler.should_sample("session-a", now=0.0) is True
    assert sampler.should_sample("session-a", now=1.0) is False
    assert sampler.should_sample("session-a", now=2.1) is True


def test_interval_strategy_tracks_sessions_independently():
    sampler = FrameSampler(strategy="interval", interval_seconds=2.0)

    assert sampler.should_sample("session-a", now=0.0) is True
    # A brand new session should sample immediately too, unaffected by
    # session-a's clock.
    assert sampler.should_sample("session-b", now=0.1) is True


def test_frame_count_strategy_samples_first_and_every_nth_frame():
    sampler = FrameSampler(strategy="frame_count", frame_count=3)

    results = [sampler.should_sample("session-a", now=float(i)) for i in range(9)]

    assert results == [True, False, False, True, False, False, True, False, False]


def test_frame_count_strategy_with_count_one_samples_every_frame():
    sampler = FrameSampler(strategy="frame_count", frame_count=1)

    results = [sampler.should_sample("session-a", now=float(i)) for i in range(4)]

    assert results == [True, True, True, True]


def test_reset_session_clears_state_so_next_frame_samples_immediately():
    sampler = FrameSampler(strategy="interval", interval_seconds=2.0)
    sampler.should_sample("session-a", now=0.0)
    assert sampler.should_sample("session-a", now=0.5) is False

    sampler.reset_session("session-a")

    assert sampler.should_sample("session-a", now=0.6) is True


def test_invalid_strategy_raises_value_error():
    with pytest.raises(ValueError):
        FrameSampler(strategy="every_frame")


def test_non_positive_interval_raises_value_error():
    with pytest.raises(ValueError):
        FrameSampler(strategy="interval", interval_seconds=0)


def test_non_positive_frame_count_raises_value_error():
    with pytest.raises(ValueError):
        FrameSampler(strategy="frame_count", frame_count=0)
