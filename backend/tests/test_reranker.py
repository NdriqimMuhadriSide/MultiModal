import math

import pytest

from rag.reranker import Reranker, _sigmoid


class FakeCrossEncoder:
    """Stands in for the real model so tests don't download or run one."""

    def __init__(self, logits: list[float]) -> None:
        self._logits = logits
        self.pairs: list[tuple[str, str]] | None = None
        self.predict_calls = 0

    def predict(self, pairs):
        self.pairs = list(pairs)
        self.predict_calls += 1
        return self._logits[: len(self.pairs)]


def _reranker(logits: list[float]) -> tuple[Reranker, FakeCrossEncoder]:
    model = FakeCrossEncoder(logits)
    reranker = Reranker(model_name="fake")
    reranker._model = model
    return reranker, model


# ---------------------------------------------------------------------------
# Score conversion
# ---------------------------------------------------------------------------


def test_sigmoid_maps_logits_into_the_unit_interval():
    """RERANK_MIN_SCORE has to mean the same thing whichever model is loaded."""
    assert 0.0 < _sigmoid(-11.4) < 0.01
    assert 0.98 < _sigmoid(4.5) < 1.0
    assert _sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_is_monotonic_so_ordering_is_untouched():
    logits = [-11.4, -3.0, 0.0, 2.5, 4.5]
    scores = [_sigmoid(value) for value in logits]

    assert scores == sorted(scores)


def test_sigmoid_does_not_overflow_on_large_negative_logits():
    """The naive 1/(1+exp(-x)) form raises OverflowError here."""
    assert _sigmoid(-800.0) == pytest.approx(0.0)
    assert _sigmoid(800.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_returns_one_score_per_text_in_input_order():
    reranker, _ = _reranker([4.5, -11.0, 1.0])

    scores = reranker.score("a question", ["good", "bad", "middling"])

    assert len(scores) == 3
    assert scores[0] > scores[2] > scores[1]


def test_score_pairs_every_text_with_the_query():
    reranker, model = _reranker([1.0, 2.0])

    reranker.score("what is the refund policy", ["chunk a", "chunk b"])

    assert model.pairs == [
        ("what is the refund policy", "chunk a"),
        ("what is the refund policy", "chunk b"),
    ]


def test_score_uses_a_single_batched_call():
    """One batched forward pass, not one per candidate."""
    reranker, model = _reranker([1.0] * 20)

    reranker.score("a question", [f"chunk {i}" for i in range(20)])

    assert model.predict_calls == 1


def test_score_of_nothing_is_nothing():
    """An unanswerable question legitimately retrieves zero candidates."""
    reranker, model = _reranker([])

    assert reranker.score("a question", []) == []
    assert model.predict_calls == 0


def test_score_rejects_an_empty_query():
    reranker, _ = _reranker([1.0])

    with pytest.raises(ValueError):
        reranker.score("   ", ["chunk"])


def test_the_model_is_not_loaded_until_something_is_scored():
    """Constructing one happens per request; loading ~80MB must not."""
    reranker = Reranker(model_name="definitely-not-a-real-model")

    assert reranker._model is None
    assert reranker.model_name == "definitely-not-a-real-model"


def test_scores_are_all_in_the_unit_interval():
    reranker, _ = _reranker([-40.0, -1.0, 0.0, 3.0, 40.0])

    scores = reranker.score("q", ["a", "b", "c", "d", "e"])

    assert all(0.0 <= score <= 1.0 for score in scores)
    assert not any(math.isnan(score) for score in scores)
