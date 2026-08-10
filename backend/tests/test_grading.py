import pytest

from rag.grading import AMBIGUOUS, CORRECT, INCORRECT, grade_retrieval
from rag.retriever import RetrievedChunk


def _chunk(rerank_score: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c",
        text="text",
        filename="a.pdf",
        page=1,
        score=0.5,
        rerank_score=rerank_score,
    )


def _grade(scores: list[float | None], accept: float = 0.5, reject: float = 0.02):
    return grade_retrieval([_chunk(s) for s in scores], accept_score=accept, reject_score=reject)


def test_a_clearly_relevant_best_candidate_is_correct():
    assert _grade([0.98, 0.01]).verdict == CORRECT


def test_nothing_relevant_is_incorrect():
    assert _grade([0.001, 0.0001]).verdict == INCORRECT


def test_a_middling_best_candidate_is_ambiguous():
    assert _grade([0.2]).verdict == AMBIGUOUS


def test_the_best_candidate_decides_not_the_average():
    """
    One good passage is enough to answer from, and a question with exactly one
    answer in the corpus is the normal case - averaging would refuse it.
    """
    assert _grade([0.99, 0.001, 0.001, 0.001, 0.001]).verdict == CORRECT


def test_no_candidates_at_all_is_incorrect():
    assert grade_retrieval([], accept_score=0.5, reject_score=0.02).verdict == INCORRECT


def test_unscored_candidates_are_incorrect_rather_than_trusted():
    """Reranking off means no grade exists; answering anyway would be a guess."""
    grade = _grade([None, None])

    assert grade.verdict == INCORRECT
    assert grade.best_score is None


def test_the_grade_reports_the_score_behind_it():
    assert _grade([0.42, 0.1]).best_score == 0.42


def test_boundaries_are_inclusive_at_accept_and_exclusive_at_reject():
    assert _grade([0.5], accept=0.5).verdict == CORRECT
    assert _grade([0.02], reject=0.02).verdict == AMBIGUOUS
    assert _grade([0.019], reject=0.02).verdict == INCORRECT


def test_inverted_thresholds_are_rejected():
    """An empty ambiguous band would silently make this a plain cutoff."""
    with pytest.raises(ValueError, match="must not exceed"):
        _grade([0.5], accept=0.1, reject=0.9)


def test_equal_thresholds_are_allowed_and_remove_the_retry_band():
    grade = _grade([0.3], accept=0.5, reject=0.5)

    assert grade.verdict == INCORRECT


def test_is_answerable_is_true_for_everything_except_incorrect():
    assert _grade([0.99]).is_answerable
    assert _grade([0.2]).is_answerable
    assert not _grade([0.0001]).is_answerable
