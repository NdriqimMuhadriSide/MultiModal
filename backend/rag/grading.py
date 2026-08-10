"""
Retrieval grading (corrective RAG).

Responsibility: look at what retrieval produced and say whether it is good
enough to answer from. Nothing else in the pipeline asks that question -
retrieval returns its best candidates whether or not any of them are any
good, and the generation step then answers from them as if they were.

That gap is the failure this closes. A question the corpus genuinely cannot
answer and a question where retrieval simply missed produce the same thing
today: five chunks and a confident answer built out of whichever passages
happened to rank highest. The model is instructed to say it doesn't know
(prompts/rag_prompts.py), but that instruction is being given alongside five
plausible-looking passages, which is the weakest possible position to ask
for a refusal from.

Three bands, following the corrective-RAG formulation:

    CORRECT     the best candidate is clearly relevant -> answer as normal
    AMBIGUOUS   something was found, but nothing convincing -> try again
                with rephrased queries before committing to an answer
    INCORRECT   nothing retrieved is relevant at all -> return nothing, so
                the pipeline refuses instead of inventing

The grade is read off the cross-encoder's score, which is why corrective
retrieval requires reranking. It has to be a score that means the same thing
from one question to the next, and neither of retrieval's own scores does:
cosine similarity varies with how the question was phrased, BM25 is
unbounded and corpus-relative. The cross-encoder was trained on exactly this
judgment - "does this passage answer this query" - and rag/reranker.py
squashes it into a fixed (0, 1) range, so a threshold written against it
holds across questions.

Grading only the *best* candidate, rather than counting how many clear the
bar: the answer needs one good passage, and a question answered by exactly
one chunk in the corpus is the normal case, not the exception. Requiring
several would refuse precisely the questions with a single crisp answer.
"""
from dataclasses import dataclass

CORRECT = "correct"
AMBIGUOUS = "ambiguous"
INCORRECT = "incorrect"


@dataclass
class Grade:
    """The verdict on one question's retrieval, and the number behind it."""

    verdict: str
    # The best cross-encoder score among the candidates, or None when there
    # were no candidates to score at all.
    best_score: float | None

    @property
    def is_answerable(self) -> bool:
        return self.verdict != INCORRECT


def grade_retrieval(chunks: list, accept_score: float, reject_score: float) -> Grade:
    """
    Band `chunks` by how relevant their best candidate is.

    Raises:
        ValueError: if the thresholds are inverted, which would make the
            AMBIGUOUS band empty and silently turn this into a plain cutoff.
    """
    if reject_score > accept_score:
        raise ValueError(
            "CORRECTIVE_REJECT_SCORE must not exceed CORRECTIVE_ACCEPT_SCORE."
        )

    scores = [chunk.rerank_score for chunk in chunks if chunk.rerank_score is not None]
    if not scores:
        # Either retrieval found nothing, or it found things nobody scored.
        # Both mean there is no evidence to answer from, and refusing is the
        # honest response to both.
        return Grade(verdict=INCORRECT, best_score=None)

    best = max(scores)
    if best >= accept_score:
        return Grade(verdict=CORRECT, best_score=best)
    if best < reject_score:
        return Grade(verdict=INCORRECT, best_score=best)
    return Grade(verdict=AMBIGUOUS, best_score=best)
