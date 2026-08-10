"""
Reranker (cross-encoder).

Responsibility: given a question and a shortlist of candidate chunks, score
how well each one actually answers it. Pure text-in, numbers-out - no
knowledge of chunks, metadata, or the vector store, in the same spirit as
rag/embedding_service.py.

Why this exists as a second scoring pass, when retrieval already scored
everything: the two models are asked different questions.

    bi-encoder (rag/embedding_service.py)
        encode(question) -> vector,  encode(chunk) -> vector,  compare
        The chunk's vector is computed at ingestion, before the question
        exists. The model never sees the pair together, so "does this chunk
        answer this question" is a judgment it structurally cannot make -
        only "are these two texts about similar things".

    cross-encoder (this module)
        encode(question, chunk) -> one relevance score
        Reads both at once, with attention running across the boundary, so
        it can tell that a chunk about refund *eligibility* answers "how do I
        get my money back" and a chunk about refund *accounting* does not.

The cost of that is the reason retrieval can't just use this model: nothing
can be precomputed. Scoring N chunks means N forward passes at query time,
so it is viable over a shortlist of ~20 and hopeless over a corpus of
100,000. Hence the funnel - cheap retrieval casts wide (rag/retriever.py
over-fetches RETRIEVAL_OVERFETCH x top_k), this narrows.

Local and free, like the embeddings: ms-marco-MiniLM-L-6-v2 is ~80MB and
runs on-device, so there is no API key and no per-query billing. Swapping in
a hosted reranker (Cohere Rerank, Voyage) means replacing this file and
nothing else - callers depend on `score()`, not on how the numbers are made.
"""
import logging
import math

from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)


def _sigmoid(value: float) -> float:
    """
    Squash a raw cross-encoder logit into (0, 1).

    ms-marco cross-encoders emit unbounded logits - about +4.5 for a good
    match and -11 for an unrelated one. Monotonic, so this changes no
    ordering whatsoever; it exists so RERANK_MIN_SCORE can be a portable
    number. A threshold written against raw logits would silently mean
    something different the moment the model is swapped, because logit scales
    are a property of how a model was trained, not of relevance.
    """
    # Split by sign for numerical stability: the naive 1/(1+exp(-x)) form
    # overflows for large negative logits, which is exactly the range an
    # irrelevant chunk lands in (-11 and below).
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


class Reranker:
    """Scores (question, chunk) pairs with a cross-encoder."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        # Loaded on first use rather than here. Constructing a Reranker
        # happens in the FastAPI dependency on every request that touches
        # RAG, while actually scoring only happens when there are candidates
        # to score - so an eager load would download ~80MB in processes that
        # never rerank anything, including a server that only ever serves
        # /health.
        self._model: CrossEncoder | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading cross-encoder reranker '%s'.", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """
        Return one relevance score in (0, 1) per text, in input order.

        Scores are comparable within a single call - which is all the caller
        needs, since ranking is always within one question's candidates.
        Across questions they are only roughly comparable, so treat
        RERANK_MIN_SCORE as a coarse floor rather than a calibrated
        probability.

        Raises:
            ValueError: if `query` is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        if not texts:
            # Not an error: retrieval legitimately returns nothing for a
            # question the corpus can't answer, and the caller shouldn't have
            # to special-case that before asking for scores.
            return []

        # predict() batches internally, so this is one batched forward pass
        # rather than len(texts) separate ones.
        logits = self._load().predict([(query, text) for text in texts])
        return [_sigmoid(float(logit)) for logit in logits]


_reranker_instance: Reranker | None = None


def get_reranker() -> Reranker:
    """
    Return a process-wide Reranker.

    Shared so the model is loaded once per process rather than once per
    request. Not decorated with @lru_cache, matching get_vector_store() and
    get_keyword_index() - tests need to swap it without a cached instance
    surviving across runs.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(model_name=settings.rerank_model_name)
    return _reranker_instance
