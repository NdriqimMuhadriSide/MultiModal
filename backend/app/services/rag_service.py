"""
RAG business logic (FastAPI-facing wrapper).

Thin adapter around rag/rag_service.py's RAGService, mirroring the shape of
app/services/chat_service.py and app/services/vision_service.py: this is
the dependency-injection seam where configuration errors (missing GROQ
API key, embedding model failing to load, etc.) get turned into clean
HTTPExceptions instead of raw 500s. No RAG logic itself lives here - that's
all in rag/rag_service.py.
"""
import logging

from fastapi import HTTPException, status

from ai.llm_service import get_llm_service
from app.core.config import settings
from rag.embedding_service import get_embedding_service
from rag.keyword_index import get_keyword_index
from rag.contextualizer import get_query_contextualizer
from rag.query_expansion import get_query_expander
from rag.rag_service import RAGAnswer, RAGService
from rag.reranker import get_reranker
from rag.retriever import Retriever
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def _build_query_expander():
    """
    Build the query expander if either feature that uses it is enabled.

    The two use the same tool for opposite policies: multi-query rephrases
    every question up front, corrective RAG rephrases only the ones grading
    says went wrong.

    They also differ in how badly they need it. Multi-query *is* expansion, so
    without an LLM it cannot run and the missing key should surface as the
    configuration error it is. Corrective RAG only uses expansion for its
    retry - its grading and refusal need no LLM at all, and that is the half
    with the measured benefit - so a missing key degrades it to grade-only
    rather than taking the whole RAG endpoint down with a 503.
    """
    if settings.multi_query_enabled:
        return get_query_expander()

    if settings.corrective_rag_enabled:
        try:
            return get_query_expander()
        except ValueError:
            logger.warning(
                "CORRECTIVE_RAG_ENABLED is on but no LLM is configured; retrieval will "
                "be graded and may refuse, but poor results will not be retried."
            )
            return None

    return None


def get_rag_service() -> RAGService:
    """
    FastAPI dependency that builds a RAGService from a Retriever (wrapping
    the shared embedding service + vector store singletons) and the shared
    LLM service singleton.
    """
    try:
        retriever = Retriever(
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store(),
            # Built only for the modes that use it, so a dense-only
            # deployment never pays for an inverted index it won't query.
            keyword_index=(
                get_keyword_index()
                if settings.retrieval_mode in ("keyword", "hybrid")
                else None
            ),
            # Constructing one is free - rag/reranker.py loads its model on
            # first use, not here - so this costs nothing until a question
            # actually has candidates to rerank.
            reranker=get_reranker() if settings.rerank_enabled else None,
            # Built only when enabled: constructing one resolves the LLM
            # service, which raises without an API key - a deployment that
            # ingests and searches documents without one should keep working.
            query_expander=_build_query_expander(),
            multi_query=settings.multi_query_enabled,
            corrective=settings.corrective_rag_enabled,
        )
        return RAGService(
            retriever=retriever,
            llm_service=get_llm_service(),
            # Only built when enabled. It is never used for a question with
            # no history, so /rag/ask pays nothing for it either way.
            contextualizer=(
                get_query_contextualizer()
                if settings.query_contextualization_enabled
                else None
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG service is not configured: {exc}",
        ) from exc


__all__ = ["RAGAnswer", "RAGService", "get_rag_service"]
