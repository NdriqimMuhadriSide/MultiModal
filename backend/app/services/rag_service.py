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
from app.schemas.rag import RAGChatSource
from rag.embedding_service import get_embedding_service
from rag.keyword_index import get_keyword_index
from rag.contextualizer import get_query_contextualizer
from rag.query_expansion import get_query_expander
from rag.rag_service import RAGAnswer, RAGService, RAGSource
from rag.reranker import get_reranker
from rag.retriever import Retriever
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def to_chat_sources(sources: list[RAGSource]) -> list[RAGChatSource]:
    """
    Project retrieval's citations onto the terser shape the chat UI renders.

    One function rather than the same comprehension in every endpoint that
    answers from documents: /rag/chat, /agent/ask, and the agent's SSE
    `sources` event all cite the same chunks, and a citation that changed
    shape depending on which endpoint answered would be a bug the frontend
    has to absorb. The streaming path serializes these with
    `model_dump(by_alias=True)`, so it emits the identical `chunkId` key the
    JSON responses do.
    """
    return [
        RAGChatSource(
            filename=source.filename,
            page=source.page,
            chunk_id=source.chunk_id,
            section=source.section,
        )
        for source in sources
    ]


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


def build_retriever() -> Retriever:
    """
    Assemble the configured Retriever from the shared singletons.

    Split out of `get_rag_service` because RAGService is no longer the only
    thing that needs one: agents/research_agent.py talks to a Retriever
    directly, since it accumulates evidence across several searches and
    generates once at the end rather than per search. Both callers must get
    the *same* retriever - same hybrid weights, same reranker, same
    expansion - or the research agent would quietly search a different
    corpus configuration than /rag/chat does.

    Raises:
        ValueError: if a component needs configuration that is missing (e.g.
            multi-query enabled without an API key). Callers turn this into
            a 503; it is left as a domain error here so non-HTTP callers
            aren't forced to catch an HTTPException.
    """
    return Retriever(
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


def get_rag_service() -> RAGService:
    """
    FastAPI dependency that builds a RAGService from a Retriever (wrapping
    the shared embedding service + vector store singletons) and the shared
    LLM service singleton.
    """
    try:
        return RAGService(
            retriever=build_retriever(),
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


__all__ = [
    "RAGAnswer",
    "RAGService",
    "build_retriever",
    "get_rag_service",
    "to_chat_sources",
]
