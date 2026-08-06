"""
RAG business logic (FastAPI-facing wrapper).

Thin adapter around rag/rag_service.py's RAGService, mirroring the shape of
app/services/chat_service.py and app/services/vision_service.py: this is
the dependency-injection seam where configuration errors (missing GROQ
API key, embedding model failing to load, etc.) get turned into clean
HTTPExceptions instead of raw 500s. No RAG logic itself lives here - that's
all in rag/rag_service.py.
"""
from fastapi import HTTPException, status

from ai.llm_service import get_llm_service
from rag.embedding_service import get_embedding_service
from rag.rag_service import RAGAnswer, RAGService
from rag.retriever import Retriever
from rag.vector_store import get_vector_store


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
        )
        return RAGService(
            retriever=retriever,
            llm_service=get_llm_service(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG service is not configured: {exc}",
        ) from exc


__all__ = ["RAGAnswer", "RAGService", "get_rag_service"]
