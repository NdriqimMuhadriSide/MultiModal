"""
RAG question-answering endpoint.

Pure HTTP layer: parses the request, calls the RAG service, maps domain
errors to HTTP status codes, and returns the response model. The actual
embed -> search -> build-context -> prompt -> LLM pipeline lives in
rag/rag_service.py, not here.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.rag import (
    RAGAskRequest,
    RAGAskResponse,
    RAGChatRequest,
    RAGChatResponse,
    RAGChatSource,
    RAGSourceResponse,
)
from app.services.rag_service import get_rag_service
from rag.rag_service import RAGService

router = APIRouter(tags=["rag"])


@router.post("/rag/ask", response_model=RAGAskResponse)
def ask_rag(
    request: RAGAskRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGAskResponse:
    """
    question -> embedding -> Chroma search -> relevant chunks ->
    prompt with context -> LLM -> answer
    """
    try:
        result = rag_service.ask(question=request.question, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return RAGAskResponse(
        question=request.question,
        answer=result.answer,
        sources=[
            RAGSourceResponse(
                chunk_id=source.chunk_id,
                filename=source.filename,
                page_number=source.page,
                similarity=source.score,
            )
            for source in result.sources
        ],
    )


@router.post("/rag/chat", response_model=RAGChatResponse)
def chat_rag(
    request: RAGChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGChatResponse:
    """
    Phase 4B chat-over-documents endpoint.

        question -> question embedding -> vector search -> top 5 chunks ->
        prompt construction (system + context + question) -> LLM -> answer

    Returns a terser {answer, sources} shape than /rag/ask - just what the
    chat UI needs to render an answer with a citation list underneath it.
    """
    try:
        result = rag_service.ask(question=request.question)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return RAGChatResponse(
        answer=result.answer,
        sources=[
            RAGChatSource(
                filename=source.filename,
                page=source.page,
                chunk_id=source.chunk_id,
            )
            for source in result.sources
        ],
    )
