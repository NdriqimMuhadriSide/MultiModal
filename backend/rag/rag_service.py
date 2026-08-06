"""
RAG (Retrieval-Augmented Generation) service.

Orchestrates the full pipeline:

    user question
        -> Retriever.retrieve()             (rag/retriever.py: embed
                                               question -> search ChromaDB
                                               -> similarity-filtered chunks)
        -> build_context()                  (rag/context_builder.py:
                                               chunks -> Document/Page/
                                               Content text block)
        -> format_rag_prompt()              (prompts/rag_prompts.py:
                                               system prompt + context +
                                               question)
        -> LLMService.generate_response()   (ai/llm_service.py)
        -> return the generated answer (+ which chunks it was grounded in)

This module is the "RAG layer" in the project's layering rules:

    API layer -> Service layer -> RAG layer -> AI layer -> Storage layer

It is the only place allowed to depend on both the retrieval side
(rag/retriever.py, rag/context_builder.py) and the generation side
(ai/llm_service.py) - it is the glue between them. The FastAPI layer
(app/services/rag_service.py, app/api/v1/endpoints/rag.py) only wraps this
in HTTP/dependency-injection concerns; it never calls the retriever, the
vector store, or the LLM directly.
"""
from collections.abc import Iterator
from dataclasses import dataclass

from ai.llm_service import LLMService
from prompts.rag_prompts import format_rag_prompt
from rag.context_builder import build_context
from rag.retriever import DEFAULT_TOP_K, Retriever


@dataclass
class RAGSource:
    """A retrieved chunk that was included in the context sent to the LLM."""

    chunk_id: str
    filename: str
    page: int
    score: float


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RAGSource]


class RAGService:
    """Ties retrieval + context building + prompt construction + LLM generation into one ask() call."""

    def __init__(self, retriever: Retriever, llm_service: LLMService) -> None:
        self._retriever = retriever
        self._llm_service = llm_service

    def ask(self, question: str, top_k: int = DEFAULT_TOP_K) -> RAGAnswer:
        """
        Answer `question` using only content retrieved from the vector store.

            question -> embed -> search -> top K chunks -> build context ->
            prompt -> LLM -> answer

        Raises:
            ValueError: if `question` is empty or top_k is not positive
                (propagated from rag.retriever).
            RuntimeError: if the LLM call fails (propagated from llm_service).
        """
        chunks = self._retriever.retrieve(question, top_k=top_k)

        context = build_context(chunks)
        if not context:
            # No relevant chunks at all - skip the LLM call rather than
            # asking it to answer from an empty context block, and give a
            # deterministic, honest response instead.
            return RAGAnswer(
                answer="I don't know. No relevant information was found in the "
                "ingested documents to answer this question.",
                sources=[],
            )

        prompt = format_rag_prompt(context=context, question=question)
        answer = self._llm_service.generate_response(prompt)

        sources = [
            RAGSource(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page=chunk.page,
                score=chunk.score,
            )
            for chunk in chunks
        ]

        return RAGAnswer(answer=answer, sources=sources)

    def stream_ask(
        self, question: str, top_k: int = DEFAULT_TOP_K
    ) -> tuple[list[RAGSource], Iterator[str]]:
        """
        The streaming counterpart to `ask`: same pipeline, with the final
        generation step yielded piece by piece.

        Returns `(sources, chunks)`. Retrieval happens eagerly at call time -
        it is a local embedding plus a Chroma query, fast and not worth
        deferring, and the caller needs the sources before the answer text
        rather than after it. Only the LLM call is lazy.

        The empty-context branch yields its fixed sentence as a single
        chunk rather than returning it differently, so callers have exactly
        one shape to consume whether or not anything was retrieved.

        Raises:
            ValueError: if `question` is empty or top_k is not positive.
            RuntimeError: if the LLM call fails - possibly mid-stream.
        """
        chunks = self._retriever.retrieve(question, top_k=top_k)
        context = build_context(chunks)

        if not context:
            def no_context() -> Iterator[str]:
                yield (
                    "I don't know. No relevant information was found in the "
                    "ingested documents to answer this question."
                )

            return [], no_context()

        sources = [
            RAGSource(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page=chunk.page,
                score=chunk.score,
            )
            for chunk in chunks
        ]

        prompt = format_rag_prompt(context=context, question=question)
        return sources, self._llm_service.stream_response(prompt)
