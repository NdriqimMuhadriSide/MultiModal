"""
RAG (Retrieval-Augmented Generation) service.

Orchestrates the full pipeline:

    user question
        -> Retriever.retrieve()             (rag/retriever.py: dense search,
                                               BM25, or both fused - see
                                               RETRIEVAL_MODE)
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
    # The score the ranking was done on - a cosine similarity, a BM25 score or
    # a fused rank score depending on RETRIEVAL_MODE. See RetrievedChunk in
    # rag/retriever.py for why this isn't called `similarity`.
    score: float
    # Heading path the chunk came from (see rag/structure.py); "" when the
    # document has no detected structure.
    section: str = ""
    # Which half of hybrid retrieval surfaced this chunk, and what each scored
    # it. Carried all the way out to the API response because "the model cited
    # a chunk that dense search never saw" is the single most useful thing to
    # know when tuning retrieval.
    dense_score: float | None = None
    keyword_score: float | None = None
    matched_by: str = "dense"
    # The cross-encoder's verdict, or None when RERANK_ENABLED is off.
    rerank_score: float | None = None
    # The query phrasing that found this chunk - the original question
    # unless MULTI_QUERY_ENABLED is on and a variant got there first.
    found_by_query: str = ""


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RAGSource]


def _to_sources(chunks: list) -> list[RAGSource]:
    """
    Project retrieved chunks down to citations - everything about where the
    text came from, without the text itself (which the model already has).

    One function rather than the same comprehension in `ask` and `stream_ask`,
    so a field added to retrieval reaches both paths or neither.
    """
    return [
        RAGSource(
            chunk_id=chunk.chunk_id,
            filename=chunk.filename,
            page=chunk.page,
            score=chunk.score,
            section=chunk.section,
            dense_score=chunk.dense_score,
            keyword_score=chunk.keyword_score,
            matched_by=chunk.matched_by,
            rerank_score=chunk.rerank_score,
            found_by_query=chunk.found_by_query,
        )
        for chunk in chunks
    ]


class RAGService:
    """Ties retrieval + context building + prompt construction + LLM generation into one ask() call."""

    def __init__(
        self,
        retriever: Retriever,
        llm_service: LLMService,
        contextualizer=None,
    ) -> None:
        self._retriever = retriever
        self._llm_service = llm_service
        # Optional: without it, `history` is accepted and ignored, which is
        # exactly the behaviour that existed before contextualization.
        self._contextualizer = contextualizer

    def _retrieval_query(self, question: str, history) -> str:
        """
        The string retrieval should actually search for.

        Differs from the question the *user* asked only for follow-ups in a
        conversation - see rag/contextualizer.py. The user's own wording is
        what still reaches the answering prompt, because that is what they
        want answered; only the search term is rewritten.
        """
        if self._contextualizer is None:
            return question
        return self._contextualizer.contextualize(question, history)

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        history: list[dict[str, str]] | None = None,
    ) -> RAGAnswer:
        """
        Answer `question` using only content retrieved from the vector store.

            question -> embed -> search -> top K chunks -> build context ->
            prompt -> LLM -> answer

        Raises:
            ValueError: if `question` is empty or top_k is not positive
                (propagated from rag.retriever).
            RuntimeError: if the LLM call fails (propagated from llm_service).
        """
        chunks = self._retriever.retrieve(
            self._retrieval_query(question, history), top_k=top_k
        )

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

        return RAGAnswer(answer=answer, sources=_to_sources(chunks))

    def stream_ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        history: list[dict[str, str]] | None = None,
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
        chunks = self._retriever.retrieve(
            self._retrieval_query(question, history), top_k=top_k
        )
        context = build_context(chunks)

        if not context:
            def no_context() -> Iterator[str]:
                yield (
                    "I don't know. No relevant information was found in the "
                    "ingested documents to answer this question."
                )

            return [], no_context()

        prompt = format_rag_prompt(context=context, question=question)
        return _to_sources(chunks), self._llm_service.stream_response(prompt)
