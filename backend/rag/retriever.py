"""
Retriever.

Responsibility: the "retrieval" half of RAG, isolated from context/prompt
construction and generation - turn a user's question into a ranked list of
relevant chunks:

    question -> embed question -> search ChromaDB -> filter by similarity
    -> top K relevant chunks

This is the RAG layer's first stage (see rag/rag_service.py for how it's
composed with rag/context_builder.py and prompts/rag_prompts.py). Kept
separate from rag/rag_service.py so retrieval-quality concerns - top_k, the
minimum similarity threshold, and eventually re-ranking or metadata
filtering - can be tuned and tested in one place, independent of how the
retrieved chunks get turned into a prompt or a generated answer.

Layering rule: this module only talks to rag/embedding_service.py and
rag/vector_store.py (the storage layer). It never imports ai/llm_service.py
or prompts/ - retrieval and generation stay separate concerns.
"""
from dataclasses import dataclass

from rag.embedding_service import EmbeddingService
from rag.vector_store import VectorStore

# Default number of chunks returned per query - matches the Phase 4B spec
# ("top_k = 5").
DEFAULT_TOP_K = 5

# Chunks with a similarity below this are dropped before ever reaching the
# LLM. Without this, a low-relevance chunk could still be included just
# because top_k asked for N results even when the store only has weak
# matches - which would work against "answer only from context, say you
# don't know otherwise."
DEFAULT_MIN_SIMILARITY = 0.2


@dataclass
class RetrievedChunk:
    """
    A single chunk retrieved for a question, with its source metadata.

    Field names match the Phase 4B retriever contract exactly:
        { "text": "", "filename": "", "page": "", "score": "" }
    """

    chunk_id: str
    text: str
    filename: str
    page: int
    score: float


class Retriever:
    """Embeds a question and searches the vector store for relevant chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._min_similarity = min_similarity

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """
        Return up to `top_k` chunks most relevant to `question`, excluding
        any below the configured similarity threshold.

        Raises:
            ValueError: if `question` is empty or top_k is not positive
                (propagated from embedding_service / vector_store).
        """
        question_embedding = self._embedding_service.embed_text(question)
        results = self._vector_store.search(question_embedding, top_k=top_k)

        return [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                text=result.text,
                filename=result.metadata.get("filename", "unknown"),
                page=result.metadata.get("page_number", 0),
                score=result.similarity,
            )
            for result in results
            if result.similarity >= self._min_similarity
        ]
