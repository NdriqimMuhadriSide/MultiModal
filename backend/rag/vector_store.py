"""
Vector store.

Responsibility: persist embedded text chunks and search over them by
similarity. This is the only file in the codebase that talks to ChromaDB -
nothing else should import `chromadb` directly.

Backed by Chroma's PersistentClient, which writes to a local directory
(settings.chroma_persist_dir) - no separate database server to run, which
keeps this consistent with the rest of the project's "free, local, no
extra infrastructure" approach. Swapping to pgvector later would mean
replacing this file's internals (and standing up a Postgres instance) while
keeping the same store_chunks/search public interface, so callers
(app/services/document_service.py, the future RAG query service) wouldn't
need to change.

The collection is explicitly configured to use cosine similarity
(hnsw:space = "cosine") rather than Chroma's default squared-L2 distance,
since cosine is what the rest of this project's explanations (and most RAG
tutorials) assume when talking about "similarity search."
"""
from dataclasses import dataclass

import chromadb

from app.core.config import settings


@dataclass
class StoredChunk:
    """A chunk of text plus the metadata needed to store it in the vector store."""

    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict


@dataclass
class SearchResult:
    """A single retrieved chunk, ranked by similarity to the query."""

    chunk_id: str
    text: str
    metadata: dict
    similarity: float  # cosine similarity in [-1, 1]; higher = more similar


class VectorStore:
    """Wraps a persistent Chroma collection for chunk storage and similarity search."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def store_chunks(self, chunks: list[StoredChunk]) -> None:
        """
        Persist a batch of embedded chunks (and their metadata) to the
        collection. Re-adding a chunk_id that already exists overwrites it
        (upsert), so re-ingesting the same document is safe.

        Raises:
            ValueError: if `chunks` is empty.
        """
        if not chunks:
            raise ValueError("chunks must not be empty.")

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """
        Find the `top_k` chunks most similar to `query_embedding`.

        Raises:
            ValueError: if `query_embedding` is empty or top_k is not positive.
        """
        if not query_embedding:
            raise ValueError("query_embedding must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        chunk_count = self._collection.count()
        if chunk_count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, chunk_count),
        )

        search_results: list[SearchResult] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # With hnsw:space="cosine", Chroma returns cosine *distance*
            # (1 - cosine_similarity), so convert back to similarity for a
            # more intuitive "higher is better" result.
            similarity = 1 - distance
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    similarity=similarity,
                )
            )

        return search_results

    def count(self) -> int:
        """Number of chunks currently stored in the collection."""
        return self._collection.count()


_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Return a process-wide VectorStore instance.

    Not decorated with @lru_cache (unlike the other get_*_service functions)
    because tests need to point this at a temporary directory via
    dependency/monkeypatch overrides without a stale cached instance
    lingering across test runs.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
    return _vector_store_instance
