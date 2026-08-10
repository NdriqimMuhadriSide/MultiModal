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


@dataclass
class ChunkRecord:
    """
    A stored chunk fetched by metadata rather than by similarity.

    Distinct from SearchResult because there is no query here and therefore
    no score to report, and distinct from StoredChunk because the embedding
    is not read back - callers that fetch a whole document want its text,
    not its vectors.
    """

    chunk_id: str
    text: str
    metadata: dict


def _as_chroma_filter(where: dict | None) -> dict | None:
    """
    Put a plain {field: value} map into the shape Chroma expects.

    Chroma accepts a single condition bare but requires several to be wrapped
    in an explicit `$and`, and rejects an empty dict outright - so callers get
    to pass the obvious thing and have it translated here.
    """
    conditions = {key: value for key, value in (where or {}).items() if value not in (None, "")}
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions
    return {"$and": [{key: value} for key, value in conditions.items()]}


class VectorStore:
    """Wraps a persistent Chroma collection for chunk storage and similarity search."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Bumped on every write. Read by fingerprint() - see there for why a
        # count alone isn't enough.
        self._writes = 0

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
        self._writes += 1

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Find the `top_k` chunks most similar to `query_embedding`, optionally
        restricted to chunks whose metadata matches `where`.

        The filter is the reason chunks carry their document's title, author
        and date (see app/services/document_service.py). Similarity alone
        cannot answer "anything by the compliance team" or "only last year's
        reports" - no embedding encodes a date - so those have to be a filter
        applied *before* ranking, which is exactly what Chroma's `where` does.

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
            # Asking for top_k after filtering still needs the whole
            # collection as the candidate pool, so the cap stays as-is;
            # Chroma applies `where` before ranking, not after.
            n_results=min(top_k, chunk_count),
            where=_as_chroma_filter(where),
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

    def get_by_document_id(self, document_id: str) -> list[ChunkRecord]:
        """
        Return every stored chunk belonging to `document_id`, in document
        order (page, then position within the page).

        This is the first read path that filters on metadata rather than
        similarity - `document_id` is written on every chunk by
        app/services/document_service.py but was, until now, never read
        back. Chroma's `get` returns flat lists (unlike `query`, which
        nests one list per query embedding) and gives no ordering
        guarantee, hence the explicit sort.

        Raises:
            ValueError: if `document_id` is empty.
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id must not be empty.")

        results = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )

        records = [
            ChunkRecord(chunk_id=chunk_id, text=text, metadata=metadata)
            for chunk_id, text, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]
        records.sort(
            key=lambda record: (
                record.metadata.get("page_number", 0),
                record.metadata.get("chunk_index", 0),
            )
        )
        return records

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete every chunk belonging to `document_id` and return how many
        were removed.

        Chroma's `delete` reports nothing about what it removed, so the ids
        are fetched first - that lookup is also what makes the return value
        honest for a document that was already gone (0 rather than a
        pretended success).

        Raises:
            ValueError: if `document_id` is empty.
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id must not be empty.")

        existing = self._collection.get(where={"document_id": document_id}, include=[])
        chunk_ids = existing["ids"]
        if not chunk_ids:
            return 0

        self._collection.delete(ids=chunk_ids)
        self._writes += 1
        return len(chunk_ids)

    def count(self) -> int:
        """Number of chunks currently stored in the collection."""
        return self._collection.count()

    def all_chunks(self) -> list[ChunkRecord]:
        """
        Return every stored chunk.

        Exists for rag/keyword_index.py, which needs the whole corpus to build
        an inverted index over it - a keyword index is not a lookup by id or
        by similarity, so neither existing read path can serve it.

        Deliberately unpaginated: the caller is going to hold all of it in
        memory anyway, so streaming it in pages would move the cost around
        rather than remove it. This is the method that stops being viable
        first if the corpus grows past what one process should hold, which is
        also the point at which BM25-in-memory should be replaced outright.
        """
        results = self._collection.get(include=["documents", "metadatas"])
        return [
            ChunkRecord(chunk_id=chunk_id, text=text, metadata=metadata)
            for chunk_id, text, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def fingerprint(self) -> tuple[int, int]:
        """
        A value that changes whenever the collection's contents change.

        Used by rag/keyword_index.py to decide whether its cached index is
        still valid. Two numbers rather than one because neither is sufficient
        alone: the write counter misses changes made by another process, and
        the count misses a delete-then-reingest that happens to land on the
        same total. Together they catch both, at the cost of one cheap
        `count()` per query.
        """
        return (self._writes, self._collection.count())


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
