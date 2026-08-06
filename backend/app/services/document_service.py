"""
Document ingestion + retrieval business logic.

Orchestrates the rag/ modules end to end:
  1. rag/pdf_loader.py         - PDF bytes -> per-page plain text
  2. rag/text_splitter.py      - per-page text -> overlapping chunks
     (chunk_size/chunk_overlap sourced from settings, not the module's
     own defaults, so CHUNK_SIZE/CHUNK_OVERLAP env vars actually take
     effect - see app/core/config.py)
  3. rag/embedding_service.py  - chunk text -> embedding vectors
  4. rag/vector_store.py       - persist embedded chunks; similarity search
  5. rag/document_registry.py  - track the document itself (id, filename,
     page/chunk counts, status) as a first-class record, independent of
     the chunks stored in Chroma

...attaching the metadata each chunk needs for retrieval: document id,
filename, page number, and a stable chunk id. `ingest_pdf` covers steps
1-5 (upload -> registered -> stored in Chroma -> READY/FAILED). `search`
covers the query side (question -> embedding -> Chroma search -> ranked
chunks), which a future RAG-answering endpoint will build on to feed
retrieved chunks into an LLM prompt.
"""
from dataclasses import dataclass

from fastapi import HTTPException, status

from app.core.config import settings
from rag.document_registry import DocumentRegistry, DocumentRecord, get_document_registry
from rag.embedding_service import EmbeddingService, get_embedding_service
from rag.pdf_loader import load_pdf
from rag.text_splitter import split_text
from rag.vector_store import SearchResult, StoredChunk, VectorStore, get_vector_store


@dataclass
class IngestedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


@dataclass
class IngestResult:
    document_id: str
    filename: str
    page_count: int
    status: str
    chunks: list[IngestedChunk]


class DocumentIngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        document_registry: DocumentRegistry,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._document_registry = document_registry

    def ingest_pdf(self, filename: str, file_bytes: bytes) -> IngestResult:
        """
        Extract text from a PDF, split it into metadata-tagged chunks,
        embed each chunk, and persist them to the vector store.

        Pipeline: upload -> extract text -> split into chunks -> generate
        embeddings -> store inside ChromaDB. The document is registered as
        PROCESSING before any of that runs, then flipped to READY (chunks
        produced) or FAILED (zero extractable text, e.g. a scanned PDF with
        no OCR layer) once it completes - so a document_id always resolves
        to an honest status, even for documents that ingest with 0 chunks.

        Raises:
            ValueError: if the PDF is empty, unreadable, or has no pages
                (propagated from rag.pdf_loader).
        """
        document_id = DocumentRegistry.new_document_id()
        self._document_registry.create_document(document_id=document_id, filename=filename)

        try:
            pages = load_pdf(file_bytes)
        except ValueError:
            # Extraction failed outright (empty/corrupt PDF, no pages) -
            # record it as FAILED rather than leaving a PROCESSING row
            # stuck forever, then re-raise so the HTTP layer still returns
            # its 400 as before.
            self._document_registry.mark_failed(document_id)
            raise

        chunks: list[IngestedChunk] = []
        for page in pages:
            for text_chunk in split_text(
                page.text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            ):
                chunk_id = f"{document_id}::p{page.page_number}::c{text_chunk.chunk_index}"
                chunks.append(
                    IngestedChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=text_chunk.chunk_index,
                        text=text_chunk.text,
                    )
                )

        if chunks:
            embedded = self._embedding_service.embed_texts([chunk.text for chunk in chunks])
            self._vector_store.store_chunks(
                [
                    StoredChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        embedding=embedded_chunk.embedding,
                        metadata={
                            "document_id": chunk.document_id,
                            "filename": chunk.filename,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                        },
                    )
                    for chunk, embedded_chunk in zip(chunks, embedded)
                ]
            )
            self._document_registry.mark_ready(
                document_id, page_count=len(pages), chunk_count=len(chunks)
            )
            result_status = "READY"
        else:
            # Valid PDF, but no extractable text (e.g. scanned images with
            # no OCR layer) - a real outcome, not a crash, so it's recorded
            # as FAILED rather than silently returning an empty success.
            self._document_registry.mark_failed(document_id, page_count=len(pages))
            result_status = "FAILED"

        return IngestResult(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            status=result_status,
            chunks=chunks,
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Embed `query` and return the `top_k` most similar stored chunks.

        Raises:
            ValueError: if `query` is empty or top_k is not positive
                (propagated from embedding_service / vector_store).
        """
        query_embedding = self._embedding_service.embed_text(query)
        return self._vector_store.search(query_embedding, top_k=top_k)

    def list_documents(self) -> list[DocumentRecord]:
        """Return every uploaded document (most recent first) for the documents list UI."""
        return self._document_registry.list_documents()


def get_document_ingestion_service() -> DocumentIngestionService:
    """
    FastAPI dependency that builds a DocumentIngestionService.

    Wraps construction in try/except (consistent with chat_service /
    vision_service) so any configuration error - e.g. the embedding model
    failing to load - surfaces as a clean HTTPException instead of a raw 500.
    """
    try:
        return DocumentIngestionService(
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store(),
            document_registry=get_document_registry(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Document ingestion service is not configured: {exc}",
        ) from exc
