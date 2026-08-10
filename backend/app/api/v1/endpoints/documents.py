"""
Document ingestion endpoint.

Pure HTTP layer: parses the multipart/form-data request, enforces the upload
size limit and file type, calls the ingestion service, maps domain errors to
HTTP status codes, and returns the response model. No parsing or chunking
logic lives here.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.document import (
    DocumentChunk,
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentSearchResult,
    DocumentSummary,
    DocumentUploadResponse,
)
from app.services.document_service import (
    DocumentIngestionService,
    get_document_ingestion_service,
)
from rag.loaders import SUPPORTED_EXTENSIONS, supported_extensions_label

router = APIRouter(tags=["documents"])


def _validate_upload(file: UploadFile, file_bytes: bytes) -> None:
    """
    Shared multipart validation for /documents/ingest and /documents/upload.

    The file type is judged by extension, not by the request's Content-Type.
    Browsers are unreliable about the latter - a .md or .csv normally arrives
    as text/plain or application/octet-stream, and the same .docx turns up
    under several different types depending on the client - so trusting it
    would reject files this server can read perfectly well. The extension is
    what the user actually named the file, and it is what rag/loaders
    dispatches on, so validating the same thing keeps the two in step.
    """
    extension = Path(file.filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{extension or file.filename}'. "
                f"Supported types: {supported_extensions_label()}."
            ),
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_size_mb}MB upload limit.",
        )


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="Document to ingest (PDF, DOCX, HTML, Markdown, CSV, TXT)."),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentIngestResponse:
    """
    Ingest a document and return the full chunk breakdown. Kept alongside
    /documents/upload (below) for callers that want the detailed,
    per-chunk response; /documents/upload returns the terser
    {documentId, chunksCreated, status} contract the frontend uses.
    """
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)

    try:
        result = ingestion_service.ingest_document(
            filename=file.filename or "unknown.pdf",
            file_bytes=file_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentIngestResponse(
        filename=result.filename,
        page_count=result.page_count,
        chunk_count=len(result.chunks),
        chunks=[
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                section=chunk.section,
            )
            for chunk in result.chunks
        ],
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document to upload and ingest (PDF, DOCX, HTML, Markdown, CSV, TXT)."),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentUploadResponse:
    """
    Phase 4A ingestion entrypoint used by the frontend's PdfUploader.

        upload -> load into blocks (rag/loaders) -> split into chunks ->
        generate embeddings -> store inside ChromaDB

    This endpoint only prepares the knowledge base (uploads + indexes a
    document) - it does not answer questions. See POST /rag/ask for that.
    """
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)

    try:
        result = ingestion_service.ingest_document(
            filename=file.filename or "unknown.pdf",
            file_bytes=file_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentUploadResponse(
        document_id=result.document_id,
        chunks_created=len(result.chunks),
        status=result.status,
        deduplicated=result.deduplicated,
        failure_reason=result.failure_reason,
    )


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentDeleteResponse:
    """
    Remove a document and every chunk it was split into.

    The counterpart to /documents/upload: without it an ingested document
    is permanent, and the only way to stop it being retrieved is to delete
    the whole Chroma directory.

    Returns 200 with the number of chunks removed, or 404 if the document
    was never registered. Deleting an already-deleted document is a 404
    rather than a silent success, so a stale UI list gets told the truth.
    """
    result = ingestion_service.delete_document(document_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document with id '{document_id}'.",
        )

    return DocumentDeleteResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunks_deleted=result.chunks_deleted,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentListResponse:
    """Return every uploaded document, most recently uploaded first."""
    records = ingestion_service.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentSummary(
                document_id=record.document_id,
                filename=record.filename,
                page_count=record.page_count,
                chunk_count=record.chunk_count,
                status=record.status,
                created_at=record.created_at,
                failure_reason=record.failure_reason,
                title=record.title,
                author=record.author,
                subject=record.subject,
                document_date=record.document_date,
            )
            for record in records
        ]
    )


@router.post("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentSearchResponse:
    """
    Embed the query and return the most similar previously-ingested chunks.

        question -> embedding -> Chroma similarity search -> ranked chunks

    This is the retrieval half of RAG. A later task will feed these results
    into an LLM prompt to actually generate a grounded answer.
    """
    try:
        results = ingestion_service.search(
            query=request.query,
            top_k=request.top_k,
            author=request.author,
            title=request.title,
            document_date=request.document_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentSearchResponse(
        query=request.query,
        results=[
            DocumentSearchResult(
                chunk_id=result.chunk_id,
                filename=result.metadata.get("filename", "unknown"),
                page_number=result.metadata.get("page_number", 0),
                similarity=result.similarity,
                text=result.text,
                section=result.metadata.get("section", ""),
                title=result.metadata.get("title", ""),
                author=result.metadata.get("author", ""),
            )
            for result in results
        ],
    )
