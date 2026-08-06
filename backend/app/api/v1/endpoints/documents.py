"""
Document ingestion endpoint.

Pure HTTP layer: parses the multipart/form-data request (PDF file),
enforces the upload size limit and content type, calls the ingestion
service, maps domain errors to HTTP status codes, and returns the response
model. No PDF-parsing or chunking logic lives here.
"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.document import (
    DocumentChunk,
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

router = APIRouter(tags=["documents"])


def _read_and_validate_pdf_upload(file: UploadFile, file_bytes: bytes) -> None:
    """Shared multipart validation for /documents/ingest and /documents/upload."""
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Only application/pdf is supported.",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_size_mb}MB upload limit.",
        )


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest."),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentIngestResponse:
    """
    Ingest a PDF and return the full chunk breakdown. Kept alongside
    /documents/upload (below) for callers that want the detailed,
    per-chunk response; /documents/upload returns the terser
    {documentId, chunksCreated, status} contract the frontend uses.
    """
    file_bytes = await file.read()
    _read_and_validate_pdf_upload(file, file_bytes)

    try:
        result = ingestion_service.ingest_pdf(
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
            )
            for chunk in result.chunks
        ],
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload and ingest."),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentUploadResponse:
    """
    Phase 4A ingestion entrypoint used by the frontend's PdfUploader.

        upload PDF -> extract text -> split into chunks -> generate
        embeddings -> store inside ChromaDB

    This endpoint only prepares the knowledge base (uploads + indexes a
    document) - it does not answer questions. See POST /rag/ask for that.
    """
    file_bytes = await file.read()
    _read_and_validate_pdf_upload(file, file_bytes)

    try:
        result = ingestion_service.ingest_pdf(
            filename=file.filename or "unknown.pdf",
            file_bytes=file_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentUploadResponse(
        document_id=result.document_id,
        chunks_created=len(result.chunks),
        status=result.status,
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
        results = ingestion_service.search(query=request.query, top_k=request.top_k)
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
            )
            for result in results
        ],
    )
