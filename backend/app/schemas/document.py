"""
Pydantic request/response models for document ingestion and search.

The ingestion request is multipart/form-data (a PDF file), so it has no
request schema here - FastAPI parses that directly as an UploadFile
parameter. The search request is plain JSON, so it does have one
(DocumentSearchRequest).

DocumentUploadResponse/DocumentSummary/DocumentListResponse use camelCase
field aliases (documentId, chunksCreated, ...) rather than this codebase's
usual snake_case (see ChatResponse.conversation_id, etc.) - this is
intentional: they're the contract for the new POST /documents/upload and
GET /documents endpoints consumed by a TypeScript frontend, so the wire
format matches typical frontend/TS naming conventions directly. FastAPI
serializes responses using these aliases by default
(response_model_by_alias=True), and `populate_by_name=True` lets the
Python side keep constructing them with snake_case keyword arguments.
"""
from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    # Heading path this chunk sits under ("2. Methods > 2.1 Field Sampling"),
    # or "" for text under no heading.
    section: str = ""


class DocumentIngestResponse(BaseModel):
    filename: str
    page_count: int
    chunk_count: int
    chunks: list[DocumentChunk]


class DocumentUploadResponse(BaseModel):
    """Response body for POST /documents/upload."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(..., alias="documentId")
    chunks_created: int = Field(..., alias="chunksCreated")
    status: str
    # True when these exact bytes were already ingested, so the existing
    # document was returned rather than a duplicate created. The response is
    # otherwise indistinguishable from a fresh upload, and the UI wants to
    # say "already uploaded" instead of "uploaded".
    deduplicated: bool = False
    # Present only when status is FAILED. A FAILED upload still returns 200
    # (the request was fine; the file just had no usable text), so this is the
    # only way the uploader can tell the user what actually happened.
    failure_reason: str | None = Field(default=None, alias="failureReason")


class DocumentDeleteResponse(BaseModel):
    """Response body for DELETE /documents/{document_id}."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(..., alias="documentId")
    filename: str
    chunks_deleted: int = Field(..., alias="chunksDeleted")


class DocumentSummary(BaseModel):
    """A single entry in the uploaded-documents list."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(..., alias="documentId")
    filename: str
    page_count: int = Field(..., alias="pageCount")
    chunk_count: int = Field(..., alias="chunkCount")
    status: str
    created_at: str = Field(..., alias="createdAt")
    # Why a FAILED document failed; None for PROCESSING/READY.
    failure_reason: str | None = Field(default=None, alias="failureReason")
    # What the document says about itself (rag/metadata.py). `title` is always
    # present - it falls back to the first heading, then the filename - so the
    # UI can show it instead of "8f2a-final-v3-FINAL.pdf".
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    # ISO 8601 date carried by the document, distinct from createdAt (when it
    # was uploaded here).
    document_date: str | None = Field(default=None, alias="documentDate")


class DocumentListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    documents: list[DocumentSummary]


class DocumentSearchResult(BaseModel):
    chunk_id: str
    filename: str
    page_number: int
    similarity: float
    text: str
    # Where in the document this chunk came from, so a result can be shown as
    # "report.pdf · p12 · 2. Methods > 2.1 Field Sampling".
    section: str = ""
    title: str = ""
    author: str = ""


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return.")
    # Exact-match metadata filters, applied before ranking. Similarity can't
    # answer "anything by the compliance team" - no embedding encodes an
    # author - so these are the axis the query itself cannot reach.
    author: str | None = Field(default=None, description="Only chunks from documents by this author.")
    title: str | None = Field(default=None, description="Only chunks from the document with this title.")
    document_date: str | None = Field(
        default=None, description="Only chunks from documents carrying this ISO 8601 date."
    )


class DocumentSearchResponse(BaseModel):
    query: str
    results: list[DocumentSearchResult]
