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


class DocumentSummary(BaseModel):
    """A single entry in the uploaded-documents list."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(..., alias="documentId")
    filename: str
    page_count: int = Field(..., alias="pageCount")
    chunk_count: int = Field(..., alias="chunkCount")
    status: str
    created_at: str = Field(..., alias="createdAt")


class DocumentListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    documents: list[DocumentSummary]


class DocumentSearchResult(BaseModel):
    chunk_id: str
    filename: str
    page_number: int
    similarity: float
    text: str


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return.")


class DocumentSearchResponse(BaseModel):
    query: str
    results: list[DocumentSearchResult]
