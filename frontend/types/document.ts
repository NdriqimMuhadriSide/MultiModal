/**
 * Types for the document ingestion domain (Phase 4A).
 *
 * Mirrors backend/app/schemas/document.py. Note the wire types below use
 * camelCase (documentId, chunksCreated, ...) rather than the snake_case
 * used by types/chat.ts's wire types (conversation_id, ...) - this is
 * intentional and matches the backend: DocumentUploadResponse/
 * DocumentSummary/DocumentListResponse declare camelCase Pydantic aliases
 * specifically for this endpoint, whereas ChatResponse does not. Always
 * check the corresponding Pydantic model before assuming a casing.
 */

/** Status of a document as it moves through the ingestion pipeline. */
export type DocumentStatus = "PROCESSING" | "READY" | "FAILED";

// ---------------------------------------------------------------------------
// Wire types — must match backend/app/schemas/document.py
// ---------------------------------------------------------------------------

/** POST /api/v1/documents/upload response body */
export interface DocumentUploadResponse {
  documentId: string;
  chunksCreated: number;
  status: DocumentStatus;
}

/** A single entry in GET /api/v1/documents' response body */
export interface DocumentSummary {
  documentId: string;
  filename: string;
  pageCount: number;
  chunkCount: number;
  status: DocumentStatus;
  createdAt: string;
}

/** GET /api/v1/documents response body */
export interface DocumentListResponse {
  documents: DocumentSummary[];
}

// ---------------------------------------------------------------------------
// Domain / UI types — used inside the frontend only
// ---------------------------------------------------------------------------

/** PDF MIME type accepted by the uploader and sent to the backend. */
export const SUPPORTED_PDF_MIME_TYPE = "application/pdf" as const;

/** Max upload size enforced client-side, ahead of the backend's own limit
 * (MAX_UPLOAD_SIZE_MB, 25MB by default — see backend/app/core/config.py).
 * Kept slightly under the backend limit so users get instant feedback
 * instead of uploading a large file only to have the server reject it. */
export const MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024; // 20MB
