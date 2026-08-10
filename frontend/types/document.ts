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
  /**
   * True when these exact bytes were already ingested, so the existing
   * document was returned rather than a second copy created. The response is
   * otherwise identical to a fresh upload.
   */
  deduplicated: boolean;
  /**
   * Why ingestion failed, when status is FAILED; null otherwise. A FAILED
   * upload still returns 200 — the request was fine, the file just had no
   * usable text — so this is the only signal explaining what happened.
   */
  failureReason: string | null;
}

/** DELETE /api/v1/documents/{documentId} response body */
export interface DocumentDeleteResponse {
  documentId: string;
  filename: string;
  chunksDeleted: number;
}

/** A single entry in GET /api/v1/documents' response body */
export interface DocumentSummary {
  documentId: string;
  filename: string;
  pageCount: number;
  chunkCount: number;
  status: DocumentStatus;
  createdAt: string;
  /** Why ingestion failed, when status is FAILED; null otherwise. */
  failureReason: string | null;
  /**
   * What the document says about itself. `title` is always present — the
   * backend falls back to the first heading, then the filename — so the list
   * can show it instead of `8f2a-final-v3-FINAL.pdf`. The rest are null
   * whenever the file didn't carry them.
   */
  title: string | null;
  author: string | null;
  subject: string | null;
  /**
   * ISO 8601 date carried by the document itself. Distinct from `createdAt`,
   * which is when it was uploaded here — a 2019 report ingested today has two
   * very different dates.
   */
  documentDate: string | null;
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

/**
 * Extensions the backend can ingest — mirrors SUPPORTED_EXTENSIONS in
 * backend/rag/loaders/__init__.py.
 *
 * Extensions rather than MIME types because that is what the backend
 * dispatches on, and for good reason: browsers report `.md` and `.csv` as
 * text/plain or application/octet-stream, and the same `.docx` arrives under
 * several different types depending on the OS. The extension is what the user
 * actually named the file.
 */
export const SUPPORTED_DOCUMENT_EXTENSIONS = [
  ".pdf",
  ".docx",
  ".html",
  ".htm",
  ".md",
  ".markdown",
  ".txt",
  ".csv",
  ".tsv",
] as const;

/** Max upload size enforced client-side, ahead of the backend's own limit
 * (MAX_UPLOAD_SIZE_MB, 25MB by default — see backend/app/core/config.py).
 * Kept slightly under the backend limit so users get instant feedback
 * instead of uploading a large file only to have the server reject it. */
export const MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024; // 20MB
