/**
 * DocumentService — talks to POST /api/v1/documents/upload and
 * GET /api/v1/documents.
 *
 * Like VisionService, this is the only module that knows these endpoints'
 * shapes. Upload is multipart/form-data (a single file part), so the FormData
 * is built here rather than in useDocuments or PdfUploader — keeping "how
 * a request is encoded" out of both the hook and the UI.
 */
import { apiClient } from "@/lib/api-client";
import type {
  DocumentDeleteResponse,
  DocumentListResponse,
  DocumentUploadResponse,
} from "@/types";

export const DocumentService = {
  /**
   * Uploads a document for ingestion (extract -> chunk -> embed -> store in
   * ChromaDB). `onUploadProgress` is optional and mainly useful for larger
   * documents — small ones upload near-instantly, but ingestion itself
   * (embedding every chunk) can still take a moment after the upload
   * completes, which is why the response only arrives once ingestion
   * finishes, not once the bytes are received.
   */
  uploadDocument(
    file: File,
    options?: { signal?: AbortSignal; onUploadProgress?: (percent: number) => void }
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    return apiClient.postForm<DocumentUploadResponse>("/documents/upload", formData, {
      signal: options?.signal,
      onUploadProgress: options?.onUploadProgress,
    });
  },

  /** Fetches every previously uploaded document, most recent first. */
  listDocuments(signal?: AbortSignal): Promise<DocumentListResponse> {
    return apiClient.get<DocumentListResponse>("/documents", { signal });
  },

  /**
   * Removes a document and every chunk it was split into. Irreversible —
   * the original file is not kept anywhere, so restoring one means
   * re-uploading it.
   *
   * 404s when the document is already gone, which useDocuments treats as
   * success: either way it is no longer in the knowledge base.
   */
  deleteDocument(documentId: string, signal?: AbortSignal): Promise<DocumentDeleteResponse> {
    return apiClient.del<DocumentDeleteResponse>(
      `/documents/${encodeURIComponent(documentId)}`,
      { signal }
    );
  },
};
