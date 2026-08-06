/**
 * DocumentService — talks to POST /api/v1/documents/upload and
 * GET /api/v1/documents.
 *
 * Like VisionService, this is the only module that knows these endpoints'
 * shapes. Upload is multipart/form-data (a PDF file part), so the FormData
 * is built here rather than in useDocuments or PdfUploader — keeping "how
 * a request is encoded" out of both the hook and the UI.
 */
import { apiClient } from "@/lib/api-client";
import type { DocumentListResponse, DocumentUploadResponse } from "@/types";

export const DocumentService = {
  /**
   * Uploads a PDF for ingestion (extract -> chunk -> embed -> store in
   * ChromaDB). `onUploadProgress` is optional and mainly useful for larger
   * PDFs — small documents upload near-instantly, but ingestion itself
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
};
