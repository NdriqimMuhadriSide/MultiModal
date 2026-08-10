/**
 * useDocuments — orchestrates uploading a document and listing uploaded documents.
 *
 * This is the boundary between UI and the API for the document ingestion
 * feature: PdfUploader and the documents list component call
 * `uploadDocument(file)` / read `documents`, `isUploading`,
 * `uploadProgress`, and `error` — they never call DocumentService
 * directly. Unlike chat (Zustand store, shared across the sidebar and chat
 * window), the document list doesn't need cross-tree state yet, so plain
 * hook state is enough for Phase 4A; promoting it to a store later (if a
 * second surface needs the same list) wouldn't change this hook's public
 * API.
 *
 * Fetches the document list once on mount and re-fetches after a
 * successful upload, so the list reflects the newly ingested document's
 * final chunk count/status without a manual page refresh.
 */
"use client";

import { useCallback, useEffect, useState } from "react";
import { DocumentService } from "@/services/document-service";
import { validatePdfFile } from "@/lib/validate-pdf";
import { ApiError } from "@/types/api";
import type { DocumentSummary } from "@/types";

function toUserFacingError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === null) {
      return "Backend Offline. Check that the API server is running and try again.";
    }
    if (err.status === 413) {
      return "That file is too large for the server to accept.";
    }
    if (err.status >= 500) {
      return "The document service ran into a problem ingesting that file.";
    }
    if (err.status >= 400) {
      return "That file couldn't be processed. Please try a different document.";
    }
  }
  return "Something went wrong while uploading the document.";
}

function toDeleteError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === null) {
      return "Backend Offline. Check that the API server is running and try again.";
    }
    if (err.status >= 500) {
      return "The document service ran into a problem deleting that document.";
    }
  }
  return "Something went wrong while deleting the document.";
}

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  // The id currently being deleted, so only that row shows a pending state
  // instead of the whole list going busy.
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshDocuments = useCallback(async (signal?: AbortSignal) => {
    setIsLoading(true);
    try {
      const response = await DocumentService.listDocuments(signal);
      setDocuments(response.documents);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      setError(toUserFacingError(err));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshDocuments(controller.signal);
    return () => controller.abort();
  }, [refreshDocuments]);

  const uploadDocument = useCallback(
    async (file: File) => {
      if (isUploading) return;

      const validation = validatePdfFile(file);
      if (!validation.valid) {
        setError(validation.error);
        return;
      }

      setError(null);
      setUploadProgress(0);
      setIsUploading(true);
      try {
        await DocumentService.uploadDocument(file, { onUploadProgress: setUploadProgress });
        // Re-fetch rather than optimistically appending: the upload
        // response only carries {documentId, chunksCreated, status} - the
        // list view also needs pageCount/filename/createdAt, which the
        // registry (not the upload response) is the source of truth for.
        await refreshDocuments();
      } catch (err) {
        setError(toUserFacingError(err));
      } finally {
        setIsUploading(false);
      }
    },
    [isUploading, refreshDocuments]
  );

  const deleteDocument = useCallback(async (documentId: string) => {
    setError(null);
    setDeletingId(documentId);
    try {
      await DocumentService.deleteDocument(documentId);
    } catch (err) {
      // 404 means it is already gone — the desired end state either way, so
      // it falls through to the same local removal rather than showing an
      // error for something the user asked for and got.
      if (!(err instanceof ApiError && err.status === 404)) {
        setError(toDeleteError(err));
        return;
      }
    } finally {
      setDeletingId(null);
    }

    // Removed locally rather than re-fetched: unlike upload, the response
    // tells us exactly which document went, so a round trip would only add
    // latency to a list we can already compute.
    setDocuments((current) => current.filter((doc) => doc.documentId !== documentId));
  }, []);

  return {
    documents,
    isLoading,
    isUploading,
    uploadProgress,
    deletingId,
    error,
    uploadDocument,
    deleteDocument,
    refreshDocuments,
  };
}
