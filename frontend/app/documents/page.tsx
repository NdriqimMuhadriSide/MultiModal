/**
 * /documents — upload documents and see the ingested knowledge base.
 *
 * Phase 4A scope only: upload -> extract -> chunk -> embed -> store. No
 * question-answering UI here yet (that's a later phase, likely building on
 * POST /rag/ask).
 */
"use client";

import { BackLink } from "@/components/layout/BackLink";
import { Header } from "@/components/layout/Header";
import { PdfUploader } from "@/components/upload/PdfUploader";
import { DocumentList } from "@/components/documents/DocumentList";
import { useDocuments } from "@/hooks/use-documents";

export default function DocumentsPage() {
  const {
    documents,
    isLoading,
    isUploading,
    uploadProgress,
    deletingId,
    error,
    uploadDocument,
    deleteDocument,
  } = useDocuments();

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-background">
      <Header />
      <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-6 overflow-y-auto px-4 py-8 sm:px-6">
        <div>
          <BackLink className="mb-1" />
          <h1 className="text-lg font-semibold tracking-tight">Documents</h1>
          <p className="text-sm text-muted-foreground">
            Upload a document to add it to the knowledge base. It will be split into
            chunks, embedded, and stored for retrieval later.
          </p>
        </div>

        <PdfUploader
          onUpload={uploadDocument}
          isUploading={isUploading}
          uploadProgress={uploadProgress}
        />

        {error && <p className="text-sm text-destructive">{error}</p>}

        <div>
          <h2 className="mb-2 text-sm font-medium text-muted-foreground">
            Uploaded documents
          </h2>
          <DocumentList
            documents={documents}
            isLoading={isLoading}
            onDelete={deleteDocument}
            deletingId={deletingId}
          />
        </div>
      </main>
    </div>
  );
}
