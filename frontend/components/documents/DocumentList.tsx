/**
 * DocumentList — read-only list of previously uploaded/ingested documents.
 *
 * Presentation-only: renders whatever useDocuments reports (filename,
 * page/chunk counts, status) and holds no fetch logic itself. Phase 4A
 * only prepares the knowledge base, so this list has no "ask a question"
 * affordance yet — it exists purely to confirm what has been ingested and
 * with what outcome (READY vs FAILED).
 */
"use client";

import { FileText, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { DocumentSummary } from "@/types";
import { cn } from "@/lib/utils";

interface DocumentListProps {
  documents: DocumentSummary[];
  isLoading: boolean;
}

export function DocumentList({ documents, isLoading }: DocumentListProps) {
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading documents...
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-muted-foreground">
        No documents uploaded yet.
      </p>
    );
  }

  return (
    <ul className="flex flex-col gap-2">
      {documents.map((doc) => (
        <li
          key={doc.documentId}
          className="flex items-center gap-3 rounded-xl border border-border bg-card p-3"
        >
          <FileText className="size-5 shrink-0 text-muted-foreground" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{doc.filename}</p>
            <p className="text-xs text-muted-foreground">
              {doc.pageCount} {doc.pageCount === 1 ? "page" : "pages"} ·{" "}
              {doc.chunkCount} {doc.chunkCount === 1 ? "chunk" : "chunks"}
            </p>
          </div>
          <StatusBadge status={doc.status} />
        </li>
      ))}
    </ul>
  );
}

function StatusBadge({ status }: { status: DocumentSummary["status"] }) {
  return (
    <Badge
      variant="secondary"
      className={cn(
        status === "READY" && "bg-emerald-500/10 text-emerald-600",
        status === "FAILED" && "bg-destructive/10 text-destructive",
        status === "PROCESSING" && "bg-muted text-muted-foreground"
      )}
    >
      {status === "READY" && "Ready"}
      {status === "FAILED" && "Failed"}
      {status === "PROCESSING" && "Processing"}
    </Badge>
  );
}
