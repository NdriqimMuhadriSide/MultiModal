/**
 * DocumentList — the ingested knowledge base, with a delete affordance.
 *
 * Presentation-only: renders whatever useDocuments reports (filename,
 * page/chunk counts, status) and holds no fetch logic itself. The one piece
 * of state it does own is which row is awaiting delete confirmation — that
 * is pure interaction state with no meaning outside this component, so
 * lifting it into the hook would only couple them.
 *
 * Deletion is two-step rather than a dialog: the project has no dialog
 * primitive, and `window.confirm` is unstyled and blocks the main thread.
 * An inline confirm keeps the destructive action deliberate while staying
 * in the same visual language as the rest of the row.
 */
"use client";

import { useState } from "react";
import { FileText, Loader2, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { DocumentSummary } from "@/types";
import { cn } from "@/lib/utils";

interface DocumentListProps {
  documents: DocumentSummary[];
  isLoading: boolean;
  onDelete?: (documentId: string) => void;
  /** Id currently being deleted, if any. */
  deletingId?: string | null;
}

export function DocumentList({
  documents,
  isLoading,
  onDelete,
  deletingId = null,
}: DocumentListProps) {
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

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
      {documents.map((doc) => {
        const isDeleting = deletingId === doc.documentId;
        const isConfirming = confirmingId === doc.documentId;

        return (
          <li
            key={doc.documentId}
            className={cn(
              "flex items-center gap-3 rounded-xl border border-border bg-card p-3",
              isConfirming && "border-destructive/40",
              isDeleting && "opacity-60"
            )}
          >
            <FileText className="size-5 shrink-0 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              {/* The document's own title, when it has one — "Annual Safety
                  Review" is what someone is looking for, not the filename it
                  happened to be saved under. The filename still shows below,
                  since it's how the file is found on disk. */}
              <p className="truncate text-sm font-medium">{doc.title || doc.filename}</p>
              {isConfirming ? (
                <p className="text-xs text-destructive">
                  Delete this document and its {doc.chunkCount}{" "}
                  {doc.chunkCount === 1 ? "chunk" : "chunks"}? This can&rsquo;t be undone.
                </p>
              ) : doc.status === "FAILED" && doc.failureReason ? (
                // A failed document has no counts worth showing (they're all
                // zero), and "Failed" on its own leaves the user with nothing
                // to act on — the reason is the useful thing in that space.
                <p className="text-xs text-destructive">{doc.failureReason}</p>
              ) : (
                <p className="truncate text-xs text-muted-foreground">
                  {[
                    doc.title ? doc.filename : null,
                    doc.author,
                    doc.documentDate,
                    `${doc.pageCount} ${doc.pageCount === 1 ? "page" : "pages"}`,
                    `${doc.chunkCount} ${doc.chunkCount === 1 ? "chunk" : "chunks"}`,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </p>
              )}
            </div>

            {isDeleting ? (
              <span
                className="flex items-center gap-2 text-xs text-muted-foreground"
                role="status"
              >
                <Loader2 className="size-4 animate-spin" />
                Deleting
              </span>
            ) : isConfirming ? (
              <div className="flex shrink-0 items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 px-2 text-xs"
                  onClick={() => setConfirmingId(null)}
                >
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  className="h-8 px-3 text-xs"
                  onClick={() => {
                    setConfirmingId(null);
                    onDelete?.(doc.documentId);
                  }}
                >
                  Delete
                </Button>
              </div>
            ) : (
              <>
                <StatusBadge status={doc.status} />
                {onDelete && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-8 shrink-0 text-muted-foreground hover:text-destructive"
                    aria-label={`Delete ${doc.filename}`}
                    onClick={() => setConfirmingId(doc.documentId)}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                )}
              </>
            )}
          </li>
        );
      })}
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
