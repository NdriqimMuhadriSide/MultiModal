/**
 * PdfUploader — drag & drop or click to select a document, then upload it for
 * ingestion into the knowledge base.
 *
 * Purely UI + orchestration boundary, same split as ImageUploader/useVision:
 * this component validates the picked/dropped file (via the shared
 * validatePdfFile rule) and hands the File to `useDocuments.uploadDocument`
 * — it never calls DocumentService itself. Unlike ImageUploader (which is
 * a small inline control lifted into ChatInput), this is a standalone
 * drop-zone component, since PDF ingestion is its own page/panel rather
 * than an attachment on a chat message.
 *
 * This does not answer questions about the uploaded PDF — Phase 4A only
 * prepares the knowledge base. Retrieval/answering is a later phase.
 */
"use client";

import { useCallback, useRef, useState, type DragEvent } from "react";
import { FileText, Loader2, UploadCloud, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { validatePdfFile } from "@/lib/validate-pdf";
import { SUPPORTED_DOCUMENT_EXTENSIONS } from "@/types";
import { cn } from "@/lib/utils";

interface PdfUploaderProps {
  onUpload: (file: File) => void | Promise<void>;
  isUploading: boolean;
  uploadProgress: number;
  disabled?: boolean;
}

export function PdfUploader({
  onUpload,
  isUploading,
  uploadProgress,
  disabled,
}: PdfUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  const acceptFile = useCallback(
    (file: File) => {
      const validation = validatePdfFile(file);
      if (!validation.valid) {
        setLocalError(validation.error);
        return;
      }
      setLocalError(null);
      setSelectedFile(file);
      void onUpload(file);
    },
    [onUpload]
  );

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Allow re-selecting the same file after a remove/error.
    event.target.value = "";
    if (file) acceptFile(file);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDraggingOver(false);
    if (disabled || isUploading) return;

    const file = event.dataTransfer.files?.[0];
    if (file) acceptFile(file);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (disabled || isUploading) return;
    setIsDraggingOver(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDraggingOver(false);
  };

  const handleRemove = () => {
    setSelectedFile(null);
    setLocalError(null);
  };

  const isBusy = disabled || isUploading;

  if (selectedFile) {
    return (
      <div className="flex flex-col gap-2 rounded-xl border border-border bg-muted/50 p-3">
        <div className="flex items-center gap-2">
          {isUploading ? (
            <Loader2 className="size-5 shrink-0 animate-spin text-muted-foreground" />
          ) : (
            <FileText className="size-5 shrink-0 text-muted-foreground" />
          )}
          <span className="min-w-0 flex-1 truncate text-sm">{selectedFile.name}</span>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label="Remove file"
            disabled={isUploading}
            onClick={handleRemove}
          >
            <X className="size-4" />
          </Button>
        </div>
        {isUploading && (
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-all duration-150"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        )}
        {localError && <p className="text-xs text-destructive">{localError}</p>}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div
        role="button"
        tabIndex={0}
        aria-disabled={isBusy}
        onClick={() => !isBusy && inputRef.current?.click()}
        onKeyDown={(event) => {
          if ((event.key === "Enter" || event.key === " ") && !isBusy) {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-border p-8 text-center transition-colors",
          isDraggingOver && "border-primary bg-primary/5",
          isBusy ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:bg-muted/50"
        )}
      >
        <UploadCloud className="size-8 text-muted-foreground" />
        <p className="text-sm font-medium">
          Drag & drop a document here, or click to browse
        </p>
        <p className="text-xs text-muted-foreground">
          PDF, Word, HTML, Markdown, CSV or text &mdash; up to 20MB
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={SUPPORTED_DOCUMENT_EXTENSIONS.join(",")}
        className="hidden"
        onChange={handleInputChange}
        disabled={isBusy}
      />
      {localError && <p className="text-xs text-destructive">{localError}</p>}
    </div>
  );
}
