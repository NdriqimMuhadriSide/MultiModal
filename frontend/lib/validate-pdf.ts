/**
 * Pure validation rule for PDF uploads — no React, no API calls.
 *
 * Used by both PdfUploader (instant feedback the moment a file is picked
 * or dropped) and useDocuments (the authoritative check right before
 * upload, in case a caller bypasses the UI). Mirrors lib/validate-image.ts
 * for the same reason: "what's a valid upload" is defined once, not
 * duplicated across a component and a hook.
 */
import { MAX_PDF_SIZE_BYTES, SUPPORTED_PDF_MIME_TYPE } from "@/types";

export type PdfValidationResult = { valid: true } | { valid: false; error: string };

export function validatePdfFile(file: File): PdfValidationResult {
  // Some browsers/OSes report an empty MIME type for PDFs picked via drag
  // and drop from certain file managers — fall back to a filename check
  // rather than rejecting a legitimate PDF outright.
  const isPdfByType = file.type === SUPPORTED_PDF_MIME_TYPE;
  const isPdfByExtension = file.name.toLowerCase().endsWith(".pdf");

  if (!isPdfByType && !isPdfByExtension) {
    return { valid: false, error: "Unsupported file type. Please upload a PDF file." };
  }

  if (file.size === 0) {
    return { valid: false, error: "That PDF is empty." };
  }

  if (file.size > MAX_PDF_SIZE_BYTES) {
    const maxMb = Math.round(MAX_PDF_SIZE_BYTES / (1024 * 1024));
    return { valid: false, error: `PDF is too large. Maximum size is ${maxMb}MB.` };
  }

  return { valid: true };
}
