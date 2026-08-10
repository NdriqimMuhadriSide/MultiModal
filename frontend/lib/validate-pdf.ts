/**
 * Pure validation rule for document uploads — no React, no API calls.
 *
 * Used by both PdfUploader (instant feedback the moment a file is picked
 * or dropped) and useDocuments (the authoritative check right before
 * upload, in case a caller bypasses the UI). Mirrors lib/validate-image.ts
 * for the same reason: "what's a valid upload" is defined once, not
 * duplicated across a component and a hook.
 *
 * Validation is by extension, matching what the backend dispatches on — see
 * SUPPORTED_DOCUMENT_EXTENSIONS. Checking MIME types here would reject files
 * the server reads perfectly well, since browsers report `.md` and `.csv` as
 * text/plain and disagree with each other about `.docx`.
 */
import { MAX_PDF_SIZE_BYTES, SUPPORTED_DOCUMENT_EXTENSIONS } from "@/types";

export type PdfValidationResult = { valid: true } | { valid: false; error: string };

export function validatePdfFile(file: File): PdfValidationResult {
  const name = file.name.toLowerCase();
  const isSupported = SUPPORTED_DOCUMENT_EXTENSIONS.some((extension) =>
    name.endsWith(extension)
  );

  if (!isSupported) {
    return {
      valid: false,
      error: `Unsupported file type. Supported: ${SUPPORTED_DOCUMENT_EXTENSIONS.join(", ")}.`,
    };
  }

  if (file.size === 0) {
    return { valid: false, error: "That file is empty." };
  }

  if (file.size > MAX_PDF_SIZE_BYTES) {
    const maxMb = Math.round(MAX_PDF_SIZE_BYTES / (1024 * 1024));
    return { valid: false, error: `File is too large. Maximum size is ${maxMb}MB.` };
  }

  return { valid: true };
}
