/**
 * Pure validation rule for image uploads — no React, no API calls.
 *
 * Used by both ImageUploader (instant feedback the moment a file is
 * picked, before any state changes) and useVision (the authoritative
 * check right before upload, in case a caller bypasses the UI). Keeping
 * the rule in one function means "what's a valid image" is defined once,
 * not duplicated across a component and a hook.
 */
import { MAX_IMAGE_SIZE_BYTES, SUPPORTED_IMAGE_MIME_TYPES } from "@/types";

export type ImageValidationResult =
  | { valid: true }
  | { valid: false; error: string };

export function validateImageFile(file: File): ImageValidationResult {
  if (!SUPPORTED_IMAGE_MIME_TYPES.includes(file.type as never)) {
    return {
      valid: false,
      error: "Unsupported file type. Please upload a PNG, JPEG, or WEBP image.",
    };
  }

  if (file.size > MAX_IMAGE_SIZE_BYTES) {
    const maxMb = Math.round(MAX_IMAGE_SIZE_BYTES / (1024 * 1024));
    return {
      valid: false,
      error: `Image is too large. Maximum size is ${maxMb}MB.`,
    };
  }

  return { valid: true };
}
