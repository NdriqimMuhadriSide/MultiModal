/**
 * Pure validation rule for audio uploads — no React, no API calls.
 *
 * Used by both AudioUploader (instant feedback the moment a file is
 * picked/dropped) and useAudio (the authoritative check right before
 * upload, in case a caller bypasses the UI). Mirrors lib/validate-image.ts
 * and lib/validate-pdf.ts for the same reason: "what's a valid upload" is
 * defined once, not duplicated across a component and a hook.
 */
import {
  MAX_AUDIO_SIZE_BYTES,
  SUPPORTED_AUDIO_EXTENSIONS,
  SUPPORTED_AUDIO_MIME_TYPES,
} from "@/types";

export type AudioValidationResult = { valid: true } | { valid: false; error: string };

export function validateAudioFile(file: File): AudioValidationResult {
  const isSupportedMimeType = SUPPORTED_AUDIO_MIME_TYPES.includes(file.type as never);
  const extension = getFileExtension(file.name);
  const isSupportedExtension = SUPPORTED_AUDIO_EXTENSIONS.includes(extension as never);

  // Matches the backend's dual check (audio_validator.py): some
  // browsers/OSes report a generic or empty MIME type for audio picked
  // from certain file managers, so a recognized extension is accepted as
  // a fallback rather than rejecting a legitimate file outright.
  if (!isSupportedMimeType && !isSupportedExtension) {
    return {
      valid: false,
      error: "Unsupported file type. Please upload an MP3, WAV, M4A, or WebM audio file.",
    };
  }

  if (file.size === 0) {
    return { valid: false, error: "That audio file is empty." };
  }

  if (file.size > MAX_AUDIO_SIZE_BYTES) {
    const maxMb = Math.round(MAX_AUDIO_SIZE_BYTES / (1024 * 1024));
    return { valid: false, error: `Audio file is too large. Maximum size is ${maxMb}MB.` };
  }

  return { valid: true };
}

function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot === -1) return "";
  return filename.slice(lastDot).toLowerCase();
}

/** Formats a duration in seconds as "M:SS" (or "H:MM:SS" for longer audio). */
export function formatDuration(seconds: number): string {
  const totalSeconds = Math.round(seconds);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}
