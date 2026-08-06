/**
 * Types for the audio understanding domain — Phase 5.
 *
 * Mirrors backend/app/schemas/audio.py exactly (snake_case is not used
 * here because that schema has no aliases — field names match as-is,
 * same convention as types/vision.ts).
 */

/** POST /api/v1/audio/analyze response body */
export interface AudioMetadataResponse {
  filename: string;
  duration: number | null;
  size: number;
  sample_rate: number | null;
  channels: number | null;
}

export interface AudioAnalyzeResponse {
  transcript: string;
  analysis: string;
  metadata: AudioMetadataResponse;
}

// ---------------------------------------------------------------------------
// Domain / UI types
// ---------------------------------------------------------------------------

/** Audio MIME types accepted by the uploader and sent to the backend. */
export const SUPPORTED_AUDIO_MIME_TYPES = [
  "audio/mpeg",
  "audio/mp3",
  "audio/wav",
  "audio/x-wav",
  "audio/wave",
  "audio/m4a",
  "audio/x-m4a",
  "audio/mp4",
  "audio/webm",
] as const;

/** Extensions accepted as a fallback when the browser reports a generic/
 * empty MIME type for an audio file (matches the backend's dual MIME +
 * extension check in processors/audio/audio_validator.py). */
export const SUPPORTED_AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".webm"] as const;

/** Max upload size enforced client-side, matching the backend's
 * MAX_AUDIO_SIZE_MB (25MB, the Whisper API's own hard limit — see
 * backend/app/core/config.py). Kept equal (not slightly under, unlike
 * images) since 25MB is a hard external constraint, not a soft one. */
export const MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024; // 25MB
