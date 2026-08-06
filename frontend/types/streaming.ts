/**
 * Types for the real-time streaming domain — Phase 7.
 *
 * Mirrors backend/app/schemas/stream.py's StreamFrameResponse exactly
 * (snake_case is not used here because that schema has no aliases, same
 * convention as types/vision.ts / types/audio.ts).
 */

/** POST /api/v1/stream/frame response body */
export interface StreamFrameResponse {
  observations: string[];
  analysis: string;
  /** True if this frame was actually sent for vision analysis (the
   * sampling strategy decided "now" was a sampling point); false if the
   * frame was received but intentionally skipped. */
  sampled: boolean;
}

// ---------------------------------------------------------------------------
// Domain / UI types
// ---------------------------------------------------------------------------

/** Which capture source a streaming session is reading frames from. */
export type StreamSource = "camera" | "screen";

/** Lifecycle of a single streaming session, as tracked by useStreaming. */
export type StreamingStatus = "idle" | "starting" | "live" | "stopping" | "error";

/** Lifecycle of a getUserMedia/getDisplayMedia permission request, as
 * tracked by useCamera/useScreenShare. Distinct from StreamingStatus:
 * permission is about browser/OS access to the device, streaming status
 * is about whether frames are actively being sampled and analyzed. */
export type MediaPermissionStatus = "idle" | "requesting" | "granted" | "denied" | "error";

/** Frame MIME type produced by canvas.toBlob() when capturing a frame —
 * kept in sync with the backend's SUPPORTED_FRAME_MIME_TYPES
 * (processors/streaming/stream_validator.py). */
export const STREAM_FRAME_MIME_TYPE = "image/jpeg" as const;

/** JPEG encoding quality (0-1) used when capturing frames — a lower value
 * keeps frame uploads small and fast without materially harming a vision
 * model's ability to describe scene content, unlike a lossless format
 * which would multiply upload size for no analysis benefit. */
export const STREAM_FRAME_QUALITY = 0.8;
