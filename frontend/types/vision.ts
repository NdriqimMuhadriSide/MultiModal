/**
 * Types for the vision domain.
 *
 * Mirrors backend/app/schemas/vision.py. Note there is no request schema
 * to mirror on the wire side — POST /vision/analyze takes
 * multipart/form-data (an `image` file part + a `question` form field),
 * not a JSON body, so there's nothing resembling ChatRequest here. The
 * multipart body is built directly in services/vision-service.ts.
 */

/** POST /api/v1/vision/analyze response body */
export interface VisionAnalyzeResponse {
  answer: string;
}

/** Image MIME types accepted by the uploader and sent to the backend. */
export const SUPPORTED_IMAGE_MIME_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
] as const;

export type SupportedImageMimeType = (typeof SUPPORTED_IMAGE_MIME_TYPES)[number];

/** Max upload size enforced client-side, ahead of the backend's own limit
 * (MAX_UPLOAD_SIZE_MB, 25MB by default — see backend/app/core/config.py).
 * Kept slightly under the backend limit so users get instant feedback
 * instead of uploading a large file only to have the server reject it. */
export const MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024; // 20MB
