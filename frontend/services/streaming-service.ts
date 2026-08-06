/**
 * StreamingService — talks to POST /api/v1/stream/frame and
 * POST /api/v1/stream/{sessionId}/end.
 *
 * Like AudioService/VisionService, this is the only module that knows
 * these endpoints' shapes. Frame upload is multipart/form-data (an image
 * part + sessionId/question/timestamp fields) — built here, not in
 * useStreaming, keeping "how a request is encoded" out of the hook. Per
 * the Phase 7 rule "do not hardcode API URLs", every call goes through
 * apiClient, which centralizes the base URL (lib/config.ts) exactly like
 * every other service in this codebase.
 */
import { apiClient } from "@/lib/api-client";
import { STREAM_FRAME_MIME_TYPE } from "@/types";
import type { StreamFrameResponse } from "@/types";

export const StreamingService = {
  /**
   * Sends one captured frame for the backend to validate, decide whether
   * to sample, and (if sampled) analyze. Most calls return
   * `sampled: false` — that's the frame sampler doing its job, not an
   * error.
   */
  analyzeFrame(
    frameBlob: Blob,
    sessionId: string,
    question: string | undefined,
    signal?: AbortSignal
  ): Promise<StreamFrameResponse> {
    const formData = new FormData();
    formData.append("frame", frameBlob, `frame.${STREAM_FRAME_MIME_TYPE === "image/jpeg" ? "jpg" : "png"}`);
    formData.append("sessionId", sessionId);
    formData.append("timestamp", new Date().toISOString());
    if (question && question.trim()) {
      formData.append("question", question.trim());
    }

    return apiClient.postForm<StreamFrameResponse>("/stream/frame", formData, { signal });
  },

  /**
   * Ends a streaming session, freeing its server-side context buffer and
   * sampling state immediately rather than waiting for TTL-based cleanup.
   * Fire-and-forget from the caller's perspective — useStreaming calls
   * this on stop but doesn't block the UI on its result.
   */
  endSession(sessionId: string, signal?: AbortSignal): Promise<void> {
    return apiClient.post<void>(`/stream/${sessionId}/end`, undefined, { signal });
  },
};
