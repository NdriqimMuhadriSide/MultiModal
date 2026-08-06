/**
 * AudioService — talks to POST /api/v1/audio/analyze.
 *
 * Like VisionService, this is the only module that knows this endpoint's
 * shape. The request is multipart/form-data (an `audio` file part + an
 * optional `question` field) — building that FormData here, not in
 * useAudio or AudioUploader, keeps "how a request is encoded" out of both
 * the hook and the UI.
 */
import { apiClient } from "@/lib/api-client";
import type { AudioAnalyzeResponse } from "@/types";

export const AudioService = {
  /**
   * Uploads an audio file (with an optional question) and returns its
   * transcript, AI analysis, and metadata. `onUploadProgress` reports the
   * upload phase only — transcription + analysis happen after the upload
   * completes and aren't separately observable from the client, hence
   * useAudio's staged loading indicator (see useRag's equivalent pattern).
   */
  analyzeAudio(
    audio: File,
    question: string | undefined,
    options?: { signal?: AbortSignal; onUploadProgress?: (percent: number) => void }
  ): Promise<AudioAnalyzeResponse> {
    const formData = new FormData();
    formData.append("audio", audio);
    if (question && question.trim()) {
      formData.append("question", question.trim());
    }

    return apiClient.postForm<AudioAnalyzeResponse>("/audio/analyze", formData, {
      signal: options?.signal,
      onUploadProgress: options?.onUploadProgress,
    });
  },
};
