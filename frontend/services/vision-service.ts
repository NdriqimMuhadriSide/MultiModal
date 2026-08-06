/**
 * VisionService — talks to POST /api/v1/vision/analyze.
 *
 * Like ChatService, this is the only module that knows this endpoint's
 * shape. The one thing that differs from ChatService: the request body is
 * `multipart/form-data` (an image file part + a question field), not JSON,
 * because binary files can't be embedded in a JSON payload. Building that
 * FormData here — instead of in useVision or the upload component — keeps
 * "how a request is encoded" out of both the hook and the UI.
 */
import { apiClient } from "@/lib/api-client";
import type { VisionAnalyzeResponse } from "@/types";

export const VisionService = {
  /**
   * Uploads an image with an accompanying question and returns the
   * model's analysis. `onUploadProgress` is optional and only meaningful
   * for large files — small images upload near-instantly.
   */
  analyzeImage(
    image: File,
    question: string,
    options?: { signal?: AbortSignal; onUploadProgress?: (percent: number) => void }
  ): Promise<VisionAnalyzeResponse> {
    const formData = new FormData();
    formData.append("image", image);
    formData.append("question", question);

    return apiClient.postForm<VisionAnalyzeResponse>("/vision/analyze", formData, {
      signal: options?.signal,
      onUploadProgress: options?.onUploadProgress,
    });
  },
};
