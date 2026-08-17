/**
 * VisionAgentService — talks to POST /api/v1/vision/ask{,/stream}.
 *
 * Separate from VisionService, which owns /vision/analyze, because the two
 * are genuinely different operations rather than two spellings of one. That
 * one sends an image to the vision model and returns what it says, in a
 * couple of seconds. This one runs a multi-step loop that chooses how to
 * read the image and can consult the document corpus, taking considerably
 * longer and returning a trace alongside the answer.
 *
 * Both stay: "describe this photo" should not pay for a loop.
 */
import { apiClient } from "@/lib/api-client";
import { API_V1_URL } from "@/lib/config";
import { streamSse } from "@/lib/sse";
import type { VisionAgentResponse, VisionAgentStreamEvent } from "@/types";

function buildForm(
  image: File,
  question: string,
  conversationId?: string | null
): FormData {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("question", question);
  // Omitted rather than sent empty for a new conversation: the backend
  // treats an absent field as "generate one", and an empty string would be
  // a conversation id of "".
  if (conversationId) formData.append("conversation_id", conversationId);
  return formData;
}

export const VisionAgentService = {
  /**
   * Non-streaming form. Kept for callers with no page attached to consume a
   * stream — the same reason AgentService keeps `ask` alongside `streamAsk`.
   */
  ask(
    image: File,
    question: string,
    conversationId?: string | null,
    options?: { signal?: AbortSignal; onUploadProgress?: (percent: number) => void }
  ): Promise<VisionAgentResponse> {
    return apiClient.postForm<VisionAgentResponse>(
      "/vision/ask",
      buildForm(image, question, conversationId),
      { signal: options?.signal, onUploadProgress: options?.onUploadProgress }
    );
  },

  /**
   * The streaming form: one frame per completed step, then the answer.
   *
   * This is the one the UI should use. A run takes several seconds per step
   * and produces no prose until the last one, so without frames the user
   * watches a spinner for twenty seconds — worse than the single
   * /vision/analyze call it is meant to improve on.
   *
   * See lib/sse.ts for why this bypasses `apiClient`, and why passing a
   * FormData there matters (the browser must set the multipart boundary).
   */
  streamAsk(
    image: File,
    question: string,
    conversationId?: string | null,
    signal?: AbortSignal
  ): AsyncGenerator<VisionAgentStreamEvent> {
    return streamSse<VisionAgentStreamEvent>(
      `${API_V1_URL}/vision/ask/stream`,
      buildForm(image, question, conversationId),
      signal
    );
  },
};
