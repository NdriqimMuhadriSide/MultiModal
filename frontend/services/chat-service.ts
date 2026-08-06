/**
 * ChatService — talks to POST /api/v1/chat and GET /api/v1/chat/{id}/history.
 *
 * This is the only module that knows the chat endpoint's URL shape and
 * request/response contract. Components and hooks call these functions
 * and work with plain data — they never construct URLs or call fetch
 * themselves. That separation is what lets the backend contract evolve
 * (e.g. adding model selection or streaming) with changes isolated here.
 */
import { apiClient } from "@/lib/api-client";
import { API_V1_URL } from "@/lib/config";
import { ApiError } from "@/types/api";
import type {
  ChatRequest,
  ChatResponse,
  ChatStreamEvent,
  ConversationHistoryResponse,
} from "@/types";

export const ChatService = {
  /**
   * Sends a user message and returns the assistant's answer.
   * `conversationId` is omitted for the first message in a new chat;
   * the backend assigns one and returns it in the response.
   */
  sendMessage(
    message: string,
    conversationId?: string | null,
    signal?: AbortSignal
  ): Promise<ChatResponse> {
    const body: ChatRequest = {
      message,
      conversation_id: conversationId ?? null,
    };
    return apiClient.post<ChatResponse, ChatRequest>("/chat", body, { signal });
  },

  /**
   * Sends a user message and yields the assistant's answer as it arrives.
   *
   * Deliberately not built on `apiClient`: every method there ends in
   * `response.json()`, which reads the body to completion — the exact thing
   * streaming exists to avoid. This is the one place in the app that calls
   * `fetch` directly, and the reason is structural rather than an oversight.
   *
   * `EventSource` isn't an option either, despite this being SSE: it only
   * issues GET requests with no body, and the message and conversation_id
   * travel in a JSON body. So the framing is parsed by hand below.
   *
   * Failures are normalized to `ApiError` to match every other service, so
   * hooks keep one error-mapping path. Note the two distinct failure modes:
   * a request that never started (offline, 4xx/5xx before any bytes) throws
   * from here, while a provider dying mid-answer arrives as an `error`
   * event — by then the response is already a committed 200.
   */
  async *streamMessage(
    message: string,
    conversationId?: string | null,
    signal?: AbortSignal
  ): AsyncGenerator<ChatStreamEvent> {
    const body: ChatRequest = { message, conversation_id: conversationId ?? null };
    const url = `${API_V1_URL}/chat/stream`;

    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(body),
        signal,
      });
    } catch {
      // fetch() itself threw — the request never reached the server.
      // status: null is what api-client uses for this, and what hooks
      // already read as "Backend Offline".
      throw new ApiError("Network request failed", null, url);
    }

    if (!response.ok || !response.body) {
      throw new ApiError(`Request failed with status ${response.status}`, response.status, url);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        // `stream: true` matters: a chunk boundary can fall inside a
        // multi-byte character, and decoding it standalone would corrupt it.
        buffer += decoder.decode(value, { stream: true });

        // Frames are terminated by a blank line. A partial frame stays in
        // the buffer until the rest of it arrives.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data: ")) continue;
          yield JSON.parse(line.slice("data: ".length)) as ChatStreamEvent;
        }
      }
    } finally {
      // Covers an abort or an early `break` by the consumer: without this
      // the connection stays open and the backend keeps generating tokens
      // for an answer nobody is reading.
      await reader.cancel().catch(() => {});
    }
  },

  /** Fetches the full persisted message history for a conversation. */
  getHistory(
    conversationId: string,
    signal?: AbortSignal
  ): Promise<ConversationHistoryResponse> {
    return apiClient.get<ConversationHistoryResponse>(
      `/chat/${encodeURIComponent(conversationId)}/history`,
      { signal }
    );
  },
};
