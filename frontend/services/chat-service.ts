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
import type {
  ChatRequest,
  ChatResponse,
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
