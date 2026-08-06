/**
 * RagService — talks to POST /api/v1/rag/chat.
 *
 * Like ChatService/VisionService/DocumentService, this is the only module
 * that knows this endpoint's URL and request/response shape. A plain JSON
 * request (unlike DocumentService's multipart upload), so it's a thin
 * wrapper over apiClient.post.
 */
import { apiClient } from "@/lib/api-client";
import type { RAGChatRequest, RAGChatResponse } from "@/types";

export const RagService = {
  /**
   * Asks a question answered using only the content of previously
   * ingested/embedded PDFs (Phase 4A). The backend embeds the question,
   * searches ChromaDB, builds a context-grounded prompt, and returns the
   * generated answer plus the chunks it was grounded in.
   */
  askQuestion(question: string, signal?: AbortSignal): Promise<RAGChatResponse> {
    const body: RAGChatRequest = { question };
    return apiClient.post<RAGChatResponse, RAGChatRequest>("/rag/chat", body, { signal });
  },
};
