/**
 * Types for the RAG (retrieval-augmented generation) domain — Phase 4B.
 *
 * Mirrors backend/app/schemas/rag.py's RAGChatRequest/RAGChatResponse.
 * Like types/document.ts, the wire types here use camelCase (chunkId) to
 * match the backend's Pydantic aliases for this endpoint specifically —
 * ChatRequest/ChatResponse in types/chat.ts remain snake_case because
 * their backend schemas don't declare aliases.
 */

// ---------------------------------------------------------------------------
// Wire types — must match backend/app/schemas/rag.py
// ---------------------------------------------------------------------------

/** POST /api/v1/rag/chat request body */
export interface RAGChatRequest {
  question: string;
}

/** A single citation returned alongside a RAG answer. */
export interface RAGChatSource {
  filename: string;
  page: number;
  chunkId: string;
}

/** POST /api/v1/rag/chat response body */
export interface RAGChatResponse {
  answer: string;
  sources: RAGChatSource[];
}
