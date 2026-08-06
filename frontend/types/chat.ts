/**
 * Types for the chat domain.
 *
 * These are split into two groups:
 *  - "Wire" types (ChatRequest, ChatResponse, ...) mirror the backend
 *    Pydantic schemas in `app/schemas/chat.py` exactly. They describe what
 *    goes over HTTP.
 *  - "Domain/UI" types (Message, Chat) describe how the frontend models a
 *    conversation internally (e.g. for rendering and for the store). They
 *    are intentionally decoupled from the wire format so backend changes
 *    don't ripple into every component that renders a message.
 */

/** Role of a message participant. Mirrors backend ConversationMessageResponse.role values. */
import type { AgentTool } from "./agent";
import type { RAGChatSource } from "./rag";

export type MessageRole = "user" | "assistant" | "system";

// ---------------------------------------------------------------------------
// Wire types — must match backend/app/schemas/chat.py
// ---------------------------------------------------------------------------

/** POST /api/v1/chat request body */
export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
}

/** POST /api/v1/chat response body */
export interface ChatResponse {
  conversation_id: string;
  answer: string;
}

/** A single stored message as returned by GET /api/v1/chat/{id}/history */
export interface ConversationMessageResponse {
  role: MessageRole;
  content: string;
  created_at: string;
}

/** GET /api/v1/chat/{id}/history response body */
export interface ConversationHistoryResponse {
  conversation_id: string;
  messages: ConversationMessageResponse[];
}

// ---------------------------------------------------------------------------
// Domain / UI types — used inside the frontend only
// ---------------------------------------------------------------------------

/**
 * A file attached to a message for the model to analyze (Phase 3: images;
 * Phase 5: audio). `url` is a client-side object URL
 * (`URL.createObjectURL`) used to render a thumbnail/player in the chat —
 * it is never sent to the backend; the raw file itself goes over
 * multipart/form-data via VisionService/AudioService. Kept as its own type
 * (rather than folding fields into Message) so a future attachment kind
 * (video) is additive, not a breaking change to every place that reads
 * Message.
 *
 * `transcript`/`audioDuration` are only populated for `type: "audio"` —
 * the transcript is shown in the message bubble alongside the AI's
 * analysis (which is just the message's own `content`), per the Phase 5
 * spec's "Display: 🎵 filename.mp3 / Transcript / AI Response".
 */
export interface Attachment {
  id: string;
  type: "image" | "audio";
  /**
   * Client-side object URL, valid only for the page session that created it.
   * The store blanks this before persisting to localStorage, so an empty
   * string means "no live URL" rather than "no media" — `cacheKey` below is
   * the durable route back to the bytes. The text fields (name, transcript,
   * audioDuration) survive a reload on their own.
   */
  url: string;
  /**
   * Cache API key holding the original file, so the attachment can be
   * rebuilt after a reload once `url` has gone stale. Undefined when the
   * write failed or the browser has no Cache API; a key can also miss if the
   * entry was evicted, so a lookup returning nothing is normal and falls back
   * to the "media expired" placeholder. See lib/media-cache.ts.
   */
  cacheKey?: string;
  /** Original filename, shown in the UI alongside the thumbnail/player. */
  name?: string;
  /** Audio only: the Whisper transcript, shown above the AI's analysis. */
  transcript?: string;
  /** Audio only: duration in seconds, if the backend could determine it. */
  audioDuration?: number | null;
}

/**
 * A single chat message as rendered in the UI.
 *
 * `id` is a client-generated identifier (used as React key + for targeting
 * a specific bubble, e.g. to show a per-message error or loading state).
 * `status` lets a just-sent user message render immediately, then be
 * reconciled once the backend responds. `status` also carries the RAG
 * loading stages ("searching" / "generating", see useRag) and the audio
 * loading stages ("transcribing" / "analyzing", see useAudio) so the
 * bubble can show pipeline-specific progress instead of one generic spinner.
 * `attachments` is populated for image/audio-analysis turns (see
 * useVision/useAudio); `sources` is populated for RAG answers (see
 * useRag) — all are omitted for plain text-only messages.
 * `toolUsed` is populated for answers that came through the agent (see
 * useAgent) and records which tool its router picked, so the UI can show
 * where an answer actually came from.
 */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  status?:
    | "sending"
    | "searching"
    | "generating"
    | "transcribing"
    | "analyzing"
    /**
     * Written but not yet delivered: the send failed with no network and the
     * request was handed to the service worker's outbox to retry (see
     * lib/outbox.ts). Deliberately *not* one of the in-flight statuses in
     * store/chat-store.ts — those are settled to an error on reload because
     * the request that would have resolved them died with the page, whereas
     * a queued request outlives the page in the Cache API and is still going
     * to be sent.
     */
    | "queued"
    | "sent"
    | "error";
  attachments?: Attachment[];
  sources?: RAGChatSource[];
  toolUsed?: AgentTool;
}

/**
 * A chat conversation/thread. Maps 1:1 to a backend conversation_id once one
 * has been assigned by the server on the first response.
 */
export interface Chat {
  id: string;
  conversationId: string | null;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}
