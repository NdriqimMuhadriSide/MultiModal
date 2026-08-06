/**
 * useAgent — orchestrates sending a message to the routing agent.
 *
 * Same boundary role as useChat/useRag (UI never calls AgentService or
 * mutates the store directly), but replaces the manual "General AI vs My
 * Documents" choice: the backend decides per message whether to search the
 * ingested PDFs, call a live external API, or answer from the model alone.
 *
 * The chosen tool comes back in `tool_used` and is stored on the message as
 * `toolUsed`, so the bubble can show where the answer came from — without
 * that, a routed answer is indistinguishable from a plain one and a
 * misroute (e.g. a documents question answered from general knowledge) is
 * invisible.
 */
"use client";

import { useCallback, useState } from "react";
import { useChatStore } from "@/store/chat-store";
import { AgentService } from "@/services/agent-service";
import { canQueue, enqueue, requestFlush } from "@/lib/outbox";
import { ApiError } from "@/types/api";
import type { Message } from "@/types";

/**
 * True only when the request never reached the server. ApiError carries a
 * null status for exactly that case; anything with a status means the backend
 * answered, and an answer — even a 500 — is not something to retry blindly.
 */
function isNetworkError(err: unknown): boolean {
  return err instanceof ApiError && err.status === null;
}

function createMessage(role: Message["role"], content: string, status?: Message["status"]): Message {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    createdAt: new Date().toISOString(),
    status,
  };
}

function toUserFacingError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === null) {
      return "Backend Offline. Check that the API server is running and try again.";
    }
    if (err.status === 503) {
      return "The assistant isn't configured. Check the backend's API keys and try again.";
    }
    if (err.status >= 500) {
      return "The assistant ran into a problem answering that. Please try again.";
    }
    if (err.status === 429) {
      return "Too many requests right now. Please wait a moment and try again.";
    }
    if (err.status >= 400) {
      return "That message couldn't be processed. Please rephrase and try again.";
    }
  }
  return "Something went wrong while contacting the assistant.";
}

/**
 * How often buffered tokens are committed to the store, in milliseconds.
 * ~12 updates/second: below the threshold where text stops looking
 * continuous, and far below the per-token rate that would thrash
 * localStorage. Kept in sync with hooks/use-chat.ts.
 */
const FLUSH_INTERVAL_MS = 80;

export function useAgent() {
  const activeChat = useChatStore((state) => state.activeChat());
  const appendMessage = useChatStore((state) => state.appendMessage);
  const appendToMessage = useChatStore((state) => state.appendToMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);
  const setConversationId = useChatStore((state) => state.setConversationId);

  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const askAgent = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !activeChat || isSending) return;

      setError(null);

      const chatId = activeChat.id;
      const userMessage = createMessage("user", trimmed, "sent");
      appendMessage(chatId, userMessage);

      const pendingAssistant = createMessage("assistant", "", "sending");
      appendMessage(chatId, pendingAssistant);

      setIsSending(true);

      // Tokens are buffered and flushed on an interval rather than written
      // straight through. Every store write re-serializes all persisted
      // chats into localStorage, so writing per token would mean hundreds of
      // full serializations for a single answer. At this cadence the text
      // still arrives faster than anyone reads it.
      let buffered = "";
      let receivedAny = false;
      const flush = () => {
        if (!buffered) return;
        appendToMessage(chatId, pendingAssistant.id, buffered);
        buffered = "";
      };
      const timer = setInterval(flush, FLUSH_INTERVAL_MS);

      try {
        let streamError = false;

        for await (const event of AgentService.streamAsk(
          trimmed,
          activeChat.conversationId
        )) {
          if (event.type === "start") {
            if (!activeChat.conversationId) {
              setConversationId(chatId, event.conversation_id);
            }
          } else if (event.type === "tool") {
            // Arrives before any text: the bubble can say where the answer
            // is coming from while it's still being written.
            updateMessage(chatId, pendingAssistant.id, {
              status: "streaming",
              toolUsed: event.tool,
            });
          } else if (event.type === "delta") {
            buffered += event.content;
            receivedAny = true;
          } else if (event.type === "error") {
            // A provider failure after the response had already committed as
            // 200. Whatever arrived stays on screen — only the status moves.
            streamError = true;
          }
        }

        flush();
        if (streamError) {
          updateMessage(chatId, pendingAssistant.id, { status: "error" });
          setError("The assistant's response was cut off. Please try again.");
        } else {
          updateMessage(chatId, pendingAssistant.id, { status: "sent" });
        }
      } catch (err) {
        // Offline with a worker to hand it to: park the request instead of
        // failing it. Only for a genuine network error — a 4xx/5xx means the
        // backend answered and replaying it later would change nothing.
        if (isNetworkError(err) && canQueue()) {
          const { url, body } = AgentService.askRequest(trimmed, activeChat.conversationId);
          const queued = await enqueue({
            chatId,
            messageId: pendingAssistant.id,
            url,
            body,
            createdAt: new Date().toISOString(),
          });

          if (queued) {
            updateMessage(chatId, pendingAssistant.id, { status: "queued" });
            void requestFlush();
            // No setError: nothing has gone wrong from the user's side yet,
            // and the bubble already says it's waiting.
            return;
          }
        }

        const message = toUserFacingError(err);
        // Only replace the bubble's text when nothing ever arrived. Once the
        // user has read half an answer, blanking it to show an error destroys
        // what they were reading; the error line below is enough.
        updateMessage(chatId, pendingAssistant.id, {
          ...(receivedAny ? {} : { content: message }),
          status: "error",
        });
        setError(message);
      } finally {
        clearInterval(timer);
        flush();
        setIsSending(false);
      }
    },
    [
      activeChat,
      isSending,
      appendMessage,
      appendToMessage,
      updateMessage,
      setConversationId,
    ]
  );

  return {
    askAgent,
    isSending,
    error,
  };
}
