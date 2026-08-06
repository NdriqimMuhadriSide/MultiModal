/**
 * useChat — orchestrates sending a message for the active chat.
 *
 * This is the boundary between UI and the API/store: ChatInput and
 * ChatWindow call `sendMessage(text)` and read `isSending` / `error`; they
 * never touch ChatService or useChatStore's mutation methods directly.
 * That keeps business logic (optimistic update, error handling, wiring the
 * returned conversation_id back into the chat) out of components, per the
 * "no business logic in components" requirement.
 */
"use client";

import { useCallback, useState } from "react";
import { useChatStore } from "@/store/chat-store";
import { ChatService } from "@/services/chat-service";
import { ApiError } from "@/types/api";
import type { Message } from "@/types";

/**
 * Maps a raw failure (network error, non-2xx response, or unexpected
 * exception) to copy a user can act on, instead of surfacing raw fetch
 * errors or backend "detail" strings straight into the chat.
 */
function toUserFacingError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === null) {
      // fetch() itself threw — the request never reached the server:
      // backend down, DNS failure, CORS block, offline, etc.
      return "Backend Offline. Check that the API server is running and try again.";
    }
    if (err.status >= 500) {
      return "The assistant service ran into a problem. Please try again.";
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

function createMessage(role: Message["role"], content: string, status?: Message["status"]): Message {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    createdAt: new Date().toISOString(),
    status,
  };
}

/**
 * How often buffered tokens are committed to the store, in milliseconds.
 *
 * ~12 updates/second: below the threshold where text stops looking
 * continuous, and far below the per-token rate that would thrash
 * localStorage. Lower this and the writes multiply for no visible gain.
 */
const FLUSH_INTERVAL_MS = 80;

export function useChat() {
  const activeChat = useChatStore((state) => state.activeChat());
  const appendMessage = useChatStore((state) => state.appendMessage);
  const appendToMessage = useChatStore((state) => state.appendToMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);
  const setConversationId = useChatStore((state) => state.setConversationId);

  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
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
      // still arrives faster than anyone reads it, and the write count drops
      // by an order of magnitude.
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

        for await (const event of ChatService.streamMessage(
          trimmed,
          activeChat.conversationId
        )) {
          if (event.type === "start") {
            if (!activeChat.conversationId) {
              setConversationId(chatId, event.conversation_id);
            }
            // Flipped here rather than at send time: until the stream is
            // actually open there is nothing to show, and "sending" is what
            // keeps the typing indicator up.
            updateMessage(chatId, pendingAssistant.id, { status: "streaming" });
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
        const message = toUserFacingError(err);
        // Only replace the bubble's text when nothing ever arrived. Once
        // the user has read half an answer, blanking it to show an error
        // destroys what they were reading; the error line below is enough.
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
    chat: activeChat,
    messages: activeChat?.messages ?? [],
    sendMessage,
    isSending,
    error,
  };
}
