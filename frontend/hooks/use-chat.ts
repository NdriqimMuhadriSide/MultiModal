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

export function useChat() {
  const activeChat = useChatStore((state) => state.activeChat());
  const appendMessage = useChatStore((state) => state.appendMessage);
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
      try {
        const response = await ChatService.sendMessage(
          trimmed,
          activeChat.conversationId
        );

        if (!activeChat.conversationId) {
          setConversationId(chatId, response.conversation_id);
        }

        updateMessage(chatId, pendingAssistant.id, {
          content: response.answer,
          status: "sent",
        });
      } catch (err) {
        const message = toUserFacingError(err);
        updateMessage(chatId, pendingAssistant.id, {
          content: message,
          status: "error",
        });
        setError(message);
      } finally {
        setIsSending(false);
      }
    },
    [activeChat, isSending, appendMessage, updateMessage, setConversationId]
  );

  return {
    chat: activeChat,
    messages: activeChat?.messages ?? [],
    sendMessage,
    isSending,
    error,
  };
}
