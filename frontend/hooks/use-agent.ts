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

export function useAgent() {
  const activeChat = useChatStore((state) => state.activeChat());
  const appendMessage = useChatStore((state) => state.appendMessage);
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
      try {
        const response = await AgentService.ask(trimmed, activeChat.conversationId);

        if (!activeChat.conversationId) {
          setConversationId(chatId, response.conversation_id);
        }

        updateMessage(chatId, pendingAssistant.id, {
          content: response.answer,
          status: "sent",
          toolUsed: response.tool_used,
        });
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
    askAgent,
    isSending,
    error,
  };
}
