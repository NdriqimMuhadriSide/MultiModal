/**
 * useRag — orchestrates asking a question answered from ingested PDFs.
 *
 * Mirrors the role useChat plays for plain chat: the boundary between UI
 * and the API/store. ChatInput/ChatLayout call `askQuestion(text)` and
 * read `isAsking` / `error`; they never call RagService or mutate the
 * chat store directly.
 *
 * Staged loading indicator: POST /rag/chat is a single request/response —
 * the backend doesn't stream "search finished, now generating" as separate
 * events. To still show the two-stage "Searching documents..." then
 * "Generating answer..." progression the UX calls for, this hook switches
 * the pending message's status from "searching" to "generating" after a
 * short fixed delay, before the real response arrives. This is a UI-only
 * approximation of pipeline progress, not a signal from the server; if the
 * backend is later changed to stream real stage events (e.g. via SSE),
 * this timer should be replaced with the actual server-reported stage.
 */
"use client";

import { useCallback, useRef, useState } from "react";
import { useChatStore } from "@/store/chat-store";
import { RagService } from "@/services/rag-service";
import { newId } from "@/lib/uuid";
import { ApiError } from "@/types/api";
import type { Message } from "@/types";

// Approximate time the "search ChromaDB for relevant chunks" phase takes,
// before generation begins — long enough to read, short enough not to feel
// stuck if the real response arrives sooner (the message is updated to
// "sent" as soon as it does, cancelling this timer).
const SEARCHING_STAGE_DURATION_MS = 900;

function createMessage(role: Message["role"], content: string, status?: Message["status"]): Message {
  return {
    id: newId(),
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
    if (err.status >= 500) {
      return "The RAG service ran into a problem answering that question.";
    }
    if (err.status >= 400) {
      return "That question couldn't be processed. Please rephrase and try again.";
    }
  }
  return "Something went wrong while searching the documents.";
}

export function useRag() {
  const activeChat = useChatStore((state) => state.activeChat());
  const appendMessage = useChatStore((state) => state.appendMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);

  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stageTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const askQuestion = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || !activeChat || isAsking) return;

      setError(null);

      const chatId = activeChat.id;
      const userMessage = createMessage("user", trimmed, "sent");
      appendMessage(chatId, userMessage);

      const pendingAssistant = createMessage("assistant", "", "searching");
      appendMessage(chatId, pendingAssistant);

      stageTimerRef.current = setTimeout(() => {
        updateMessage(chatId, pendingAssistant.id, { status: "generating" });
      }, SEARCHING_STAGE_DURATION_MS);

      setIsAsking(true);
      try {
        const response = await RagService.askQuestion(trimmed);

        if (stageTimerRef.current) clearTimeout(stageTimerRef.current);
        updateMessage(chatId, pendingAssistant.id, {
          content: response.answer,
          status: "sent",
          sources: response.sources,
        });
      } catch (err) {
        if (stageTimerRef.current) clearTimeout(stageTimerRef.current);
        const message = toUserFacingError(err);
        updateMessage(chatId, pendingAssistant.id, {
          content: message,
          status: "error",
        });
        setError(message);
      } finally {
        setIsAsking(false);
      }
    },
    [activeChat, isAsking, appendMessage, updateMessage]
  );

  return {
    askQuestion,
    isAsking,
    error,
  };
}
