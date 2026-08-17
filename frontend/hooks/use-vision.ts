/**
 * useVision — orchestrates sending an image + question for analysis.
 *
 * Mirrors the role useChat plays for text: it's the boundary between UI
 * and the API/store. ImageUploader and ChatInput never call VisionService
 * or mutate the chat store directly — they call `analyzeImage(file,
 * question)` and read `isUploading` / `uploadProgress` / `error`. This is
 * what "do not put image logic inside ChatInput" means in practice: this
 * hook, not the input component, owns the request lifecycle.
 *
 * Design choice: image messages are appended to the *same* chat/message
 * list as text messages (via the shared chat store), because in the UI
 * they're just another turn in one conversation. Vision analysis is
 * currently a single-shot call (no conversation_id/history, matching the
 * backend's stateless /vision/analyze endpoint) — unlike chat, follow-up
 * questions about the same image are not yet threaded server-side.
 */
"use client";

import { useCallback, useState } from "react";
import { useChatStore } from "@/store/chat-store";
import { VisionService } from "@/services/vision-service";
import { validateImageFile } from "@/lib/validate-image";
import { mediaKey, putMedia } from "@/lib/media-cache";
import { newId } from "@/lib/uuid";
import { ApiError } from "@/types/api";
import type { Message } from "@/types";

function createMessage(
  role: Message["role"],
  content: string,
  extra?: Partial<Message>
): Message {
  return {
    id: newId(),
    role,
    content,
    createdAt: new Date().toISOString(),
    ...extra,
  };
}

function toUserFacingError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === null) {
      return "Backend Offline. Check that the API server is running and try again.";
    }
    if (err.status === 413) {
      return "That image is too large for the server to accept.";
    }
    if (err.status >= 500) {
      return "The vision service ran into a problem analyzing that image.";
    }
    if (err.status >= 400) {
      return "That image or question couldn't be processed. Please try again.";
    }
  }
  return "Something went wrong while analyzing the image.";
}

export function useVision() {
  const activeChat = useChatStore((state) => state.activeChat());
  const appendMessage = useChatStore((state) => state.appendMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const analyzeImage = useCallback(
    async (image: File, question: string) => {
      if (!activeChat || isUploading) return;

      const validation = validateImageFile(image);
      if (!validation.valid) {
        setError(validation.error);
        return;
      }

      setError(null);
      setUploadProgress(0);

      const chatId = activeChat.id;
      const previewUrl = URL.createObjectURL(image);
      const trimmedQuestion = question.trim();

      const effectiveQuestion =
        trimmedQuestion || "Describe this image and point out anything notable.";

      // The key is derived up front so the message can be appended
      // immediately — the user's own bubble should not wait on a disk write.
      // The write is fire-and-forget for the same reason: if it fails, the
      // key resolves to nothing after a reload, which is exactly the expired
      // placeholder this attachment would have shown anyway.
      const attachmentId = newId();
      void putMedia(attachmentId, image);

      const userMessage = createMessage("user", effectiveQuestion, {
        status: "sent",
        attachments: [
          {
            id: attachmentId,
            type: "image",
            url: previewUrl,
            name: image.name,
            cacheKey: mediaKey(attachmentId),
          },
        ],
      });
      appendMessage(chatId, userMessage);

      const pendingAssistant = createMessage("assistant", "", { status: "sending" });
      appendMessage(chatId, pendingAssistant);

      setIsUploading(true);
      try {
        const response = await VisionService.analyzeImage(image, effectiveQuestion, {
          onUploadProgress: setUploadProgress,
        });

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
        setIsUploading(false);
      }
    },
    [activeChat, isUploading, appendMessage, updateMessage]
  );

  return {
    analyzeImage,
    isUploading,
    uploadProgress,
    error,
  };
}
