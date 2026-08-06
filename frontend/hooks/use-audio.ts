/**
 * useAudio — orchestrates uploading an audio file (with an optional
 * question) for transcription + AI analysis.
 *
 * Mirrors the role useVision plays for images and useRag plays for
 * document Q&A: the boundary between UI and the API/store. AudioUploader
 * and ChatLayout never call AudioService or mutate the chat store
 * directly — they call `analyzeAudio(file, question)` and read
 * `isProcessing` / `uploadProgress` / `error`.
 *
 * Staged loading indicator: POST /audio/analyze is a single request/
 * response covering upload + transcription + analysis — the backend
 * doesn't stream "transcription done, now analyzing" as a separate event.
 * To still show "Transcribing..." then "Analyzing..." (per the Phase 5
 * spec), this hook switches the pending message's status from
 * "transcribing" to "analyzing" once the upload itself completes (100%
 * progress) — a reasonable proxy, since the client can observe the upload
 * finishing but not the server-side pipeline stages. If the backend is
 * later changed to stream real stage events (e.g. via SSE), this should
 * be replaced with the actual server-reported stage, same caveat as useRag.
 */
"use client";

import { useCallback, useState } from "react";
import { useChatStore } from "@/store/chat-store";
import { AudioService } from "@/services/audio-service";
import { validateAudioFile } from "@/lib/validate-audio";
import { mediaKey, putMedia } from "@/lib/media-cache";
import { ApiError } from "@/types/api";
import type { Message } from "@/types";

function createMessage(
  role: Message["role"],
  content: string,
  extra?: Partial<Message>
): Message {
  return {
    id: crypto.randomUUID(),
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
      return "That audio file is too large for the server to accept.";
    }
    if (err.status >= 500) {
      return "The audio service ran into a problem transcribing or analyzing that file.";
    }
    if (err.status >= 400) {
      return "That audio file couldn't be processed. Please try a different file.";
    }
  }
  return "Something went wrong while analyzing the audio.";
}

export function useAudio() {
  const activeChat = useChatStore((state) => state.activeChat());
  const appendMessage = useChatStore((state) => state.appendMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);

  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const analyzeAudio = useCallback(
    async (audio: File, question: string) => {
      if (!activeChat || isProcessing) return;

      const validation = validateAudioFile(audio);
      if (!validation.valid) {
        setError(validation.error);
        return;
      }

      setError(null);
      setUploadProgress(0);

      const chatId = activeChat.id;
      const previewUrl = URL.createObjectURL(audio);
      const trimmedQuestion = question.trim();

      // One cached copy of the file, shared by the user bubble and the
      // assistant bubble below — they're two views of the same upload, so
      // caching it twice would double the disk cost for nothing. Derived up
      // front and written fire-and-forget; see the note in use-vision.ts.
      const attachmentId = crypto.randomUUID();
      const cacheKey = mediaKey(attachmentId);
      void putMedia(attachmentId, audio);

      const userMessage = createMessage("user", trimmedQuestion || "Analyze this audio.", {
        status: "sent",
        attachments: [
          {
            id: attachmentId,
            type: "audio",
            url: previewUrl,
            name: audio.name,
            cacheKey,
          },
        ],
      });
      appendMessage(chatId, userMessage);

      const pendingAssistant = createMessage("assistant", "", { status: "transcribing" });
      appendMessage(chatId, pendingAssistant);

      setIsProcessing(true);
      try {
        const response = await AudioService.analyzeAudio(audio, trimmedQuestion, {
          onUploadProgress: (percent) => {
            setUploadProgress(percent);
            if (percent >= 100) {
              updateMessage(chatId, pendingAssistant.id, { status: "analyzing" });
            }
          },
        });

        updateMessage(chatId, pendingAssistant.id, {
          content: response.analysis,
          status: "sent",
          attachments: [
            {
              id: crypto.randomUUID(),
              type: "audio",
              url: previewUrl,
              name: response.metadata.filename,
              transcript: response.transcript,
              audioDuration: response.metadata.duration,
              // Same upload as the user bubble, so it points at the same
              // cache entry rather than storing a second copy.
              cacheKey,
            },
          ],
        });
      } catch (err) {
        const message = toUserFacingError(err);
        updateMessage(chatId, pendingAssistant.id, {
          content: message,
          status: "error",
        });
        setError(message);
      } finally {
        setIsProcessing(false);
      }
    },
    [activeChat, isProcessing, appendMessage, updateMessage]
  );

  return {
    analyzeAudio,
    isProcessing,
    uploadProgress,
    error,
  };
}
