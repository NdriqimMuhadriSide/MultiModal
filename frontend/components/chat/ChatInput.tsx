/**
 * ChatInput — text entry + send button at the bottom of the chat window.
 *
 * Owns only the transient input string (what the user is currently typing).
 * Submission is delegated to the `onSend` callback (backed by useChat/
 * useVision via ChatLayout), so this component has no idea how a message
 * gets to the backend — and, per the Phase 3 architecture requirement, no
 * idea *what kind* of request it becomes either.
 *
 * Image attachment is handled the same way: ChatInput renders
 * <ImageUploader> and forwards its callbacks (onSelectImage/onRemoveImage/
 * onImageError) straight up to whatever the caller passed in. It holds no
 * File state, no validation rule, and never imports VisionService — all of
 * that lives in ImageUploader (validation/preview) and useVision
 * (upload + API call), wired together one level up in ChatLayout.
 *
 * The input string is no longer plain component state: useDraft keeps it in
 * sessionStorage so a reload doesn't discard a half-written message. The chat
 * id arrives as a prop rather than being read from the store, so this
 * component still knows nothing about where chats live.
 */
"use client";

import { type KeyboardEvent } from "react";
import { SendHorizonal } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { ImageUploader } from "@/components/upload/ImageUploader";
import { AudioUploader } from "@/components/upload/AudioUploader";
import { useDraft } from "@/hooks/use-draft";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  /**
   * Which chat the draft belongs to. Null keeps the text purely in memory —
   * used before the chat store has hydrated, when the active id is still a
   * throwaway placeholder that would strand the draft under a dead key.
   */
  chatId: string | null;
  /** Currently attached image, if any (state lives in the parent). */
  attachedImage: File | null;
  onSelectImage: (file: File) => void;
  onRemoveImage: () => void;
  onImageError: (message: string) => void;
  /** 0-100 while an image is uploading; shown as a progress bar. */
  uploadProgress?: number;
  isUploading?: boolean;
  /** Currently attached audio file, if any (state lives in the parent). */
  attachedAudio: File | null;
  onSelectAudio: (file: File) => void;
  onRemoveAudio: () => void;
  onAudioError: (message: string) => void;
  /** Overrides the default placeholder — e.g. RAG document mode in ChatLayout. */
  placeholder?: string;
}

export function ChatInput({
  onSend,
  disabled,
  chatId,
  attachedImage,
  onSelectImage,
  onRemoveImage,
  onImageError,
  uploadProgress = 0,
  isUploading = false,
  attachedAudio,
  onSelectAudio,
  onRemoveAudio,
  onAudioError,
  placeholder,
}: ChatInputProps) {
  const { value, setValue, clear } = useDraft(chatId);
  const hasAttachment = Boolean(attachedImage || attachedAudio);

  const handleSend = () => {
    // With an attachment, an empty question is fine (backend gets a
    // sensible default) — without one, an empty message is a no-op.
    if (!value.trim() && !hasAttachment) return;
    if (disabled) return;
    onSend(value);
    // clear(), not setValue("") — a sent message is no longer a draft, so the
    // stored key goes away rather than lingering as an empty string.
    clear();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts a newline (standard chat UX).
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-border bg-background px-3 py-3 sm:px-6">
      <div className="mx-auto max-w-3xl">
        {attachedImage && (
          <div className="mb-2">
            <ImageUploader
              file={attachedImage}
              onSelect={onSelectImage}
              onRemove={onRemoveImage}
              onError={onImageError}
              disabled={disabled}
            />
            {isUploading && (
              <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full bg-primary transition-all duration-150"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
          </div>
        )}

        {attachedAudio && (
          <div className="mb-2">
            <AudioUploader
              file={attachedAudio}
              onSelect={onSelectAudio}
              onRemove={onRemoveAudio}
              onError={onAudioError}
              disabled={disabled}
            />
            {isUploading && (
              <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full bg-primary transition-all duration-150"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
          </div>
        )}

        <div className="flex items-end gap-2 rounded-2xl border border-border bg-card p-2 shadow-sm">
          {!attachedImage && !attachedAudio && (
            <>
              <ImageUploader
                file={attachedImage}
                onSelect={onSelectImage}
                onRemove={onRemoveImage}
                onError={onImageError}
                disabled={disabled}
              />
              <AudioUploader
                file={attachedAudio}
                onSelect={onSelectAudio}
                onRemove={onRemoveAudio}
                onError={onAudioError}
                disabled={disabled}
              />
            </>
          )}
          <Textarea
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              attachedImage
                ? "Ask a question about this image (optional)..."
                : attachedAudio
                  ? "Ask a question about this audio (optional)..."
                  : placeholder ?? "Message the assistant..."
            }
            disabled={disabled}
            rows={1}
            className="max-h-40 min-h-10 resize-none border-none px-2 shadow-none focus-visible:ring-0"
          />
          <Button
            size="icon"
            onClick={handleSend}
            disabled={disabled || (!value.trim() && !hasAttachment)}
            aria-label="Send message"
            className="rounded-xl"
          >
            <SendHorizonal className="size-4" />
          </Button>
        </div>
      </div>
      <p className="mx-auto mt-1.5 max-w-3xl text-center text-[11px] text-muted-foreground">
        Press Enter to send, Shift+Enter for a new line.
      </p>
    </div>
  );
}
