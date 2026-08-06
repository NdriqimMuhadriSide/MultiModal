/**
 * AudioUploader — drag & drop, click to pick, preview, and remove an audio
 * file before sending.
 *
 * Purely UI: validates the picked/dropped file (via the shared
 * validateAudioFile rule) and reports either a File or an error string up
 * to its parent (ChatLayout, via ChatInput) — it never calls AudioService
 * or useAudio itself. Prop shape mirrors ImageUploader's
 * (file/onSelect/onRemove/onError/disabled) so ChatInput can wire it in
 * exactly the same way, but this component additionally supports drag &
 * drop and shows an inline <audio> preview player with duration, since
 * audio benefits from being playable before sending in a way an image
 * thumbnail doesn't need.
 */
"use client";

import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { Music, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { validateAudioFile, formatDuration } from "@/lib/validate-audio";
import { SUPPORTED_AUDIO_MIME_TYPES } from "@/types";
import { cn } from "@/lib/utils";

interface AudioUploaderProps {
  /** Currently selected file, or null if none. Lifted to the parent so
   *  ChatInput/ChatLayout can decide when to clear it (e.g. after a
   *  successful send). */
  file: File | null;
  onSelect: (file: File) => void;
  onRemove: () => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

export function AudioUploader({
  file,
  onSelect,
  onRemove,
  onError,
  disabled,
}: AudioUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [durationSeconds, setDurationSeconds] = useState<number | null>(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  const acceptFile = (selected: File) => {
    const validation = validateAudioFile(selected);
    if (!validation.valid) {
      onError(validation.error);
      return;
    }

    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return URL.createObjectURL(selected);
    });
    setDurationSeconds(null);
    onSelect(selected);
  };

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    // Allow re-selecting the same file after a remove/error.
    event.target.value = "";
    if (selected) acceptFile(selected);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDraggingOver(false);
    if (disabled) return;
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) acceptFile(dropped);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!disabled) setIsDraggingOver(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDraggingOver(false);
  };

  const handleRemove = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setDurationSeconds(null);
    onRemove();
  };

  if (file && previewUrl) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-border bg-muted/50 p-2">
        <Music className="size-5 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="min-w-0 flex-1 truncate text-xs font-medium text-foreground">
              {file.name}
            </span>
            {durationSeconds !== null && (
              <span className="shrink-0 text-xs text-muted-foreground">
                {formatDuration(durationSeconds)}
              </span>
            )}
          </div>
          <audio
            src={previewUrl}
            controls
            className="mt-1 h-8 w-full max-w-xs"
            onLoadedMetadata={(event) => {
              const { duration } = event.currentTarget;
              if (Number.isFinite(duration)) setDurationSeconds(duration);
            }}
          />
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Remove audio"
          disabled={disabled}
          onClick={handleRemove}
        >
          <X className="size-4" />
        </Button>
      </div>
    );
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      aria-label="Attach audio file"
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(event) => {
        if ((event.key === "Enter" || event.key === " ") && !disabled) {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={cn(
        "flex shrink-0 items-center justify-center rounded-lg transition-colors",
        isDraggingOver && "bg-primary/10",
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:bg-muted"
      )}
    >
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-hidden
        tabIndex={-1}
        disabled={disabled}
        className="pointer-events-none"
      >
        <Music className="size-4" />
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept={SUPPORTED_AUDIO_MIME_TYPES.join(",")}
        className="hidden"
        onChange={handleChange}
        disabled={disabled}
      />
    </div>
  );
}
