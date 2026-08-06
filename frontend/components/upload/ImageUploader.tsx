/**
 * ImageUploader — select, preview, and remove an image before sending.
 *
 * Purely UI: it validates the picked file (via the shared validateImageFile
 * rule) and reports either a File or an error string up to its parent. It
 * never calls VisionService or useVision itself — the composing component
 * (ChatInput's parent, ChatLayout) decides what happens with the selected
 * file. This keeps the uploader reusable anywhere an image needs to be
 * picked, independent of "what happens when you hit send".
 */
"use client";

import { useRef, useState, type ChangeEvent } from "react";
import { ImagePlus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { validateImageFile } from "@/lib/validate-image";
import { SUPPORTED_IMAGE_MIME_TYPES } from "@/types";
import { cn } from "@/lib/utils";

interface ImageUploaderProps {
  /** Currently selected file, or null if none. Lifted to the parent so
   *  ChatInput/ChatLayout can decide when to clear it (e.g. after a
   *  successful send). */
  file: File | null;
  onSelect: (file: File) => void;
  onRemove: () => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

export function ImageUploader({
  file,
  onSelect,
  onRemove,
  onError,
  disabled,
}: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    // Allow re-selecting the same file after a remove/error.
    event.target.value = "";
    if (!selected) return;

    const validation = validateImageFile(selected);
    if (!validation.valid) {
      onError(validation.error);
      return;
    }

    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return URL.createObjectURL(selected);
    });
    onSelect(selected);
  };

  const handleRemove = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    onRemove();
  };

  if (file && previewUrl) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-border bg-muted/50 p-2">
        {/* eslint-disable-next-line @next/next/no-img-element -- object URLs aren't compatible with next/image's optimizer */}
        <img
          src={previewUrl}
          alt={`Preview of ${file.name}`}
          className="size-12 shrink-0 rounded-lg object-cover"
        />
        <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
          {file.name}
        </span>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Remove image"
          disabled={disabled}
          onClick={handleRemove}
        >
          <X className="size-4" />
        </Button>
      </div>
    );
  }

  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-label="Attach image"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        className={cn("shrink-0")}
      >
        <ImagePlus className="size-4" />
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept={SUPPORTED_IMAGE_MIME_TYPES.join(",")}
        className="hidden"
        onChange={handleChange}
        disabled={disabled}
      />
    </>
  );
}
