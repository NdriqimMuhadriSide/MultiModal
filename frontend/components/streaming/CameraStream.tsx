/**
 * CameraStream — renders the live camera preview.
 *
 * Purely presentational: binds whatever `MediaStream` it's given to a
 * <video> element for the user's own live preview. It never calls
 * getUserMedia itself — that lives entirely in hooks/use-camera.ts, per
 * the Phase 7 architecture rule "do not place browser APIs inside UI
 * components." This component would render identically if handed a
 * screen-share stream instead of a camera stream; the distinction between
 * the two lives in which hook the parent page chose to call, not in this
 * component.
 */
"use client";

import { useEffect, useRef } from "react";
import { Video, VideoOff } from "lucide-react";
import { cn } from "@/lib/utils";

interface CameraStreamProps {
  stream: MediaStream | null;
  className?: string;
}

export function CameraStream({ stream, className }: CameraStreamProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = stream;
    if (stream) void video.play().catch(() => {});
  }, [stream]);

  return (
    <div
      className={cn(
        "relative flex aspect-video w-full items-center justify-center overflow-hidden rounded-xl bg-muted",
        className
      )}
    >
      {stream ? (
        <video
          ref={videoRef}
          muted
          playsInline
          className="h-full w-full object-cover"
          aria-label="Live camera preview"
        />
      ) : (
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <VideoOff className="size-8" />
          <p className="text-sm">Camera is off</p>
        </div>
      )}
      {stream && (
        <span className="absolute top-2 left-2 flex items-center gap-1 rounded-full bg-background/80 px-2 py-0.5 text-xs font-medium text-foreground">
          <Video className="size-3" />
          Camera
        </span>
      )}
    </div>
  );
}
