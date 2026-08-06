/**
 * ScreenShare — renders the live screen-share preview.
 *
 * Purely presentational, mirroring CameraStream.tsx exactly: binds
 * whatever `MediaStream` it's given to a <video> element. It never calls
 * getDisplayMedia itself — that lives entirely in
 * hooks/use-screen-share.ts, per the same "no browser APIs in UI
 * components" rule CameraStream.tsx follows.
 */
"use client";

import { useEffect, useRef } from "react";
import { MonitorOff, MonitorUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface ScreenShareProps {
  stream: MediaStream | null;
  className?: string;
}

export function ScreenShare({ stream, className }: ScreenShareProps) {
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
          className="h-full w-full object-contain bg-black"
          aria-label="Live screen share preview"
        />
      ) : (
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <MonitorOff className="size-8" />
          <p className="text-sm">Screen sharing is off</p>
        </div>
      )}
      {stream && (
        <span className="absolute top-2 left-2 flex items-center gap-1 rounded-full bg-background/80 px-2 py-0.5 text-xs font-medium text-foreground">
          <MonitorUp className="size-3" />
          Screen share
        </span>
      )}
    </div>
  );
}
