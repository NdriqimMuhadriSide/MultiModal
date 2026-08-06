/**
 * useCamera — owns the camera MediaStream lifecycle.
 *
 * This is the ONLY place in the codebase allowed to call
 * `navigator.mediaDevices.getUserMedia()`, per the Phase 7 architecture
 * rule "do not place browser APIs inside UI components." CameraStream.tsx
 * renders whatever `stream` this hook returns into a <video> element; it
 * never touches getUserMedia directly, and useStreaming.ts (the frame
 * capture / sampling layer) is handed the resulting MediaStream rather
 * than knowing how it was acquired — camera and screen share are
 * interchangeable inputs to that layer.
 *
 * Permission handling: `permissionStatus` tracks the getUserMedia promise
 * lifecycle (idle -> requesting -> granted | denied | error) so the UI can
 * show "Requesting camera access..." vs. a clear "Camera access was
 * denied" message, rather than a generic failure. A `NotAllowedError`
 * (user or OS denied permission) is distinguished from other failures
 * (`NotFoundError` - no camera device, `NotReadableError` - camera in use
 * by another app) so the error message is actually actionable.
 *
 * Cleanup: `stop()` calls `.stop()` on every track in the stream, which is
 * what actually releases the camera (turns off the recording indicator) -
 * merely dropping the MediaStream reference does NOT release the hardware.
 * This is also run automatically on unmount, so navigating away from the
 * streaming page can never leave a camera silently recording.
 */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { MediaPermissionStatus } from "@/types";

function describeGetUserMediaError(err: unknown): string {
  if (err instanceof DOMException) {
    switch (err.name) {
      case "NotAllowedError":
        return "Camera access was denied. Allow camera access in your browser settings and try again.";
      case "NotFoundError":
        return "No camera was found on this device.";
      case "NotReadableError":
        return "The camera is already in use by another application.";
      default:
        return `Could not access the camera (${err.name}).`;
    }
  }
  return "Something went wrong while requesting camera access.";
}

export function useCamera() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [permissionStatus, setPermissionStatus] = useState<MediaPermissionStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStream(null);
    setPermissionStatus("idle");
  }, []);

  const start = useCallback(async () => {
    if (streamRef.current) return; // already running

    setError(null);
    setPermissionStatus("requesting");
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });

      // A user can stop sharing via the OS/browser's own camera indicator
      // (or unplug the device) without ever calling our stop() — listen
      // for that so state stays accurate instead of claiming "live"
      // against a stream that has actually ended.
      mediaStream.getVideoTracks().forEach((track) => {
        track.addEventListener("ended", stop);
      });

      streamRef.current = mediaStream;
      setStream(mediaStream);
      setPermissionStatus("granted");
    } catch (err) {
      const message = describeGetUserMediaError(err);
      setError(message);
      setPermissionStatus(err instanceof DOMException && err.name === "NotAllowedError" ? "denied" : "error");
    }
  }, [stop]);

  // Guarantees the camera is released if the component using this hook
  // unmounts while a stream is active (e.g. the user navigates away
  // instead of clicking "Stop").
  useEffect(() => stop, [stop]);

  return {
    stream,
    permissionStatus,
    error,
    isActive: stream !== null,
    start,
    stop,
  };
}
