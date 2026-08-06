/**
 * useScreenShare — owns the screen/window/tab-sharing MediaStream lifecycle.
 *
 * This is the ONLY place in the codebase allowed to call
 * `navigator.mediaDevices.getDisplayMedia()`, mirroring useCamera.ts's
 * role for the camera. ScreenShare.tsx renders whatever `stream` this hook
 * returns; it never calls getDisplayMedia directly.
 *
 * Unlike getUserMedia, getDisplayMedia always shows the browser's native
 * "choose what to share" picker (screen / window / tab) — there is no way
 * to skip that UI, by design, since screen sharing is a much higher-
 * sensitivity permission than camera access. The user can cancel that
 * picker, which getDisplayMedia surfaces as a `NotAllowedError`, same as
 * a denied camera permission — handled the same way here.
 *
 * Cleanup: the browser also shows its own "Stop sharing" indicator/bar
 * outside our page; clicking it ends the stream without ever calling our
 * `stop()`. The `ended` event listener on the video track (same pattern as
 * useCamera.ts) is what keeps this hook's state accurate when that
 * happens, rather than the UI claiming a share is still live.
 */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { MediaPermissionStatus } from "@/types";

function describeGetDisplayMediaError(err: unknown): string {
  if (err instanceof DOMException) {
    switch (err.name) {
      case "NotAllowedError":
        return "Screen sharing was cancelled or denied.";
      case "NotFoundError":
        return "No screen or window was available to share.";
      default:
        return `Could not start screen sharing (${err.name}).`;
    }
  }
  return "Something went wrong while starting screen sharing.";
}

export function useScreenShare() {
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
      const mediaStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      });

      mediaStream.getVideoTracks().forEach((track) => {
        track.addEventListener("ended", stop);
      });

      streamRef.current = mediaStream;
      setStream(mediaStream);
      setPermissionStatus("granted");
    } catch (err) {
      const message = describeGetDisplayMediaError(err);
      setError(message);
      setPermissionStatus(err instanceof DOMException && err.name === "NotAllowedError" ? "denied" : "error");
    }
  }, [stop]);

  // Guarantees screen sharing stops if the component using this hook
  // unmounts while active (e.g. the user navigates away instead of
  // clicking "Stop").
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
