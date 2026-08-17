/**
 * useStreaming — captures frames from a MediaStream on a fixed interval
 * and sends each one to the backend for (possible) analysis.
 *
 * This hook is source-agnostic: it takes whatever MediaStream useCamera or
 * useScreenShare produced and treats camera vs. screen the same way — it
 * has no idea which one it's looking at, and doesn't call getUserMedia or
 * getDisplayMedia itself. That's what lets CameraStream.tsx and
 * ScreenShare.tsx both drive the exact same streaming pipeline underneath.
 *
 * Frame capture mechanics: a MediaStream has no direct "give me a still
 * image" API, so this hook renders the stream into an offscreen <video>
 * element, draws that video's current frame onto an offscreen <canvas>,
 * and reads the canvas back out as a JPEG Blob (`canvas.toBlob`). This is
 * the standard, only way to pull a still frame off a live stream in a
 * browser.
 *
 * Client-side capture interval vs. server-side sampling: this hook
 * captures a frame every `captureIntervalMs` (default 1s, deliberately
 * more frequent than the backend's default 2s sampling interval) and
 * posts every captured frame to POST /stream/frame. The *backend* decides
 * whether each posted frame is actually analyzed (`sampled: true`) or
 * dropped (`sampled: false`) — see processors/streaming/frame_sampler.py.
 * Capturing somewhat more often than the server samples means the backend
 * always has a fresh frame available right when its own sampling clock
 * fires, rather than analyzing a frame that's already stale by the
 * client's capture interval. The sampling decision itself is
 * intentionally server-side (not duplicated here) so it stays
 * configurable in one place and survives a page reload mid-session
 * (state lives with the sessionId, not the browser tab).
 *
 * FPS/frame-count tracking: `framesProcessed` counts frames the backend
 * actually analyzed (`sampled: true`), not every frame captured — this is
 * what the spec's "Frames Processed" UI field should reflect. `currentFps`
 * is the client-side capture rate (frames captured per second), shown
 * separately since it's a different, always-available number even before
 * any frame has been sampled server-side.
 */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { StreamingService } from "@/services/streaming-service";
import { newId } from "@/lib/uuid";
import { ApiError } from "@/types/api";
import { STREAM_FRAME_MIME_TYPE, STREAM_FRAME_QUALITY } from "@/types";

const DEFAULT_CAPTURE_INTERVAL_MS = 1000;

function toUserFacingError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === null) {
      return "Backend Offline. Check that the API server is running and try again.";
    }
    if (err.status >= 500) {
      return "The streaming service ran into a problem analyzing a frame.";
    }
    if (err.status >= 400) {
      return "That frame couldn't be processed.";
    }
  }
  return "Something went wrong while streaming.";
}

function captureFrameAsBlob(video: HTMLVideoElement, canvas: HTMLCanvasElement): Promise<Blob | null> {
  const context = canvas.getContext("2d");
  if (!context || video.videoWidth === 0 || video.videoHeight === 0) {
    return Promise.resolve(null);
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), STREAM_FRAME_MIME_TYPE, STREAM_FRAME_QUALITY);
  });
}

export function useStreaming(
  stream: MediaStream | null,
  options?: { captureIntervalMs?: number }
) {
  const captureIntervalMs = options?.captureIntervalMs ?? DEFAULT_CAPTURE_INTERVAL_MS;

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [framesProcessed, setFramesProcessed] = useState(0);
  const [currentFps, setCurrentFps] = useState(0);
  const [latestObservations, setLatestObservations] = useState<string[]>([]);
  const [latestAnalysis, setLatestAnalysis] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pendingQuestionRef = useRef<string | undefined>(undefined);
  const framesCapturedInWindowRef = useRef(0);
  const fpsWindowStartRef = useRef(0);
  const previewUrlRef = useRef<string | null>(null);

  // Lazily creates the offscreen video/canvas elements used for capture -
  // never inserted into the DOM, never rendered; they exist purely as
  // frame-grabbing scratch space, distinct from whatever <video> the UI
  // components render for the user's own live preview.
  const ensureCaptureElements = useCallback(() => {
    if (!videoRef.current) {
      videoRef.current = document.createElement("video");
      videoRef.current.muted = true;
      videoRef.current.playsInline = true;
    }
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }
    return { video: videoRef.current, canvas: canvasRef.current };
  }, []);

  const captureAndSendFrame = useCallback(async () => {
    if (!sessionId) return;
    const { video, canvas } = ensureCaptureElements();
    if (video.readyState < 2) return; // not enough data to grab a frame yet

    const blob = await captureFrameAsBlob(video, canvas);
    if (!blob) return;

    framesCapturedInWindowRef.current += 1;
    const now = performance.now();
    const windowElapsed = now - fpsWindowStartRef.current;
    if (windowElapsed >= 1000) {
      setCurrentFps(
        Math.round((framesCapturedInWindowRef.current / windowElapsed) * 1000 * 10) / 10
      );
      framesCapturedInWindowRef.current = 0;
      fpsWindowStartRef.current = now;
    }

    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const url = URL.createObjectURL(blob);
    previewUrlRef.current = url;
    setPreviewUrl(url);

    try {
      const response = await StreamingService.analyzeFrame(
        blob,
        sessionId,
        pendingQuestionRef.current
      );
      if (response.sampled) {
        setFramesProcessed((count) => count + 1);
        setLatestObservations(response.observations);
        setLatestAnalysis(response.analysis);
      }
    } catch (err) {
      setError(toUserFacingError(err));
    }
  }, [sessionId, ensureCaptureElements]);

  const start = useCallback(() => {
    if (!stream || isStreaming) return;

    setError(null);
    setFramesProcessed(0);
    setCurrentFps(0);
    setLatestObservations([]);
    setLatestAnalysis("");
    framesCapturedInWindowRef.current = 0;
    fpsWindowStartRef.current = performance.now();

    const { video } = ensureCaptureElements();
    video.srcObject = stream;
    void video.play().catch(() => {
      // Autoplay can be blocked in some contexts even for a muted,
      // offscreen video - captureAndSendFrame's readyState guard makes
      // this a soft no-op (no frames captured) rather than a crash.
    });

    const newSessionId = newId();
    setSessionId(newSessionId);
    setIsStreaming(true);

    intervalRef.current = setInterval(() => {
      void captureAndSendFrame();
    }, captureIntervalMs);
  }, [stream, isStreaming, ensureCaptureElements, captureAndSendFrame, captureIntervalMs]);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
      setPreviewUrl(null);
    }
    setIsStreaming(false);

    if (sessionId) {
      // Fire-and-forget: the backend also expires idle sessions via TTL,
      // so a failure here (e.g. the tab is closing) doesn't leak session
      // state indefinitely.
      void StreamingService.endSession(sessionId).catch(() => {});
    }
    setSessionId(null);
  }, [sessionId]);

  const askQuestion = useCallback((question: string) => {
    pendingQuestionRef.current = question.trim() || undefined;
  }, []);

  // Restarting the capture loop's interval whenever captureIntervalMs
  // changes mid-session would be unusual UX (the sampling rate is meant
  // to be a fixed setting, not a live dial) - so this hook intentionally
  // reads captureIntervalMs only at start() time.

  // Guarantees the interval timer and any pending preview object URL are
  // cleaned up if the component using this hook unmounts while streaming.
  useEffect(() => stop, [stop]);

  // If the underlying stream disappears (e.g. useCamera/useScreenShare's
  // stop() was called elsewhere, or the track ended), stop capturing
  // rather than continuing to post frames from a dead stream.
  useEffect(() => {
    if (!stream && isStreaming) {
      stop();
    }
  }, [stream, isStreaming, stop]);

  return {
    isStreaming,
    framesProcessed,
    currentFps,
    latestObservations,
    latestAnalysis,
    previewUrl,
    error,
    start,
    stop,
    askQuestion,
  };
}
