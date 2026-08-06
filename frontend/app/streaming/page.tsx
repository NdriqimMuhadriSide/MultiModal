/**
 * /streaming — live camera analysis and screen-share analysis.
 *
 * This is the composing layer for Phase 7: it owns useCamera,
 * useScreenShare, and useStreaming, and wires their state/callbacks into
 * the presentational components (CameraStream/ScreenShare/FramePreview/
 * StreamingControls). Those components never call browser APIs or the
 * streaming service themselves - this page is the one place that decides
 * "which source is active" and hands the resulting MediaStream to
 * useStreaming, which treats camera and screen share identically.
 *
 * Only one source can be active at a time by design (starting the other
 * source while one is live stops the first) - running both concurrently
 * would mean two competing sampling loops sharing one sessionId concept,
 * which the current architecture (one useStreaming instance) doesn't
 * support, and isn't part of the Phase 7 spec.
 */
"use client";

import { useCallback, useEffect, useState } from "react";
import { Header } from "@/components/layout/Header";
import { CameraStream } from "@/components/streaming/CameraStream";
import { ScreenShare } from "@/components/streaming/ScreenShare";
import { FramePreview } from "@/components/streaming/FramePreview";
import { StreamingControls } from "@/components/streaming/StreamingControls";
import { useCamera } from "@/hooks/use-camera";
import { useScreenShare } from "@/hooks/use-screen-share";
import { useStreaming } from "@/hooks/use-streaming";
import type { StreamSource } from "@/types";

export default function StreamingPage() {
  const [activeSource, setActiveSource] = useState<StreamSource | null>(null);

  const camera = useCamera();
  const screenShare = useScreenShare();

  const activeStream = activeSource === "camera" ? camera.stream : activeSource === "screen" ? screenShare.stream : null;
  const streaming = useStreaming(activeStream);

  const stopEverything = useCallback(() => {
    streaming.stop();
    camera.stop();
    screenShare.stop();
    setActiveSource(null);
  }, [streaming, camera, screenShare]);

  const handleStartCamera = useCallback(async () => {
    if (activeSource === "screen") {
      screenShare.stop();
    }
    setActiveSource("camera");
    await camera.start();
  }, [activeSource, camera, screenShare]);

  const handleStartScreenShare = useCallback(async () => {
    if (activeSource === "camera") {
      camera.stop();
    }
    setActiveSource("screen");
    await screenShare.start();
  }, [activeSource, camera, screenShare]);

  // Once permission is granted for the chosen source, start the frame
  // capture/sampling loop against that stream. Effect (not a render-time
  // call) since starting the capture loop is a side effect that must
  // happen after the stream state has actually committed.
  useEffect(() => {
    if (activeStream && !streaming.isStreaming) {
      streaming.start();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- streaming.start/isStreaming intentionally excluded: this effect should only re-run when the underlying stream identity changes, not on every streaming state update.
  }, [activeStream]);

  const permissionError = activeSource === "camera" ? camera.error : activeSource === "screen" ? screenShare.error : null;

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-background">
      <Header />
      <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-6 overflow-y-auto px-4 py-8 sm:px-6">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Live Streaming</h1>
          <p className="text-sm text-muted-foreground">
            Analyze your camera or screen in real time. Frames are sampled
            periodically (not every frame) and analyzed by AI as you go.
          </p>
        </div>

        <StreamingControls
          activeSource={activeSource}
          cameraPermission={camera.permissionStatus}
          screenPermission={screenShare.permissionStatus}
          isStreaming={streaming.isStreaming}
          currentFps={streaming.currentFps}
          framesProcessed={streaming.framesProcessed}
          error={permissionError ?? streaming.error}
          onStartCamera={handleStartCamera}
          onStartScreenShare={handleStartScreenShare}
          onStop={stopEverything}
          onAskQuestion={streaming.askQuestion}
        />

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {activeSource === "screen" ? (
            <ScreenShare stream={screenShare.stream} />
          ) : (
            <CameraStream stream={camera.stream} />
          )}
          <FramePreview
            previewUrl={streaming.previewUrl}
            observations={streaming.latestObservations}
            analysis={streaming.latestAnalysis}
          />
        </div>
      </main>
    </div>
  );
}
