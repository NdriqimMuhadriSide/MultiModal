/**
 * StreamingControls — start/stop buttons, status, and the "ask a
 * question" input for whichever source (camera or screen) is active.
 *
 * Purely presentational + event delegation: every piece of state it
 * displays (permission status, streaming status, fps, frames processed,
 * error) is passed in as props, and every action (start/stop/ask) is a
 * callback prop. It never calls useCamera/useScreenShare/useStreaming
 * itself — the composing page (app/streaming/page.tsx) owns those hooks
 * and wires their state/callbacks into this component. This keeps the
 * control surface reusable regardless of which source is active, and
 * keeps this component trivially testable without mocking browser APIs.
 */
"use client";

import { useState, type KeyboardEvent } from "react";
import { Camera, Loader2, MonitorUp, SendHorizonal, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import type { MediaPermissionStatus, StreamSource } from "@/types";
import { cn } from "@/lib/utils";

interface StreamingControlsProps {
  activeSource: StreamSource | null;
  cameraPermission: MediaPermissionStatus;
  screenPermission: MediaPermissionStatus;
  isStreaming: boolean;
  currentFps: number;
  framesProcessed: number;
  error: string | null;
  onStartCamera: () => void;
  onStartScreenShare: () => void;
  onStop: () => void;
  onAskQuestion: (question: string) => void;
}

export function StreamingControls({
  activeSource,
  cameraPermission,
  screenPermission,
  isStreaming,
  currentFps,
  framesProcessed,
  error,
  onStartCamera,
  onStartScreenShare,
  onStop,
  onAskQuestion,
}: StreamingControlsProps) {
  const [question, setQuestion] = useState("");
  const isBusy = cameraPermission === "requesting" || screenPermission === "requesting";

  const handleAsk = () => {
    if (!question.trim()) return;
    onAskQuestion(question);
    setQuestion("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant={activeSource === "camera" ? "secondary" : "outline"}
          disabled={isBusy || (isStreaming && activeSource !== "camera")}
          onClick={onStartCamera}
        >
          {cameraPermission === "requesting" && activeSource === null ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Camera className="size-4" />
          )}
          {activeSource === "camera" ? "Camera running" : "Start Camera"}
        </Button>

        <Button
          variant={activeSource === "screen" ? "secondary" : "outline"}
          disabled={isBusy || (isStreaming && activeSource !== "screen")}
          onClick={onStartScreenShare}
        >
          {screenPermission === "requesting" && activeSource === null ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <MonitorUp className="size-4" />
          )}
          {activeSource === "screen" ? "Sharing screen" : "Start Screen Share"}
        </Button>

        {isStreaming && (
          <Button variant="destructive" onClick={onStop}>
            <Square className="size-4" />
            Stop
          </Button>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <StatusBadge isStreaming={isStreaming} activeSource={activeSource} />
        <Badge variant="outline">{currentFps.toFixed(1)} fps</Badge>
        <Badge variant="outline">{framesProcessed} frames processed</Badge>
      </div>

      {error && <p className="text-xs text-destructive">{error}</p>}

      {isStreaming && (
        <div className="flex items-end gap-2 rounded-2xl border border-border bg-card p-2 shadow-sm">
          <Textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeSource === "screen"
                ? "Ask about what's on screen (e.g. 'Explain this error message.')"
                : "Ask about what the camera sees (e.g. 'What objects are on my desk?')"
            }
            rows={1}
            className="max-h-32 min-h-10 resize-none border-none px-2 shadow-none focus-visible:ring-0"
          />
          <Button
            size="icon"
            onClick={handleAsk}
            disabled={!question.trim()}
            aria-label="Ask question"
            className="rounded-xl"
          >
            <SendHorizonal className="size-4" />
          </Button>
        </div>
      )}
    </div>
  );
}

function StatusBadge({
  isStreaming,
  activeSource,
}: {
  isStreaming: boolean;
  activeSource: StreamSource | null;
}) {
  if (!isStreaming) {
    return <Badge variant="outline">Idle</Badge>;
  }
  return (
    <Badge className={cn("bg-emerald-500/10 text-emerald-600")}>
      <span className="mr-1 size-1.5 rounded-full bg-emerald-500" />
      Live · {activeSource === "screen" ? "Screen" : "Camera"}
    </Badge>
  );
}
