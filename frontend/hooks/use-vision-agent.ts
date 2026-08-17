/**
 * useVisionAgent — runs an image question through the multi-step agent and
 * exposes its progress as it happens.
 *
 * The difference from useVision is the whole reason this exists. That hook
 * awaits one request and flips `loading` to false when a string arrives.
 * This one has something to show during the wait: the agent's steps land
 * one at a time over several seconds, and rendering them as they arrive is
 * what turns a dead spinner into visible progress.
 *
 * State is deliberately accumulated rather than replaced per frame, because
 * the finished trace is part of the answer — a reader checking whether the
 * agent looked at the right things needs all of it, not just the last step.
 */
"use client";

import { useCallback, useRef, useState } from "react";
import { VisionAgentService } from "@/services/vision-agent-service";
import type {
  AgentStep,
  RAGChatSource,
  StoppedBecause,
  VisionAgentStreamEvent,
} from "@/types";
import { ApiError } from "@/types";

export interface VisionAgentRun {
  steps: AgentStep[];
  answer: string;
  sources: RAGChatSource[];
  unverifiedValues: string[];
  stoppedBecause: StoppedBecause | null;
  conversationId: string | null;
}

const EMPTY_RUN: VisionAgentRun = {
  steps: [],
  answer: "",
  sources: [],
  unverifiedValues: [],
  stoppedBecause: null,
  conversationId: null,
};

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    // A null status is lib/sse.ts's marker for "the request never left" —
    // the same signal the rest of the app reads as the backend being down.
    if (error.status === null) {
      return "Can't reach the backend. Check it's running and try again.";
    }
    if (error.status === 413) {
      return "That image is too large. Try one under 25MB.";
    }
    if (error.status === 400) {
      return "That file type isn't supported. Try a PNG, JPEG, WEBP or GIF.";
    }
  }
  return "The agent ran into a problem reading that image.";
}

export function useVisionAgent() {
  const [run, setRun] = useState<VisionAgentRun>(EMPTY_RUN);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
  }, []);

  const reset = useCallback(() => {
    cancel();
    setRun(EMPTY_RUN);
    setError(null);
  }, [cancel]);

  const ask = useCallback(
    async (image: File, question: string, conversationId?: string | null) => {
      // A second submit while one is in flight abandons the first rather
      // than interleaving two runs into one trace.
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setRun({ ...EMPTY_RUN, conversationId: conversationId ?? null });
      setError(null);
      setRunning(true);

      try {
        const stream = VisionAgentService.streamAsk(
          image,
          question,
          conversationId,
          controller.signal
        );

        for await (const event of stream as AsyncGenerator<VisionAgentStreamEvent>) {
          switch (event.type) {
            case "start":
              setRun((current) => ({
                ...current,
                conversationId: event.conversation_id,
              }));
              break;
            case "step":
              setRun((current) => ({
                ...current,
                steps: [...current.steps, event.step],
              }));
              break;
            case "sources":
              setRun((current) => ({ ...current, sources: event.sources }));
              break;
            case "answer":
              setRun((current) => ({ ...current, answer: event.content }));
              break;
            case "unverified":
              setRun((current) => ({
                ...current,
                unverifiedValues: event.values,
              }));
              break;
            case "done":
              setRun((current) => ({
                ...current,
                stoppedBecause: event.stopped_because,
              }));
              break;
            case "error":
              // A mid-stream failure. The steps already on screen stay —
              // they happened, and they are the most useful thing to show
              // someone whose run died halfway.
              setError(event.detail);
              break;
          }
        }
      } catch (caught) {
        // An abort is the user's own doing, not a failure to report.
        if (controller.signal.aborted) return;
        setError(errorMessage(caught));
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        setRunning(false);
      }
    },
    []
  );

  return { run, running, error, ask, cancel, reset };
}
