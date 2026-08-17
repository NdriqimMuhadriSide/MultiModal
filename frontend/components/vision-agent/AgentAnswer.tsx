/**
 * AgentAnswer — the answer, with the two qualifiers that make it checkable.
 *
 * Both exist because this agent's output carries claims a reader cannot
 * verify by looking at it:
 *
 * `unverifiedValues` marks figures character recognition did not confirm.
 * The agent's whole advantage over a single vision call is that quoted
 * numbers were *read* rather than guessed, and a reader has no way to tell
 * which is which unless the UI says. Rendered as a caution rather than a
 * warning: a correctly derived per-head amount and a limit quoted from a
 * policy both land here legitimately.
 *
 * `stoppedBecause` distinguishes a completed run from one synthesised after
 * the step budget ran out. Showing those identically hides that the second
 * was built from partial work.
 */
"use client";

import { AlertTriangle, FileText } from "lucide-react";
import type { RAGChatSource, StoppedBecause } from "@/types";

interface AgentAnswerProps {
  answer: string;
  sources: RAGChatSource[];
  unverifiedValues: string[];
  stoppedBecause: StoppedBecause | null;
}

const STOP_NOTICE: Partial<Record<StoppedBecause, string>> = {
  step_limit:
    "The agent ran out of steps and wrote this from what it had gathered so far.",
  parse_failures:
    "The agent lost its footing partway through and wrote this from what it had gathered so far.",
};

export function AgentAnswer({
  answer,
  sources,
  unverifiedValues,
  stoppedBecause,
}: AgentAnswerProps) {
  if (!answer) return null;

  const notice = stoppedBecause ? STOP_NOTICE[stoppedBecause] : undefined;

  return (
    <section aria-label="Answer" className="flex flex-col gap-3">
      <p className="whitespace-pre-wrap text-sm leading-relaxed">{answer}</p>

      {notice ? (
        <p className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-200">
          {notice}
        </p>
      ) : null}

      {unverifiedValues.length > 0 ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2">
          <p className="flex items-center gap-1.5 text-xs font-medium text-amber-900 dark:text-amber-200">
            <AlertTriangle aria-hidden className="size-3.5" />
            Not confirmed by character recognition
          </p>
          <ul className="mt-1.5 flex flex-wrap gap-1.5">
            {unverifiedValues.map((value) => (
              <li
                key={value}
                className="rounded border border-amber-500/40 bg-background px-1.5 py-0.5 font-mono text-xs tabular-nums"
              >
                {value}
              </li>
            ))}
          </ul>
          <p className="mt-1.5 text-xs text-muted-foreground">
            These were derived, quoted from a document, or read by the vision
            model rather than off the image itself.
          </p>
        </div>
      ) : null}

      {sources.length > 0 ? (
        <div>
          <p className="mb-1.5 text-xs font-medium text-muted-foreground">
            From your documents
          </p>
          <ul className="flex flex-col gap-1">
            {sources.map((source, index) => (
              <li
                key={source.chunkId}
                className="flex items-center gap-1.5 text-xs text-muted-foreground"
              >
                <span className="font-mono text-primary">[E{index + 1}]</span>
                <FileText aria-hidden className="size-3 shrink-0" />
                <span className="truncate">
                  {source.filename} · p{source.page}
                  {source.section ? ` · ${source.section}` : ""}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
