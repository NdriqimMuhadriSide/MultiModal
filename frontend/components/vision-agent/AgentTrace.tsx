/**
 * AgentTrace — the agent's steps, as they arrive.
 *
 * This is the component that makes the wait tolerable. A run takes several
 * seconds per step and produces no prose until the last one, so without a
 * trace the user watches a spinner; with it they watch the agent decide to
 * read the text, then check a policy, then answer.
 *
 * It stays on screen after the run finishes rather than collapsing into the
 * answer. An answer assembled over four tool calls the user never asked for
 * is not something they can sanity-check otherwise — the trace is how a
 * reader tells a thorough answer from a lucky one.
 *
 * Observations are collapsed by default. A recognition grid is thousands of
 * characters of monospace whitespace: valuable when you want it, ruinous to
 * the page when you don't.
 */
"use client";

import { useState } from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { VISION_TOOL_LABELS, type AgentStep } from "@/types";
import { cn } from "@/lib/utils";

interface AgentTraceProps {
  steps: AgentStep[];
  /** True while the run is still going — draws the pending row. */
  running: boolean;
}

function StepRow({ step, index }: { step: AgentStep; index: number }) {
  const [open, setOpen] = useState(false);
  const label = VISION_TOOL_LABELS[step.tool] ?? step.tool ?? "Thinking";

  return (
    <li className="relative pl-6">
      <span
        aria-hidden
        className="absolute left-[5px] top-2 size-2 rounded-full bg-primary"
      />
      <div className="flex flex-col gap-0.5">
        <div className="flex items-baseline gap-2">
          <span className="text-xs font-mono text-muted-foreground tabular-nums">
            {index + 1}
          </span>
          <span className="text-sm font-medium">{label}</span>
        </div>

        {step.thought ? (
          <p className="text-sm text-muted-foreground">{step.thought}</p>
        ) : null}

        {step.observation ? (
          <>
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              aria-expanded={open}
              className="mt-0.5 flex w-fit items-center gap-1 text-xs text-muted-foreground underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-primary"
            >
              <ChevronDown
                className={cn(
                  "size-3 transition-transform motion-reduce:transition-none",
                  open && "rotate-180"
                )}
              />
              {open ? "Hide" : "Show"} what it found
            </button>
            {open ? (
              <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-muted/50 p-2 font-mono text-xs">
                {step.observation}
              </pre>
            ) : null}
          </>
        ) : null}
      </div>
    </li>
  );
}

export function AgentTrace({ steps, running }: AgentTraceProps) {
  if (steps.length === 0 && !running) return null;

  return (
    <section aria-label="What the agent did" className="rounded-xl border border-border p-3">
      <ol className="relative flex flex-col gap-3">
        {/* The rail behind the dots. Inset so it stops at the last row. */}
        <span
          aria-hidden
          className="absolute bottom-2 left-[9px] top-2 w-px bg-border"
        />
        {steps.map((step, index) => (
          <StepRow key={index} step={step} index={index} />
        ))}

        {running ? (
          <li className="relative pl-6">
            <Loader2
              aria-hidden
              className="absolute left-0 top-0.5 size-3.5 animate-spin text-muted-foreground motion-reduce:animate-none"
            />
            <span className="text-sm text-muted-foreground">Working…</span>
          </li>
        ) : null}
      </ol>
    </section>
  );
}
