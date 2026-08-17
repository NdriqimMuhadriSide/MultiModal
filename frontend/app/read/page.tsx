/**
 * /read — ask a question about a document image.
 *
 * Its own route rather than a mode inside the chat, for two reasons. The
 * interaction is one image and one question rather than a conversation, and
 * the output is a trace plus an answer plus caveats — a shape a chat bubble
 * would have to be contorted to hold. The chat's existing image path stays
 * on /vision/analyze, which is still the right call for "describe this
 * photo": one round trip instead of five.
 */
"use client";

import { useState } from "react";
import { Loader2, ScanText } from "lucide-react";
import { AgentAnswer } from "@/components/vision-agent/AgentAnswer";
import { AgentTrace } from "@/components/vision-agent/AgentTrace";
import { BackLink } from "@/components/layout/BackLink";
import { Header } from "@/components/layout/Header";
import { ImageUploader } from "@/components/upload/ImageUploader";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useVisionAgent } from "@/hooks/use-vision-agent";

const EXAMPLES = [
  "What is the total on this receipt?",
  "Does this comply with our expense policy?",
  "What kind of document is this, and is anything missing?",
];

export default function ReadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [pickError, setPickError] = useState<string | null>(null);
  const { run, running, error, ask, reset } = useVisionAgent();

  const canSubmit = Boolean(file) && question.trim().length > 0 && !running;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!file || !canSubmit) return;
    void ask(file, question.trim(), run.conversationId);
  };

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-background">
      {/* This route had no chrome at all: no way back to the chat, no
          connection status, no theme toggle. Landing here from the header
          link left the app with no visible exit, which is the one thing every
          other route already had. */}
      <Header />
      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 overflow-y-auto px-4 py-8 sm:px-6">
        <header className="flex flex-col gap-1">
          <BackLink className="mb-1" />
          <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
            <ScanText aria-hidden className="size-6 text-primary" />
            Read a document
          </h1>
          <p className="text-sm text-muted-foreground">
            The agent decides how to read your image — the vision model for
            layout and diagrams, character recognition for exact numbers — and
            can check what it finds against your ingested documents.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <div className="flex items-start gap-2">
            <ImageUploader
              file={file}
              onSelect={(selected) => {
                setFile(selected);
                setPickError(null);
              }}
              onRemove={() => {
                setFile(null);
                reset();
              }}
              onError={setPickError}
              disabled={running}
            />
          </div>

          {pickError ? (
            <p role="alert" className="text-sm text-destructive">
              {pickError}
            </p>
          ) : null}

          <Textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What do you want to know about it?"
            rows={2}
            disabled={running}
            aria-label="Question about the image"
          />

          <div className="flex flex-wrap items-center gap-2">
            <Button type="submit" disabled={!canSubmit}>
              {running ? (
                <>
                  <Loader2
                    aria-hidden
                    className="size-4 animate-spin motion-reduce:animate-none"
                  />
                  Reading…
                </>
              ) : (
                "Ask"
              )}
            </Button>
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                disabled={running}
                onClick={() => setQuestion(example)}
                className="rounded-full border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50"
              >
                {example}
              </button>
            ))}
          </div>
        </form>

        {error ? (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        ) : null}

        <AgentTrace steps={run.steps} running={running} />

        <AgentAnswer
          answer={run.answer}
          sources={run.sources}
          unverifiedValues={run.unverifiedValues}
          stoppedBecause={run.stoppedBecause}
        />
      </main>
    </div>
  );
}
