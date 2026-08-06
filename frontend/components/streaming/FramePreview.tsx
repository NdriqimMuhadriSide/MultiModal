/**
 * FramePreview — shows the most recently captured frame, plus the latest
 * AI observation for it.
 *
 * Purely presentational: renders whatever `previewUrl` (an object URL
 * created by useStreaming.ts from a captured frame Blob) and
 * `observations`/`analysis` it's given. This is deliberately a *separate*
 * still image, not the live <video> preview — it exists to make the
 * sampling behavior visible to the user ("this is the exact frame the AI
 * just looked at"), which the constantly-updating live preview can't show
 * on its own since the analyzed frame is always slightly behind "now."
 */
"use client";

import { ImageOff } from "lucide-react";
import { cn } from "@/lib/utils";

interface FramePreviewProps {
  previewUrl: string | null;
  observations: string[];
  analysis: string;
  className?: string;
}

export function FramePreview({
  previewUrl,
  observations,
  analysis,
  className,
}: FramePreviewProps) {
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex aspect-video w-full items-center justify-center overflow-hidden rounded-xl border border-border bg-muted">
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element -- object URLs aren't compatible with next/image's optimizer
          <img
            src={previewUrl}
            alt="Most recently captured frame"
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-muted-foreground">
            <ImageOff className="size-6" />
            <p className="text-xs">No frame captured yet</p>
          </div>
        )}
      </div>

      <div className="rounded-xl border border-border bg-card p-3">
        <p className="mb-1 text-xs font-medium text-muted-foreground">Latest AI observation</p>
        {analysis ? (
          <p className="text-sm text-foreground">{analysis}</p>
        ) : (
          <p className="text-sm text-muted-foreground">Waiting for the first analyzed frame...</p>
        )}
        {observations.length > 0 && (
          <ul className="mt-2 flex flex-wrap gap-1.5">
            {observations.slice(0, 8).map((observation, index) => (
              <li
                key={`${observation}-${index}`}
                className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground"
              >
                {observation}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
