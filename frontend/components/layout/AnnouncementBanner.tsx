/**
 * AnnouncementBanner — dismissible strip under the header.
 *
 * A banner rather than a tooltip: a tooltip anchored to a control has to
 * point at something, blocks what's underneath, and has no sensible place to
 * go on a narrow screen. A full-width strip pushes the layout down instead
 * of covering it, and reads the same at every width.
 *
 * Presentation-only — which announcement to show and whether it has already
 * been dismissed live in lib/announcements.ts and store/onboarding-store.ts,
 * reached through useAnnouncement.
 */
"use client";

import { Sparkles, X } from "lucide-react";
import { useAnnouncement } from "@/hooks/use-announcement";
import { Button } from "@/components/ui/button";

export function AnnouncementBanner() {
  const { announcement, dismiss, isHydrated } = useAnnouncement();

  // Nothing until localStorage has been read. Rendering before then would
  // show the banner to everyone who has already dismissed it, for one tick,
  // on every single load. Unlike ThemeToggle there's no placeholder: the
  // steady state for a returning user is no banner at all, so reserving
  // height would leave a permanent empty strip.
  if (!isHydrated || !announcement) return null;

  return (
    <div
      role="status"
      className="flex shrink-0 items-start gap-2.5 border-b border-border bg-muted/50 px-3 py-2.5 sm:px-4"
    >
      <Sparkles className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />

      <div className="min-w-0 flex-1 text-xs sm:text-sm">
        <p className="font-medium">{announcement.title}</p>
        <p className="mt-0.5 text-muted-foreground">{announcement.body}</p>
      </div>

      <Button
        variant="ghost"
        size="icon-sm"
        onClick={dismiss}
        aria-label={`Dismiss: ${announcement.title}`}
        className="-mr-1 shrink-0"
      >
        <X className="size-4" />
      </Button>
    </div>
  );
}
