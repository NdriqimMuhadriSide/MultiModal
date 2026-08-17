/**
 * Header — top bar showing the app name, a link to /read, and live backend
 * connection status as a single coloured dot.
 *
 * Presentation-only: it renders whatever `useHealthCheck` reports and holds
 * no fetch/timer logic itself, keeping the "no business logic in
 * components" boundary intact.
 *
 * /read is linked here as well as from the sidebar's Library section
 * because the sidebar is `hidden md:flex` — below that breakpoint the only
 * way to it is through the hamburger, which is a discoverability problem
 * for a feature that isn't a conversation. The header is the one piece of
 * chrome present at every width and on every route that renders it.
 */
"use client";

import Link from "next/link";
import { Menu, ScanText } from "lucide-react";
import { useHealthCheck } from "@/hooks/use-health-check";
import { ThemeToggle } from "./ThemeToggle";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/**
 * The wording that used to sit next to the dot. Kept as real text rather than
 * dropped, because it is still what the tooltip and the screen reader use -
 * only its position on screen changed.
 */
const STATUS_LABEL = {
  checking: "Checking backend…",
  online: "Backend connected",
  offline: "Backend offline",
} as const;

interface HeaderProps {
  onToggleSidebar?: () => void;
}

export function Header({ onToggleSidebar }: HeaderProps) {
  const { status } = useHealthCheck();

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-background px-3 sm:px-4">
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          aria-label="Toggle sidebar"
          onClick={onToggleSidebar}
        >
          <Menu className="size-5" />
        </Button>
        <h1 className="text-sm font-semibold tracking-tight sm:text-base">
          Multimodal AI Workspace
        </h1>

        {/* The label collapses to the icon on the narrowest screens, where
            the title and the status pill already compete for the row. The
            aria-label carries the name either way. */}
        <Link
          href="/read"
          aria-label="Read a document"
          className="ml-1 flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <ScanText aria-hidden className="size-4 shrink-0" />
          <span className="hidden sm:inline">Read a document</span>
        </Link>
      </div>

      <div className="flex items-center gap-2 text-xs font-medium">
        {/* The dot alone, with the wording moved to a tooltip and to
            screen-reader-only text.
            The label cannot simply be deleted along with the visible string.
            A bare coloured circle is meaningless to anyone using a screen
            reader, and to anyone who cannot distinguish the red from the
            green - which is the single most common form of colour blindness,
            and this is a control whose entire content IS a colour. So the dot
            stops being `aria-hidden` and carries the status itself:
            `role="status"` announces changes as they happen, `title` gives
            sighted users the wording on hover, and the sr-only span is what
            actually gets read out. */}
        <span
          role="status"
          aria-label={STATUS_LABEL[status]}
          title={STATUS_LABEL[status]}
          className={cn(
            "size-2 rounded-full",
            status === "online" && "bg-emerald-500",
            status === "offline" && "bg-red-500",
            status === "checking" && "animate-pulse bg-muted-foreground"
          )}
        >
          <span className="sr-only">{STATUS_LABEL[status]}</span>
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
