/**
 * Header — top bar showing the app name, a link to /read, and live backend
 * connection status.
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
        <span
          className={cn(
            "size-2 rounded-full",
            status === "online" && "bg-emerald-500",
            status === "offline" && "bg-red-500",
            status === "checking" && "animate-pulse bg-muted-foreground"
          )}
          aria-hidden
        />
        <span
          className={cn(
            "text-muted-foreground",
            status === "online" && "text-emerald-600",
            status === "offline" && "text-red-600"
          )}
        >
          {status === "checking" && "Checking backend..."}
          {status === "online" && "Backend Connected"}
          {status === "offline" && "Backend Offline"}
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
