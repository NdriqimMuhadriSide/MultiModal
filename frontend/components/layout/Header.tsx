/**
 * Header — top bar showing the app name and live backend connection status.
 *
 * Presentation-only: it renders whatever `useHealthCheck` reports and holds
 * no fetch/timer logic itself, keeping the "no business logic in
 * components" boundary intact.
 */
"use client";

import { Menu } from "lucide-react";
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
