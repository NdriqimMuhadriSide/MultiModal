/**
 * ThemeToggle — cycles light → dark → system.
 *
 * A single cycling button rather than a dropdown: three options is few
 * enough that clicking through them is faster than opening a menu, and it
 * keeps the header uncluttered. The label spells out where the next click
 * goes, so the cycle isn't something the user has to discover by trial.
 *
 * Presentation-only — the preference and its persistence live in
 * store/theme-store.ts, reached through useTheme.
 */
"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "@/hooks/use-theme";
import { Button } from "@/components/ui/button";
import type { Theme } from "@/lib/theme";

const CYCLE: Record<Theme, Theme> = {
  light: "dark",
  dark: "system",
  system: "light",
};

const ICONS: Record<Theme, typeof Sun> = {
  light: Sun,
  dark: Moon,
  system: Monitor,
};

const LABELS: Record<Theme, string> = {
  light: "Light",
  dark: "Dark",
  system: "System",
};

export function ThemeToggle() {
  const { theme, setTheme, isHydrated } = useTheme();

  // Until localStorage has been read, the store still reports the default
  // ("system"), which may not be what this browser actually chose. Render a
  // same-sized placeholder instead of an icon that's about to change.
  if (!isHydrated) {
    return <div className="size-9 shrink-0" aria-hidden />;
  }

  const Icon = ICONS[theme];
  const next = CYCLE[theme];

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(next)}
      title={`Theme: ${LABELS[theme]} — switch to ${LABELS[next]}`}
      aria-label={`Theme: ${LABELS[theme]}. Switch to ${LABELS[next]}.`}
    >
      <Icon className="size-4" />
    </Button>
  );
}
