/**
 * BackLink — the way out of a secondary screen.
 *
 * WHY A LINK AND NOT `router.back()`
 *
 * These are two different things, and only one of them is correct here.
 * `router.back()` is *history* navigation: it returns to wherever the user
 * happened to be, which for anyone arriving from a shared URL, a bookmark or
 * a refresh is either the wrong page or outside the app entirely. What this
 * needs is *up* navigation - "the parent of this screen", which is a fixed
 * property of the route tree rather than of one visitor's history. Android
 * draws exactly this distinction between Up and Back; the web has no built-in
 * Up, so it is a plain link to the parent.
 *
 * Being a real <a> also means it behaves like one: cmd-click opens a new tab,
 * the browser shows the target on hover, and it still works if the page's
 * JavaScript failed to hydrate. A button wired to `router.back()` gives up all
 * three.
 *
 * `href` is a prop rather than hardcoded so a future nested route (say
 * /documents/[id]) can point at its own parent instead of the root.
 */
"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";

interface BackLinkProps {
  /** Where "up" is from this screen. The chat is the app's root. */
  href?: string;
  label?: string;
  className?: string;
}

export function BackLink({
  href = "/",
  label = "Back to chat",
  className,
}: BackLinkProps) {
  return (
    <Link
      href={href}
      className={cn(
        "inline-flex w-fit items-center gap-1.5 rounded-lg py-1 pr-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        className
      )}
    >
      {/* The arrow is decorative: the adjacent text already names the
          destination, so announcing it twice would be noise to a screen
          reader rather than information. */}
      <ArrowLeft aria-hidden className="size-4 shrink-0" />
      {label}
    </Link>
  );
}
