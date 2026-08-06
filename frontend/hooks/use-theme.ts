/**
 * useTheme — reads the stored theme preference and exposes a setter.
 *
 * Two effects, each covering a case the other doesn't:
 *
 *   1. Rehydration, for the same reason as the chat store — localStorage is
 *      read after mount so server and client render identical markup.
 *   2. An OS listener, active only while the preference is "system". Without
 *      it, "system" would mean "whatever the OS said when the page loaded"
 *      and a machine switching to dark at sunset would leave the app light
 *      until the next refresh.
 *
 * `resolvedTheme` is what's on screen right now; `theme` is what the user
 * picked. They differ whenever the preference is "system".
 */
"use client";

import { useCallback, useEffect, useState } from "react";
import { useThemeStore } from "@/store/theme-store";
import {
  applyTheme,
  DARK_MEDIA_QUERY,
  resolveTheme,
  type ResolvedTheme,
  type Theme,
} from "@/lib/theme";

interface UseThemeResult {
  theme: Theme;
  resolvedTheme: ResolvedTheme | null;
  setTheme: (theme: Theme) => void;
  /** False until localStorage has been read — the toggle uses it to avoid
   *  rendering a preference it doesn't know yet. */
  isHydrated: boolean;
}

export function useTheme(): UseThemeResult {
  const theme = useThemeStore((state) => state.theme);
  const setTheme = useThemeStore((state) => state.setTheme);

  const [isHydrated, setIsHydrated] = useState(false);
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme | null>(null);

  useEffect(() => {
    const unsubscribe = useThemeStore.persist.onFinishHydration(() =>
      setIsHydrated(true)
    );

    if (useThemeStore.persist.hasHydrated()) {
      setIsHydrated(true);
    } else {
      void useThemeStore.persist.rehydrate();
    }

    return unsubscribe;
  }, []);

  // Track what's actually displayed, including OS changes while on "system".
  useEffect(() => {
    const media = window.matchMedia(DARK_MEDIA_QUERY);

    const sync = () => setResolvedTheme(resolveTheme(theme));
    sync();

    if (theme !== "system") return;

    const onSystemChange = () => {
      // applyTheme rather than touching the DOM directly: it's the one place
      // that also refreshes the appearance cookie, so the next server render
      // agrees with what the OS just switched to.
      applyTheme(theme);
      sync();
    };

    media.addEventListener("change", onSystemChange);
    return () => media.removeEventListener("change", onSystemChange);
  }, [theme]);

  const set = useCallback((next: Theme) => setTheme(next), [setTheme]);

  return { theme, resolvedTheme, setTheme: set, isHydrated };
}
