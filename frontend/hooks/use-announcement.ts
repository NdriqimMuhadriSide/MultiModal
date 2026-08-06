/**
 * useAnnouncement — resolves which announcement (if any) to show, and
 * records the dismissal.
 *
 * Rehydration follows the same pattern as the chat and theme stores:
 * localStorage is read after mount, so server and client render identical
 * markup. The `isHydrated` flag matters more here than elsewhere — the
 * pre-hydration state is an empty `dismissed` map, which means "show the
 * banner". Rendering on that would flash the welcome banner at every
 * returning user on every load, which is the exact bug this feature exists
 * to prevent.
 *
 * `now` is sampled once, when hydration finishes, rather than read during
 * render. Reading the clock in a render pass would make the output depend on
 * when React happened to run, and expiry is a day-scale decision — an open
 * tab crossing an expiry boundary at midnight can keep its banner until the
 * next reload.
 */
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useOnboardingStore } from "@/store/onboarding-store";
import { selectVisibleAnnouncement, type Announcement } from "@/lib/announcements";

interface UseAnnouncementResult {
  /** The announcement to render, or null when there's nothing to show. */
  announcement: Announcement | null;
  /** Dismisses the currently visible announcement. No-op if there isn't one. */
  dismiss: () => void;
  isHydrated: boolean;
}

export function useAnnouncement(): UseAnnouncementResult {
  const dismissed = useOnboardingStore((state) => state.dismissed);
  const recordDismissal = useOnboardingStore((state) => state.dismiss);

  const [isHydrated, setIsHydrated] = useState(false);
  const [now, setNow] = useState<number | null>(null);

  useEffect(() => {
    const finish = () => {
      setNow(Date.now());
      setIsHydrated(true);
    };

    // Subscribe before triggering: localStorage is synchronous, so
    // rehydrate() can finish before it returns, and a listener attached
    // afterwards would never fire.
    const unsubscribe = useOnboardingStore.persist.onFinishHydration(finish);

    if (useOnboardingStore.persist.hasHydrated()) {
      finish();
    } else {
      void useOnboardingStore.persist.rehydrate();
    }

    return unsubscribe;
  }, []);

  const announcement = useMemo(
    () => (now === null ? null : selectVisibleAnnouncement(dismissed, now)),
    [dismissed, now]
  );

  const dismiss = useCallback(() => {
    if (!announcement) return;
    recordDismissal(announcement.id, announcement.version);
  }, [announcement, recordDismissal]);

  return { announcement, dismiss, isHydrated };
}
