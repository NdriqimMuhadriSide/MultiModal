/**
 * Announcement registry and the rules for deciding which one to show.
 *
 * "Has this user seen X yet?" is device-local preference state, the same
 * tier as the theme: a few bytes that belong to *this browser*, worth
 * nothing to the backend, and costing nothing worse than one extra banner
 * if lost. So it lives in localStorage next to the theme, under its own key.
 *
 * Two ideas do the real work here:
 *
 *   1. **Dismissal is recorded per version, not as a boolean.** Storing
 *      `true` would mean an announcement can only ever be shown once — so
 *      rewriting the copy for a bigger release leaves everyone who dismissed
 *      the old one unable to see the new one. Storing the version they
 *      dismissed lets you bump `version` to re-show it, and leaves the
 *      never-seen-it case (`undefined`) working unchanged.
 *   2. **Announcements expire.** Without `expiresAt`, "Welcome to the new AI
 *      features!" is still greeting brand-new users a year after those
 *      features stopped being new, and the registry becomes a graveyard
 *      nobody dares prune. An expiry date makes removal safe and obvious.
 *
 * Note this is *onboarding state*, not feature flags in the gating sense.
 * Anything that decides whether a capability actually exists has to be
 * resolved server-side — a value in localStorage is editable by anyone with
 * DevTools open, so it can hide UI but it can never protect anything.
 */

/** localStorage key, namespaced to match the theme and chat stores. */
export const ONBOARDING_STORAGE_KEY = "multimodal:onboarding";

/** Bump alongside any breaking change to the persisted shape. */
export const ONBOARDING_STORAGE_VERSION = 1;

export interface Announcement {
  /** Stable across copy edits — this is what gets written to storage. */
  id: string;
  /** Bump to show this announcement again to people who dismissed it. */
  version: number;
  title: string;
  body: string;
  /**
   * ISO date after which this stops being shown to anyone, seen or not.
   * Omit for announcements that should run until manually removed.
   */
  expiresAt?: string;
}

/** id → the announcement version that was dismissed. */
export type DismissedMap = Record<string, number>;

/**
 * The live registry, highest priority first.
 *
 * Only one banner is ever on screen — stacking them turns the top of the app
 * into a notification centre nobody reads. `selectVisibleAnnouncement` takes
 * the first eligible entry, so ordering here is the priority order.
 */
export const ANNOUNCEMENTS: readonly Announcement[] = [
  {
    id: "welcome-ai-features",
    version: 1,
    title: "Welcome to the new AI features!",
    body: "Attach an image or an audio file to any message, or ask about a document you've uploaded — the assistant now picks the right tool on its own.",
    expiresAt: "2026-12-31T00:00:00.000Z",
  },
];

/**
 * True once the user has dismissed this announcement *at its current
 * version*. An unknown id reads as version 0, so anything never dismissed
 * is correctly treated as unseen.
 */
export function isDismissed(
  announcement: Announcement,
  dismissed: DismissedMap
): boolean {
  return (dismissed[announcement.id] ?? 0) >= announcement.version;
}

/**
 * True once past `expiresAt`. An unparseable date is treated as *not*
 * expired: a typo in the registry should show a banner slightly too long,
 * not silently suppress it with no visible symptom to debug.
 */
export function isExpired(announcement: Announcement, now: number): boolean {
  if (!announcement.expiresAt) return false;
  const expiry = Date.parse(announcement.expiresAt);
  return Number.isNaN(expiry) ? false : now >= expiry;
}

/**
 * The one announcement to render, or null when there's nothing to say.
 *
 * `now` is passed in rather than read from the clock here so the caller
 * controls when time is sampled — this runs after hydration, never during
 * render, which keeps server and client markup identical.
 */
export function selectVisibleAnnouncement(
  dismissed: DismissedMap,
  now: number
): Announcement | null {
  return (
    ANNOUNCEMENTS.find(
      (announcement) =>
        !isDismissed(announcement, dismissed) && !isExpired(announcement, now)
    ) ?? null
  );
}
