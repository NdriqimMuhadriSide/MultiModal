/**
 * Onboarding state (Zustand + localStorage) — what the user has already seen.
 *
 * A third store rather than a field on chat-store or theme-store, for the
 * same reason those two are separate: distinct localStorage keys mean
 * clearing chat history can't reset onboarding, and resetting onboarding
 * can't wipe the theme. Each of the three is independently disposable.
 *
 * The persisted shape is a map of id → dismissed version, not a list of
 * ids, so bumping an announcement's `version` re-shows it. See
 * lib/announcements.ts for the reasoning.
 */
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import {
  ONBOARDING_STORAGE_KEY,
  ONBOARDING_STORAGE_VERSION,
  type DismissedMap,
} from "@/lib/announcements";

interface OnboardingState {
  dismissed: DismissedMap;

  /** Records that `id` was dismissed at `version`. Idempotent. */
  dismiss: (id: string, version: number) => void;

  /** Clears every dismissal, so all announcements show again. */
  resetDismissals: () => void;
}

/** The subset actually written to storage — data only, never the methods. */
type PersistedOnboardingState = Pick<OnboardingState, "dismissed">;

export const useOnboardingStore = create<OnboardingState>()(
  persist(
    (set) => ({
      // Deferred hydration means this is also what the server renders: an
      // empty map, i.e. "nothing dismissed yet". The banner doesn't render
      // until hydration finishes, so that never reaches the screen.
      dismissed: {},

      dismiss: (id, version) =>
        set((state) => ({
          // Highest version wins. Guards the case where two tabs are open on
          // different builds — an older tab dismissing v1 must not walk back
          // a v2 dismissal already recorded by the newer one.
          dismissed: {
            ...state.dismissed,
            [id]: Math.max(state.dismissed[id] ?? 0, version),
          },
        })),

      resetDismissals: () => set({ dismissed: {} }),
    }),
    {
      name: ONBOARDING_STORAGE_KEY,
      version: ONBOARDING_STORAGE_VERSION,
      storage: createJSONStorage(() => localStorage),
      partialize: (state): PersistedOnboardingState => ({
        dismissed: state.dismissed,
      }),

      // Same reason as the other two stores: localStorage doesn't exist
      // during server rendering, so the read is triggered after mount —
      // see hooks/use-announcement.ts.
      skipHydration: true,

      // Defends against a hand-edited or truncated payload. `dismissed` is
      // read with property access on every render; a stored `null` or an
      // array would throw or silently never match.
      merge: (persisted, current): OnboardingState => {
        const stored = (persisted ?? {}) as Partial<PersistedOnboardingState>;
        const dismissed =
          stored.dismissed && typeof stored.dismissed === "object" && !Array.isArray(stored.dismissed)
            ? stored.dismissed
            : {};
        return { ...current, dismissed };
      },
    }
  )
);

/**
 * Wipes persisted onboarding state. Exported separately from
 * `resetDismissals` (which only resets the in-memory store) so a caller can
 * do both — mirrors `clearPersistedChats` in store/chat-store.ts.
 *
 * Handy in the console for testing a banner without clearing everything:
 * `useOnboardingStore.getState().resetDismissals()`
 */
export function clearPersistedOnboarding(): void {
  useOnboardingStore.persist.clearStorage();
}
