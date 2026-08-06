/**
 * Theme preference (Zustand + localStorage).
 *
 * Kept as its own store rather than a field on chat-store because the two
 * have nothing to do with each other and very different write patterns: the
 * chat store is rewritten on every keystroke-sized state change, this one
 * changes when a user clicks a toggle. Separate stores mean separate
 * localStorage keys, so clearing chat history can never wipe the theme.
 *
 * This is the storage tier working as intended: a few bytes of UI
 * preference that belongs to *this browser*, not to the account. Nobody
 * needs the backend to know it, and losing it costs one click to restore —
 * the opposite of the media-storage problem, where the client being the
 * only copy means real data loss.
 */
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import {
  applyTheme,
  THEME_STORAGE_KEY,
  THEME_STORAGE_VERSION,
  type Theme,
} from "@/lib/theme";

interface ThemeState {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      // Deferred hydration means this is also what the server renders, so
      // it must match what the inline script assumes when no preference is
      // stored yet (see app/layout.tsx).
      theme: "system",

      setTheme: (theme) => {
        // DOM first, then state: the class change is what the user actually
        // sees, and it shouldn't wait on a React render to land.
        applyTheme(theme);
        set({ theme });
      },
    }),
    {
      name: THEME_STORAGE_KEY,
      version: THEME_STORAGE_VERSION,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ theme: state.theme }),

      // Same reason as the chat store: localStorage doesn't exist during
      // server rendering. The inline script has already painted the correct
      // theme by now — this only re-syncs the React-side value.
      skipHydration: true,

      onRehydrateStorage: () => (state) => {
        if (state) applyTheme(state.theme);
      },
    }
  )
);
