/**
 * Theme primitives — shared by the store, the toggle, and the blocking
 * script in app/layout.tsx.
 *
 * The rendering side is already in place: app/globals.css declares
 * `@custom-variant dark (&:is(.dark *))` and a full `.dark` palette. All
 * that's needed to switch themes is adding/removing `dark` on <html>, which
 * is what `applyTheme` does.
 *
 * "system" is a real third option, not a synonym for light: it follows the
 * OS setting, so a machine that flips to dark at sunset flips the app too.
 * That means there are two distinct concepts — the *preference* the user
 * picked (`Theme`) and the *result* after resolving "system"
 * (`ResolvedTheme`). Conflating them is why toggles get stuck showing the
 * wrong icon.
 */
export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

/**
 * localStorage key. Also hardcoded in the inline script in app/layout.tsx —
 * that script runs before any JavaScript module loads, so it can't import
 * this constant. Change one, change both.
 */
export const THEME_STORAGE_KEY = "multimodal:theme";

/** Bump alongside any breaking change to the persisted shape. */
export const THEME_STORAGE_VERSION = 1;

export const DARK_MEDIA_QUERY = "(prefers-color-scheme: dark)";

/**
 * Cookie holding the *resolved* appearance — "dark" or "light", never
 * "system".
 *
 * The only reason this exists is that the server can't read any of the other
 * storage tiers. localStorage, sessionStorage, the Cache API and IndexedDB
 * are all invisible to it, which is why app/layout.tsx needed a blocking
 * inline script: the server had no way to know the theme, so it always
 * emitted light HTML and a dark-mode user saw a white flash. A cookie is sent
 * with the navigation request itself, so the server can put `class="dark"` on
 * <html> before the response leaves.
 *
 * Resolved and not the preference, because "system" is unanswerable on the
 * server — `prefers-color-scheme` is a browser media query. Storing what was
 * actually painted last time gives the server something it can act on, and
 * the client corrects it on the rare occasion the OS setting changed in
 * between.
 *
 * The *preference* still lives in localStorage; this is a rendering hint, not
 * a second source of truth.
 *
 * Underscore rather than the `multimodal:` prefix used everywhere else: `:`
 * is a separator in the cookie grammar (RFC 6265), and while browsers
 * tolerate it, proxies and parsers are less forgiving.
 */
export const APPEARANCE_COOKIE = "multimodal_appearance";

/** A year. Re-set on every applyTheme, so it never realistically expires. */
const APPEARANCE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

/**
 * Publishes the resolved appearance to the server.
 *
 * Not HttpOnly — it can't be, since the client is what knows the answer and
 * has to write it. That's acceptable precisely because it holds no secret:
 * the worst an attacker can do with it is make the first paint the wrong
 * colour.
 *
 * `SameSite=Lax` still sends it on top-level navigations, which is the only
 * moment it's read. `Secure` is conditional because this runs on plain http
 * in local development, where a Secure cookie would be dropped.
 */
export function writeAppearanceCookie(resolved: ResolvedTheme): void {
  if (typeof document === "undefined") return;
  const secure = window.location.protocol === "https:" ? "; Secure" : "";
  document.cookie =
    `${APPEARANCE_COOKIE}=${resolved}; Path=/; Max-Age=${APPEARANCE_COOKIE_MAX_AGE}; SameSite=Lax${secure}`;
}

/** Turns the stored preference into the theme actually being displayed. */
export function resolveTheme(theme: Theme): ResolvedTheme {
  if (theme !== "system") return theme;
  return window.matchMedia(DARK_MEDIA_QUERY).matches ? "dark" : "light";
}

/**
 * Writes the theme to the DOM, and records the result for the server.
 *
 * `colorScheme` is set alongside the class so browser-native UI — scrollbars,
 * form controls, the autofill dropdown — matches too. Without it those stay
 * light against a dark page.
 *
 * The cookie write lives here rather than in the store because this is the
 * one function every path goes through: the toggle, rehydration, and the OS
 * change listener in hooks/use-theme.ts. Anywhere else and some path would
 * eventually paint a theme the server never hears about.
 */
export function applyTheme(theme: Theme): ResolvedTheme {
  const resolved = resolveTheme(theme);
  const root = document.documentElement;
  root.classList.toggle("dark", resolved === "dark");
  root.style.colorScheme = resolved;
  writeAppearanceCookie(resolved);
  return resolved;
}
