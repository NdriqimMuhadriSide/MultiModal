/**
 * Unsent message drafts (sessionStorage).
 *
 * This is the first thing in the app that deliberately does *not* use
 * localStorage, and the choice is the whole point:
 *
 *   - **Per tab.** Two tabs open on two different chats must each keep their
 *     own half-typed message. A shared store would have them overwrite each
 *     other on every keystroke.
 *   - **Per visit.** A draft you walked away from is a thought you abandoned.
 *     Resurfacing it a week later is noise, not a feature — so it should die
 *     with the tab, which is exactly what sessionStorage does for free.
 *
 * Stored as a raw string rather than a JSON envelope: there's no versioned
 * shape to migrate, and it keeps the value legible in DevTools.
 *
 * Keyed per chat (`multimodal:draft:<chatId>`) rather than one global draft,
 * so switching chats in the sidebar and coming back doesn't lose what you
 * were writing. Orphaned keys (a chat cleared while a draft existed) are
 * bounded by the tab's lifetime — sessionStorage discards them on close.
 */

/** Namespaced to match the other stores; the chat id is appended. */
export const DRAFT_KEY_PREFIX = "multimodal:draft:";

export function draftKey(chatId: string): string {
  return `${DRAFT_KEY_PREFIX}${chatId}`;
}

/**
 * Every access is wrapped: `sessionStorage` throws rather than returning null
 * in a few real situations (Safari private browsing, quota exhausted, storage
 * blocked by cookie policy). A lost draft is a minor annoyance; an exception
 * thrown out of an onChange handler unmounts the whole input, so it's never
 * worth propagating.
 */
export function readDraft(chatId: string | null): string {
  if (!chatId || typeof window === "undefined") return "";
  try {
    return window.sessionStorage.getItem(draftKey(chatId)) ?? "";
  } catch {
    return "";
  }
}

export function writeDraft(chatId: string | null, value: string): void {
  if (!chatId || typeof window === "undefined") return;
  try {
    // An empty draft is the absence of a draft — remove the key instead of
    // storing "", so DevTools shows only chats actually mid-composition.
    if (value) {
      window.sessionStorage.setItem(draftKey(chatId), value);
    } else {
      window.sessionStorage.removeItem(draftKey(chatId));
    }
  } catch {
    // Storage unavailable or full. The draft stays in React state and simply
    // doesn't survive a reload.
  }
}

export function clearDraft(chatId: string | null): void {
  writeDraft(chatId, "");
}
