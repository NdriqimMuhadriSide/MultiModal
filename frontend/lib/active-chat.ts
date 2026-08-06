/**
 * Which chat this tab is looking at (sessionStorage).
 *
 * The chats themselves belong to the browser and stay in localStorage. The
 * *selection* doesn't: it's a property of a window, not of the account. With
 * both in one localStorage payload, two tabs fight — switching chats in tab A
 * rewrites `activeChatId` for everyone, so tab B silently jumps to a
 * different conversation the next time it loads. Splitting the pointer into
 * sessionStorage gives each tab its own place in shared history, which is how
 * every other tabbed document editor behaves.
 *
 * The pointer is deliberately allowed to dangle. A chat can disappear (cleared
 * history, the 50-chat persistence cap, another tab) while this tab still
 * names it, so the id is validated against the restored list on every read —
 * see `merge` in store/chat-store.ts. A tab that has never chosen anything
 * has no key at all, and falls back to the most recent chat.
 *
 * Same defensive wrapping as lib/drafts.ts, for the same reason: storage
 * access throws in private-browsing and quota cases, and the worst outcome
 * here is landing on the wrong chat — never worth an exception.
 */

/** Namespaced to match the other stores. */
export const ACTIVE_CHAT_STORAGE_KEY = "multimodal:active-chat";

export function readActiveChatId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage.getItem(ACTIVE_CHAT_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function writeActiveChatId(chatId: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (chatId) {
      window.sessionStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, chatId);
    } else {
      window.sessionStorage.removeItem(ACTIVE_CHAT_STORAGE_KEY);
    }
  } catch {
    // Storage unavailable. The selection still works for this page view; it
    // just won't survive a reload.
  }
}

export function clearActiveChatId(): void {
  writeActiveChatId(null);
}
