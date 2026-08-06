/**
 * Where each chat was scrolled to (sessionStorage).
 *
 * Same tier as drafts and the active chat id, for the same reason: scroll
 * position describes a *window looking at* the transcript, not the transcript
 * itself. Two tabs on the same chat should each keep their own place, and a
 * position from last week is meaningless against a conversation that has
 * grown since.
 *
 * Stored as a pixel offset string keyed per chat. A pixel offset is a weak
 * anchor — it's only valid against the exact content height that produced it,
 * so it survives a reload but not a transcript that changed underneath it.
 * That's the accepted tradeoff: anchoring to a message id instead would
 * survive edits, but costs a per-message ref map and a measurement pass to
 * restore, which is a lot of machinery for "put me back where I was."
 * `useChatScroll` clamps the restored value so a stale offset degrades to
 * "bottom of the list" rather than to a blank viewport.
 */

/** Namespaced to match the other stores; the chat id is appended. */
export const SCROLL_KEY_PREFIX = "multimodal:scroll:";

export function scrollKey(chatId: string): string {
  return `${SCROLL_KEY_PREFIX}${chatId}`;
}

export function readScrollTop(chatId: string | null): number | null {
  if (!chatId || typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(scrollKey(chatId));
    if (raw === null) return null;

    // Null rather than 0 on garbage: 0 is a legitimate position (scrolled to
    // the very top), so it can't double as "nothing stored".
    const value = Number.parseInt(raw, 10);
    return Number.isFinite(value) && value >= 0 ? value : null;
  } catch {
    return null;
  }
}

export function writeScrollTop(chatId: string | null, top: number): void {
  if (!chatId || typeof window === "undefined") return;
  try {
    // Sub-pixel offsets are real (zoom, fractional layouts) but meaningless
    // to restore, and rounding keeps the stored value short.
    window.sessionStorage.setItem(scrollKey(chatId), String(Math.round(top)));
  } catch {
    // Storage unavailable or full — scrolling still works, it just won't be
    // remembered across a reload.
  }
}

export function clearScrollTop(chatId: string | null): void {
  if (!chatId || typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(scrollKey(chatId));
  } catch {
    // Nothing to do — see writeScrollTop.
  }
}
