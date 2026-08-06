/**
 * useChatScroll — owns every reason the transcript scrolls.
 *
 * Restoring a saved position and auto-scrolling to the newest message are the
 * same concern, not two: implemented separately they fight, and the
 * unconditional "scroll to bottom whenever messages change" this replaces
 * would win every time, wiping the restored position on the tick after it
 * landed. So both live here, ordered deliberately.
 *
 * Three rules:
 *
 *   1. **Restore once per chat**, on the first commit where that chat has
 *      messages to measure against. Tracked in a ref rather than state — it
 *      gates an imperative DOM write, and re-rendering on it would do nothing
 *      but schedule the same effect again.
 *   2. **Auto-scroll only when already at the bottom.** This is the standard
 *      chat "stick to bottom" rule, and it's what makes restore survive: a
 *      user reading history mid-transcript isn't yanked to the end when a
 *      reply arrives, and neither is a just-restored position.
 *   3. **Writes are debounced.** Unlike the draft (one write per keystroke),
 *      scroll fires at frame rate, and sessionStorage writes are synchronous
 *      on the main thread — exactly the way to make a scroll janky.
 *
 * The pending write carries its own chat id rather than reading `chatId` at
 * fire time, so scrolling and immediately switching chats can't file the old
 * chat's offset under the new chat's key.
 */
"use client";

import { useCallback, useEffect, useLayoutEffect, useRef } from "react";
import { readScrollTop, writeScrollTop } from "@/lib/scroll-positions";

/**
 * How close to the end still counts as "at the bottom". Not zero: smooth
 * scrolling, fractional device pixels, and a growing last bubble all leave a
 * few pixels of slack, and requiring an exact match would make the stick-to-
 * bottom behaviour fail intermittently.
 */
const BOTTOM_THRESHOLD_PX = 64;

/** Long enough to collapse a flick into one write, short enough to survive
 *  a reload that follows a scroll. */
const WRITE_DEBOUNCE_MS = 150;

/**
 * useLayoutEffect warns when a component is rendered on the server, and these
 * components are. The restore has to happen before paint though — in a
 * passive effect the browser paints at offset 0 first, so the transcript
 * visibly jumps. Picking the hook by environment is the standard resolution.
 */
const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

interface UseChatScrollResult {
  /** Attach to the scrolling element. */
  containerRef: React.RefObject<HTMLDivElement | null>;
  /** Attach to that same element's onScroll. */
  handleScroll: () => void;
}

export function useChatScroll(
  chatId: string | null,
  messages: readonly unknown[]
): UseChatScrollResult {
  const containerRef = useRef<HTMLDivElement>(null);

  // Starts true so a brand-new chat sticks to the bottom from its first
  // message, before any scroll event has had a chance to measure anything.
  const isAtBottomRef = useRef(true);

  /** Which chat has already had its position restored. */
  const restoredForRef = useRef<string | null>(null);

  const timerRef = useRef<number | null>(null);
  const pendingRef = useRef<{ chatId: string; top: number } | null>(null);

  /** Writes any debounced offset immediately. */
  const flush = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    const pending = pendingRef.current;
    if (pending) {
      pendingRef.current = null;
      writeScrollTop(pending.chatId, pending.top);
    }
  }, []);

  const handleScroll = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const distanceFromBottom =
      container.scrollHeight - container.clientHeight - container.scrollTop;
    isAtBottomRef.current = distanceFromBottom <= BOTTOM_THRESHOLD_PX;

    if (!chatId) return;

    // Snapshot both the id and the offset now; the timeout fires later, by
    // which time either could have moved on.
    pendingRef.current = { chatId, top: container.scrollTop };
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      const pending = pendingRef.current;
      if (!pending) return;
      pendingRef.current = null;
      writeScrollTop(pending.chatId, pending.top);
    }, WRITE_DEBOUNCE_MS);
  }, [chatId]);

  // Flush on chat switch and on unmount. Cleanup — not an event handler —
  // because switching chats doesn't unmount ChatWindow, it just re-renders it
  // with different messages, and the last 150ms of scrolling would otherwise
  // be dropped.
  useEffect(() => flush, [chatId, flush]);

  // Rule 1: restore. Runs on every commit but does work exactly once per
  // chat, and only once there is content to measure — an empty transcript has
  // no scrollHeight worth restoring against.
  useIsomorphicLayoutEffect(() => {
    const container = containerRef.current;
    if (!container || !chatId || messages.length === 0) return;
    if (restoredForRef.current === chatId) return;

    restoredForRef.current = chatId;

    const saved = readScrollTop(chatId);
    const maxScrollTop = container.scrollHeight - container.clientHeight;

    // Clamped, because the stored offset was measured against a transcript
    // that may since have shrunk — in-flight messages settle to shorter error
    // text on reload (see settleInFlightMessage), and history can be cleared
    // from another tab. An unclamped offset past the end silently lands at
    // the bottom anyway; clamping makes that the explicit, intended outcome.
    const target = saved === null ? maxScrollTop : Math.min(saved, maxScrollTop);

    container.scrollTop = target;
    isAtBottomRef.current = maxScrollTop - target <= BOTTOM_THRESHOLD_PX;
  }, [chatId, messages.length]);

  // Rule 2: stick to the bottom as messages arrive. Passive, so it runs after
  // the restore above in the same commit and can read the position it chose.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Nothing has been restored for this chat yet, so there's no established
    // position to preserve or override — leave it to the effect above.
    if (restoredForRef.current !== chatId) return;
    if (!isAtBottomRef.current) return;

    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [messages, chatId]);

  return { containerRef, handleScroll };
}
