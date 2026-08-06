/**
 * useDraft — the chat input's text, backed by sessionStorage.
 *
 * Same deferred-read discipline as the persisted stores, for the same reason:
 * sessionStorage doesn't exist during server rendering, so reading it while
 * rendering would make the server emit an empty textarea and the browser emit
 * a filled one — a hydration mismatch. The initial state is therefore always
 * `""`, and the stored draft lands one tick later, from an effect.
 *
 * Writes go through `setValue` rather than an effect watching `value`. That
 * ordering matters when switching chats: an effect would fire after `chatId`
 * had already changed and write the outgoing chat's text under the incoming
 * chat's key. Writing at the point of the keystroke means the id is always
 * the one the user was actually typing into.
 *
 * Each write is synchronous, one per keystroke. That's deliberate rather than
 * debounced — the payload is a short string, and it matches how the chat
 * store already behaves. A debounce would add a flush-on-unmount path and the
 * stale-key race described above, to save microseconds.
 */
"use client";

import { useCallback, useEffect, useState } from "react";
import { clearDraft, readDraft, writeDraft } from "@/lib/drafts";

interface UseDraftResult {
  value: string;
  /** Updates the draft and persists it for this tab. */
  setValue: (next: string) => void;
  /** Drops the draft — call after a successful send. */
  clear: () => void;
}

export function useDraft(chatId: string | null): UseDraftResult {
  const [value, setValueState] = useState("");

  // Runs on mount (the deferred read) and again on every chat switch, which
  // is the same operation: load whatever this tab has stored for this chat.
  // A chat with no draft correctly resets the box to empty.
  useEffect(() => {
    setValueState(readDraft(chatId));
  }, [chatId]);

  const setValue = useCallback(
    (next: string) => {
      setValueState(next);
      writeDraft(chatId, next);
    },
    [chatId]
  );

  const clear = useCallback(() => {
    setValueState("");
    clearDraft(chatId);
  }, [chatId]);

  return { value, setValue, clear };
}
