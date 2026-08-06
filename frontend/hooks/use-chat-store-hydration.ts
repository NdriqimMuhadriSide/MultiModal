/**
 * useChatStoreHydration — reads persisted chat state back in after mount.
 *
 * The store sets `skipHydration: true`, so it deliberately starts empty and
 * localStorage is read only when this hook runs. That ordering matters in
 * Next.js: these components are server-rendered first, where localStorage
 * doesn't exist. If the store hydrated during render, the server would emit
 * "one empty chat" while the browser emitted the restored history, and React
 * would report a hydration mismatch. Rehydrating in an effect means both
 * render the same initial markup and the restored state lands one tick later.
 *
 * Returns whether hydration has finished, so callers can hold back UI that
 * would otherwise be actively misleading for that tick — an "empty chat"
 * placeholder shown to someone who has 20 saved chats, for instance.
 *
 * Safe to call from more than one component: `rehydrate()` is idempotent, and
 * every caller gets the same flag.
 */
"use client";

import { useEffect, useState } from "react";
import { toStoredMessages, useChatStore } from "@/store/chat-store";
import { collectMediaKeys, pruneMedia } from "@/lib/media-cache";
import { putMessages } from "@/lib/message-db";

export function useChatStoreHydration(): boolean {
  const [isHydrated, setIsHydrated] = useState(false);

  useEffect(() => {
    const finish = () => {
      setIsHydrated(true);
      // The one moment the full chat list is known and settled, which makes
      // it the natural place to garbage-collect cached media. The store caps
      // persistence at MAX_PERSISTED_CHATS, so chats fall off the end on
      // their own and would otherwise leave their images on disk forever.
      // Fire-and-forget: nothing renders differently either way.
      const { chats } = useChatStore.getState();
      void pruneMedia(collectMediaKeys(chats));

      // Backfills the search archive. The subscription in chat-store only
      // sees messages that change from now on, so without this every chat
      // saved before the archive existed would be unsearchable. `put` is
      // keyed by message id, so re-writing what's already there is a no-op
      // rather than a duplicate.
      void putMessages(toStoredMessages(chats));
    };

    // Subscribe before triggering: localStorage is synchronous, so
    // rehydrate() can finish before it returns, and a listener attached
    // afterwards would never fire.
    const unsubscribe = useChatStore.persist.onFinishHydration(finish);

    if (useChatStore.persist.hasHydrated()) {
      finish();
    } else {
      void useChatStore.persist.rehydrate();
    }

    return unsubscribe;
  }, []);

  return isHydrated;
}
