/**
 * useOutboxListener — applies the service worker's delivery reports.
 *
 * The worker replays queued sends but can't write the result anywhere the UI
 * can see: the chat store is page state, in page memory, and the worker has
 * neither. So it reports outcomes by postMessage and this hook is the one
 * place that turns those into store writes.
 *
 * Mounted once, in ChatLayout. Store access goes through `getState()` rather
 * than selectors because these are imperative writes from outside React —
 * subscribing would only cause re-renders this hook has no use for, and would
 * make the effect depend on values that change on every message.
 *
 * Also drains on mount, not just on `online`. A queued message survives a
 * reload (the entry is in the Cache API, the bubble is in localStorage), so
 * coming back to a tab that was closed while offline has to be a moment where
 * delivery is attempted — otherwise the message sits as "waiting" until the
 * connection happens to change state again.
 */
"use client";

import { useEffect } from "react";
import { useChatStore } from "@/store/chat-store";
import { requestFlush } from "@/lib/outbox";

const FAILED_NOTICE =
  "This message couldn't be delivered. Please try sending it again.";

export function useOutboxListener(): void {
  useEffect(() => {
    if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) return;

    const onMessage = (event: MessageEvent) => {
      const data = event.data;
      if (!data || typeof data !== "object") return;

      const store = useChatStore.getState();

      if (data.type === "outbox:delivered") {
        store.updateMessage(data.chatId, data.messageId, {
          content: data.answer,
          status: "sent",
          toolUsed: data.toolUsed,
        });
        // First reply in a chat assigns the backend conversation id; without
        // storing it the next turn starts a new server-side conversation and
        // silently loses this one's context.
        if (data.conversationId) {
          store.setConversationId(data.chatId, data.conversationId);
        }
        return;
      }

      if (data.type === "outbox:failed") {
        store.updateMessage(data.chatId, data.messageId, {
          content: FAILED_NOTICE,
          status: "error",
        });
      }
    };

    const onOnline = () => void requestFlush();

    navigator.serviceWorker.addEventListener("message", onMessage);
    window.addEventListener("online", onOnline);
    void requestFlush();

    return () => {
      navigator.serviceWorker.removeEventListener("message", onMessage);
      window.removeEventListener("online", onOnline);
    };
  }, []);
}
