/**
 * Outbox — writes that failed for lack of a network, kept until they land.
 *
 * The page can't retry a request it no longer exists to make: close the tab
 * and the retry goes with it. The service worker does outlive the page, so
 * the queue lives in a cache both sides can reach, and the worker drains it
 * when connectivity returns — see the sync/message handlers in public/sw.js.
 *
 * Stored in the Cache API rather than IndexedDB because a queue entry is
 * exactly what a Response already is (a body plus a URL to identify it), and
 * because a service worker has no localStorage at all — the storage this app
 * reaches for everywhere else simply isn't available on that side.
 *
 * Deliberately *not* versioned with the worker's VERSION constant, and
 * deliberately absent from MANAGED_PREFIXES in sw.js: every other cache can
 * be rebuilt from the network, but this one holds the only copy of something
 * the user wrote. A worker upgrade must never take it out.
 */

export const OUTBOX_CACHE = "multimodal:outbox-v1";
export const OUTBOX_KEY_PREFIX = "/__outbox/";

/** Background Sync tag. Must match the `sync` handler in public/sw.js. */
export const OUTBOX_SYNC_TAG = "multimodal-outbox";

export interface OutboxEntry {
  /** Where the reply belongs once it arrives. */
  chatId: string;
  /** The pending assistant bubble to fill in. */
  messageId: string;
  /**
   * Absolute URL and JSON body, captured at enqueue time. The worker is a
   * static file in /public and can't see NEXT_PUBLIC_API_URL, so the page has
   * to tell it where the backend is rather than the worker deriving it.
   */
  url: string;
  body: unknown;
  createdAt: string;
}

/** Background Sync isn't in TypeScript's DOM lib and isn't in every browser. */
interface SyncCapableRegistration extends ServiceWorkerRegistration {
  sync?: { register: (tag: string) => Promise<void> };
}

/**
 * Whether queueing can work at all.
 *
 * `controller` is null when no service worker is driving this page — no
 * registration yet, first load before activation, or development, where the
 * worker is never registered. Without one, a queued entry would sit in the
 * cache with nothing to ever send it, so callers fall back to reporting the
 * failure instead of promising a delivery that can't happen.
 */
export function canQueue(): boolean {
  return (
    typeof navigator !== "undefined" &&
    "serviceWorker" in navigator &&
    navigator.serviceWorker.controller !== null &&
    typeof caches !== "undefined"
  );
}

export async function enqueue(entry: OutboxEntry): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  try {
    const cache = await caches.open(OUTBOX_CACHE);
    await cache.put(
      `${OUTBOX_KEY_PREFIX}${entry.messageId}`,
      new Response(JSON.stringify(entry), {
        headers: { "Content-Type": "application/json" },
      })
    );
    return true;
  } catch {
    return false;
  }
}

/**
 * Asks the worker to drain the queue.
 *
 * Background Sync is the good path: the browser owns the retry schedule and
 * will fire the event once connectivity is back even if every tab has since
 * been closed. It's Chromium-only, so the fallback is a direct message to the
 * worker — which works everywhere but only while a page is open, since
 * nothing else would be around to send it.
 */
export async function requestFlush(): Promise<void> {
  if (!canQueue()) return;

  try {
    const registration = (await navigator.serviceWorker.ready) as SyncCapableRegistration;

    if (registration.sync) {
      await registration.sync.register(OUTBOX_SYNC_TAG);
      return;
    }

    registration.active?.postMessage({ type: "outbox:flush" });
  } catch {
    // sync.register() rejects if the browser has denied background sync for
    // this origin. The direct nudge still works while the page is open.
    try {
      const registration = await navigator.serviceWorker.ready;
      registration.active?.postMessage({ type: "outbox:flush" });
    } catch {
      // No worker. The entry stays queued for the next load.
    }
  }
}
