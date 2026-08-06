/**
 * Uploaded media (Cache API).
 *
 * The fourth storage tier in this app, and the first that isn't a key-value
 * store of strings. It exists for one reason the other three can't cover:
 * `Attachment.url` comes from `URL.createObjectURL`, which is valid only for
 * the page session that created it. `stripObjectUrls` blanks it before every
 * write to localStorage precisely because persisting the string would leave a
 * pointer to a blob the browser already released — see store/chat-store.ts.
 *
 * localStorage can't fix that: a `File` is binary, and 5MB of quota wouldn't
 * hold one photo. The Cache API stores whole `Response` objects on disk under
 * the origin's real quota, so the *bytes* can outlive the page and the
 * attachment can be rebuilt after a reload.
 *
 * The entries are keyed by synthetic paths (`/__media/<attachment-id>`) that
 * intentionally correspond to no server route. `cache.match()` never touches
 * the network, so the URL is just an opaque key that happens to satisfy the
 * Request type. There's no Service Worker here and none is needed — that's
 * only required to *intercept* fetches, not to read and write a cache.
 *
 * Every function degrades to a no-op or null instead of throwing:
 * `caches` is undefined outside a secure context, the quota can be exhausted,
 * and the browser evicts under disk pressure without asking. A missing entry
 * therefore has to be normal, not exceptional — it falls back to the same
 * "media expired" placeholder that is already the status quo today.
 */
import type { Chat } from "@/types";

const CACHE_NAME = "multimodal:media-v1";

/** Synthetic path prefix — deliberately not a real route. */
const KEY_PREFIX = "/__media/";

/** False during server rendering and in insecure contexts (plain http). */
function isSupported(): boolean {
  return typeof window !== "undefined" && "caches" in window;
}

export function mediaKey(attachmentId: string): string {
  return `${KEY_PREFIX}${attachmentId}`;
}

/**
 * Stores a file and resolves to its key, or null if it couldn't be cached.
 *
 * Callers can derive the key up front with `mediaKey()` and treat this as
 * fire-and-forget: a failed write is indistinguishable from a later eviction,
 * and both already resolve to the expired placeholder.
 */
export async function putMedia(attachmentId: string, file: File): Promise<string | null> {
  if (!isSupported()) return null;
  const key = mediaKey(attachmentId);
  try {
    const cache = await caches.open(CACHE_NAME);
    await cache.put(
      key,
      // The Content-Type is what makes the entry replayable — it's what the
      // <img>/<audio> element gets when the blob is read back out.
      new Response(file, {
        headers: {
          "Content-Type": file.type || "application/octet-stream",
          "Content-Length": String(file.size),
        },
      })
    );
    return key;
  } catch {
    return null;
  }
}

/**
 * Rebuilds a usable object URL from a cached entry, or null on a miss.
 *
 * The caller owns the returned URL and must revoke it — see
 * hooks/use-attachment-url.ts.
 */
export async function getMediaObjectUrl(key: string): Promise<string | null> {
  if (!isSupported()) return null;
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(key);
    if (!response) return null;
    return URL.createObjectURL(await response.blob());
  } catch {
    return null;
  }
}

/** Every cache key still referenced by a message somewhere in `chats`. */
export function collectMediaKeys(chats: Chat[]): string[] {
  const keys: string[] = [];
  for (const chat of chats) {
    for (const message of chat.messages) {
      for (const attachment of message.attachments ?? []) {
        if (attachment.cacheKey) keys.push(attachment.cacheKey);
      }
    }
  }
  return keys;
}

/**
 * Drops cached media that no chat references any more.
 *
 * Without this the cache only grows: the store persists at most
 * MAX_PERSISTED_CHATS, so older chats fall off the end while their images
 * stay on disk forever — megabytes each, invisible to the user.
 *
 * Accepted race: a second tab that uploads a file in the moment between this
 * tab reading its chat list and running the sweep would have its brand-new
 * entry deleted. The window is milliseconds, both tabs share the same
 * localStorage list, and the failure mode is one thumbnail degrading to the
 * expired placeholder — the same outcome as a browser eviction, which this
 * code already has to tolerate.
 */
export async function pruneMedia(keysInUse: Iterable<string>): Promise<void> {
  if (!isSupported()) return;
  try {
    const keep = new Set(keysInUse);
    const cache = await caches.open(CACHE_NAME);
    const requests = await cache.keys();
    await Promise.all(
      requests.map((request) => {
        // Stored as a path, read back as an absolute URL.
        const path = new URL(request.url).pathname;
        return keep.has(path) ? Promise.resolve(false) : cache.delete(request);
      })
    );
  } catch {
    // Sweeping is opportunistic — a failure just means it runs again later.
  }
}

/** Drops the whole cache. Pairs with clearPersistedChats(). */
export async function clearMediaCache(): Promise<void> {
  if (!isSupported()) return;
  try {
    await caches.delete(CACHE_NAME);
  } catch {
    // Nothing to do — see pruneMedia.
  }
}
