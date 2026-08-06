/**
 * Service worker — offline support.
 *
 * Everything before this used the Cache API from the page, which is enough to
 * *store* things but not to change what happens when a request fails. A
 * service worker is a separate script the browser keeps running outside any
 * page, sitting between the app and the network, and it's the only place a
 * failed request can be answered from cache instead. That is the entire
 * reason it exists here.
 *
 * Step 1: when a *navigation* fails because there's no network, serve a
 * cached offline page instead of the browser's dinosaur.
 *
 * Step 2: serve the app's static assets (JS, CSS, fonts, icons) from cache
 * instead of the network. These are content-hashed by the build, so a given
 * URL's bytes can never change — which makes cache-first not just safe but
 * strictly correct, and turns a repeat visit into zero asset requests.
 *
 * Step 3: keep a copy of every page you actually visit, and serve it when the
 * network is gone. Combined with step 2 this is what makes the real app come
 * up offline rather than the fallback: the HTML and the JS both resolve from
 * cache, and the chat history was never on the server to begin with — it's in
 * localStorage. The assistant still can't answer anything, because that lives
 * on the backend, but the app itself is fully readable.
 *
 * Step 4: keep the last successful response for read-only API data, so the
 * ingested-document list is readable offline instead of an error state.
 *
 * Step 5: hold writes that failed for lack of a network and replay them when
 * it returns. This is the one thing here that isn't caching at all — it's the
 * worker acting on its own, after the page that wanted the request is gone.
 *
 * Not registered in development — see components/ServiceWorkerRegistrar.tsx.
 */

/**
 * Bumping the version makes `install` build a fresh cache and `activate`
 * delete the old one. Cache contents are immutable in practice (the browser
 * has no reason to re-fetch what it already has), so a version in the name is
 * the standard way to ship an update.
 */
const VERSION = "v1";
const OFFLINE_CACHE = `multimodal:offline-${VERSION}`;
const STATIC_CACHE = `multimodal:static-${VERSION}`;
const PAGES_CACHE = `multimodal:pages-${VERSION}`;
const DATA_CACHE = `multimodal:data-${VERSION}`;
const OFFLINE_URL = "/offline";

/**
 * The outbox. Unversioned and absent from MANAGED_PREFIXES on purpose: every
 * other cache here can be rebuilt from the network, but this one holds the
 * only copy of something the user wrote. A worker upgrade must not take it
 * out. Keep in sync with lib/outbox.ts.
 */
const OUTBOX_CACHE = "multimodal:outbox-v1";
const OUTBOX_SYNC_TAG = "multimodal-outbox";

/**
 * Cache-name prefixes this worker owns and is allowed to delete on activate.
 *
 * `multimodal:media-` is deliberately absent: it holds user-uploaded images
 * and audio, is written by the page (lib/media-cache.ts), and is the one
 * cache here whose loss destroys something the user can't get back.
 */
const MANAGED_PREFIXES = [
  "multimodal:offline-",
  "multimodal:static-",
  "multimodal:pages-",
  "multimodal:data-",
];

/**
 * Read-only API endpoints worth keeping a copy of.
 *
 * An explicit allowlist, not a `/api/v1/` prefix match, because most of this
 * backend is the opposite of cacheable: /chat, /agent, /vision, /audio are
 * POSTs whose whole purpose is to produce something new.
 *
 * `/health` is deliberately excluded even though it's a cheap GET. It exists
 * to answer "is the backend reachable right now", and a cached 200 would make
 * the header claim "Backend Connected" with no backend in sight — turning a
 * correct offline indicator into a lie.
 *
 * Matched by pathname across any origin: the backend host comes from
 * NEXT_PUBLIC_API_URL, which is baked into the page bundle at build time and
 * is not visible to a static file in /public.
 */
const CACHEABLE_API_PATHS = ["/api/v1/documents"];

function isCacheableApi(url) {
  return CACHEABLE_API_PATHS.includes(url.pathname);
}

/**
 * Cross-origin responses are `cors`, same-origin ones are `basic`; both have
 * readable bodies. An `opaque` response (no CORS headers) reports status 0
 * and an unreadable body, so caching one stores a black box that can never be
 * validated or served meaningfully.
 */
function isCacheableResponse(response) {
  return response.ok && (response.type === "basic" || response.type === "cors");
}

/**
 * Assets safe to serve cache-first.
 *
 * `/_next/static/` is the whole point: every filename there contains a hash
 * of its contents, so a URL is a permanent identifier for exactly those
 * bytes. A new build produces new filenames rather than new contents at the
 * same name, which means a cached entry can never go stale — it can only
 * become unreferenced. Fonts loaded by next/font are self-hosted under that
 * same path, so they come along for free.
 *
 * The extension check covers /public (icons, svgs), which is *not* hashed.
 * That's a real tradeoff: replacing favicon.ico without renaming it means
 * clients keep the old one until the cache version is bumped. Acceptable for
 * static decoration, and the reason nothing else unhashed is listed here.
 */
function isStaticAsset(url) {
  if (url.origin !== self.location.origin) return false;
  if (url.pathname.startsWith("/_next/static/")) return true;
  return /\.(?:svg|png|jpe?g|gif|webp|ico|woff2?)$/i.test(url.pathname);
}

/**
 * install: fetch and store the fallback page, once, when this worker version
 * is first seen. `waitUntil` holds the worker alive until it resolves — the
 * browser would otherwise be free to kill the script mid-download and the
 * worker would activate with an empty cache.
 */
self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(OFFLINE_CACHE);
      // `cache: "reload"` bypasses the HTTP cache, so a stale copy of the
      // offline page can't be baked into a fresh install.
      await cache.add(new Request(OFFLINE_URL, { cache: "reload" }));

      // Take over as soon as install finishes instead of waiting for every
      // tab using the old worker to close. Safe here because the worker only
      // adds a fallback — there's no old behaviour a page could depend on.
      await self.skipWaiting();
    })()
  );
});

/**
 * activate: drop caches from older versions of *this* worker.
 *
 * Scoped to MANAGED_PREFIXES, and to names this version doesn't currently
 * use. This is also the only cleanup the static cache gets: entries from
 * previous builds are never individually evicted, because the worker has no
 * way to know which hashed filenames the newest build still references.
 * Bumping VERSION is therefore the mechanism for reclaiming that space, and
 * the reason a stale asset cache is a size problem rather than a correctness
 * one — nothing in it can ever serve wrong content.
 */
self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const current = new Set([OFFLINE_CACHE, STATIC_CACHE, PAGES_CACHE]);
      const names = await caches.keys();
      await Promise.all(
        names
          .filter(
            (name) =>
              MANAGED_PREFIXES.some((prefix) => name.startsWith(prefix)) && !current.has(name)
          )
          .map((name) => caches.delete(name))
      );
      await self.clients.claim();
    })()
  );
});

/**
 * Navigations: network-first, then this page's own cached copy, then the
 * generic offline page.
 *
 * Network-first and not cache-first, because unlike hashed assets a page's
 * HTML at a given URL genuinely does change between deploys. Serving the
 * cached copy first would pin every visitor to whatever build they saw last,
 * which is a far worse failure than a slow load.
 *
 * The three-level fallback matters: a page you've opened before comes back
 * exactly as it was, and only a URL this browser has never successfully
 * loaded drops to the generic "you're offline" screen.
 *
 * `fetch()` only rejects on a *network* failure — a 404 or 500 resolves
 * normally and is passed through, because the server answering with an error
 * is not the same as being offline. That also keeps error pages out of the
 * cache, since the `ok` check below rejects them.
 */
async function pageNetworkFirst(request) {
  const cache = await caches.open(PAGES_CACHE);

  try {
    const response = await fetch(request);
    if (response.ok && response.type === "basic") {
      void cache.put(request, response.clone());
    }
    return response;
  } catch {
    // `ignoreSearch` so `/documents?tab=x` can be answered by the copy saved
    // for `/documents`. Safe only because no route here renders differently
    // based on its query string — this is the line to revisit if one ever
    // does.
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return cached;

    const offline = await caches.open(OFFLINE_CACHE);
    const fallback = await offline.match(OFFLINE_URL);
    return fallback ?? Response.error();
  }
}

/**
 * Static assets: cache-first, filling the cache as they're requested.
 *
 * Runtime caching rather than precaching the whole shell at install. Doing it
 * properly at install would mean knowing every hashed filename up front,
 * which only the build knows — that's a generated manifest and a build step
 * (next-pwa, Serwist). Populating on first request costs one uncached visit
 * and needs no tooling at all; from the second visit on the result is
 * identical.
 */
async function cacheFirst(request) {
  const cache = await caches.open(STATIC_CACHE);

  const cached = await cache.match(request);
  if (cached) return cached;

  const response = await fetch(request);

  // Errors aren't cached — a 404 that got cached would outlive the deploy
  // that caused it.
  if (isCacheableResponse(response)) {
    // Clone before returning: a Response body is a stream and can only be
    // read once, so the page and the cache each need their own copy.
    void cache.put(request, response.clone());
  }

  return response;
}

/**
 * Read-only API data: stale-while-revalidate.
 *
 * Returns the cached copy immediately if there is one, and refreshes it in
 * the background regardless. The page therefore renders instantly from the
 * last known list and picks up any change on the next visit — the deliberate
 * tradeoff being that a document ingested moments ago on another device
 * shows up one load late.
 *
 * That's acceptable *here* because the documents list is append-mostly and
 * the cost of being one revision behind is cosmetic. It would not be
 * acceptable for anything the user acts on destructively.
 *
 * `event.waitUntil` keeps the worker alive for the background refresh; without
 * it the browser is free to shut the worker down the moment the cached
 * response is returned, cancelling the revalidation every time.
 */
function staleWhileRevalidate(request, event) {
  return (async () => {
    const cache = await caches.open(DATA_CACHE);
    const cached = await cache.match(request);

    const network = fetch(request)
      .then((response) => {
        if (isCacheableResponse(response)) {
          void cache.put(request, response.clone());
        }
        return response;
      })
      // Offline with a cached copy is the normal path, not an error — the
      // catch exists so the rejection is handled rather than logged as
      // uncaught. Falling back to `cached` covers the offline-first-load case
      // where there is nothing to serve.
      .catch(() => cached ?? Response.error());

    if (cached) {
      event.waitUntil(network);
      return cached;
    }

    return network;
  })();
}

/**
 * fetch: route each request to a strategy, or leave it alone.
 *
 * Returning without calling respondWith() hands the request back to the
 * browser untouched — that's the default for everything not matched here,
 * including every API call to the backend.
 */
self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only GET is cacheable, and only GET is safe to replay. A POST is a
  // request to *change* something and must always reach the server.
  if (request.method !== "GET") return;

  // Only full-document requests: typing a URL, following a link that misses
  // the client router, or reloading. Next's client-side navigations fetch RSC
  // payloads instead, which are *not* mode "navigate" and are deliberately
  // left to the network — they're served from the same URLs as the documents
  // above, so caching both would have them overwrite each other. Offline, a
  // failed RSC fetch makes Next fall back to a full page load, which lands
  // here and is answered from cache anyway.
  if (request.mode === "navigate") {
    event.respondWith(pageNetworkFirst(request));
    return;
  }

  const url = new URL(request.url);

  if (isStaticAsset(url)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  if (isCacheableApi(url)) {
    event.respondWith(staleWhileRevalidate(request, event));
  }
});

/* ------------------------------------------------------------------ */
/* Step 5 — the outbox                                                 */
/* ------------------------------------------------------------------ */

/**
 * Tells every open tab what happened to a queued message.
 *
 * The worker can't touch the chat store — that's page state, in page memory.
 * All it can do is report the outcome and let a page apply it; see
 * hooks/use-outbox-listener.ts. `includeUncontrolled` covers a tab that
 * loaded before this worker took over and so isn't formally controlled by it,
 * but is still showing the message waiting for this reply.
 */
async function broadcast(message) {
  const clients = await self.clients.matchAll({
    includeUncontrolled: true,
    type: "window",
  });
  for (const client of clients) {
    client.postMessage(message);
  }
}

/**
 * Sends everything in the outbox, oldest first.
 *
 * Three outcomes per entry, and the difference between them is the whole
 * design:
 *
 *   - **Delivered.** Drop it and tell the pages, which fill in the bubble.
 *   - **Rejected (4xx).** Drop it too. A malformed or unauthorized request
 *     will be just as malformed on the tenth retry; keeping it would mean
 *     retrying forever and showing a message as pending that never resolves.
 *   - **Still unreachable (network error or 5xx).** Keep it, and rethrow at
 *     the end so Background Sync schedules another attempt with its own
 *     backoff. Swallowing the failure here would consume the sync event and
 *     nothing would ever retry.
 *
 * Sequential rather than parallel: these are conversation turns, and firing
 * them at once would let the backend answer them out of order.
 */
async function drainOutbox() {
  const cache = await caches.open(OUTBOX_CACHE);
  const keys = await cache.keys();
  let unresolved = 0;

  for (const key of keys) {
    const stored = await cache.match(key);
    if (!stored) continue;

    let entry;
    try {
      entry = await stored.json();
    } catch {
      // Unreadable entry — nothing can ever be done with it.
      await cache.delete(key);
      continue;
    }

    try {
      const response = await fetch(entry.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(entry.body),
      });

      if (response.ok) {
        const data = await response.json();
        await cache.delete(key);
        await broadcast({
          type: "outbox:delivered",
          chatId: entry.chatId,
          messageId: entry.messageId,
          answer: data.answer,
          toolUsed: data.tool_used,
          conversationId: data.conversation_id,
        });
        continue;
      }

      if (response.status >= 400 && response.status < 500) {
        await cache.delete(key);
        await broadcast({
          type: "outbox:failed",
          chatId: entry.chatId,
          messageId: entry.messageId,
        });
        continue;
      }

      unresolved++;
    } catch {
      // Still offline.
      unresolved++;
    }
  }

  if (unresolved > 0) {
    throw new Error(`outbox: ${unresolved} entries pending retry`);
  }
}

/**
 * The good path: the browser fires this once it believes connectivity is
 * back, with its own retry schedule, and it will do so even if every tab has
 * been closed since. Chromium-only.
 */
self.addEventListener("sync", (event) => {
  if (event.tag === OUTBOX_SYNC_TAG) {
    event.waitUntil(drainOutbox());
  }
});

/**
 * The fallback for browsers without Background Sync, and the nudge a page
 * sends when it sees an `online` event. Works everywhere, but only while a
 * page is open to send it.
 */
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "outbox:flush") {
    // Caught here: unlike the sync event there's no retry schedule to signal
    // to, and an unhandled rejection would just be noise in the console.
    event.waitUntil(drainOutbox().catch(() => {}));
  }
});
