/**
 * Message archive (IndexedDB).
 *
 * The fifth storage tier, and the first that can answer a *question* rather
 * than return a thing you already knew the key of. That's the whole reason it
 * exists here: searching chat history means "find messages whose content
 * matches X, newest first", and neither of the other tiers can express that.
 * localStorage would mean parsing the entire history into memory on every
 * keystroke; the Cache API would mean reading every entry and inspecting it,
 * since `match()` only takes an exact URL.
 *
 * It also quietly removes two limits. The store caps localStorage at
 * MAX_PERSISTED_CHATS = 50 because ~5MB is the ceiling there — messages
 * archived here have no such cap, so a chat that falls off the sidebar is
 * still findable. And writes are per-record: one message, one `put`, instead
 * of re-serializing every chat on every change.
 *
 * Written against the raw IndexedDB API rather than a wrapper like `idb`.
 * That costs the promise plumbing below, but this is one store with two
 * indexes and no migrations yet — not enough surface to justify a dependency.
 *
 * Every operation degrades to an empty result instead of throwing. The
 * archive is derived data: it's rebuilt from the chat store on every
 * hydration (see hooks/use-chat-store-hydration.ts), so losing it costs a
 * search result, never a message.
 */
import type { MessageRole } from "@/types";

const DB_NAME = "multimodal";
const DB_VERSION = 1;
const STORE = "messages";

/** Newest-first cursor scan stops here. */
const DEFAULT_LIMIT = 30;

/** Below this, a substring match returns most of the archive. */
export const MIN_QUERY_LENGTH = 2;

export interface StoredMessage {
  /** The message id — also the keyPath, which makes writes idempotent. */
  id: string;
  chatId: string;
  /**
   * Snapshot of the chat's title at write time, so a result can name its
   * conversation even when that chat has aged out of the store. Safe here
   * because a chat is titled once, from its first user message, and never
   * renamed afterwards.
   */
  chatTitle: string;
  role: MessageRole;
  content: string;
  createdAt: string;
}

let dbPromise: Promise<IDBDatabase | null> | null = null;

/**
 * Opens (and on first run, creates) the database. Memoised — repeated calls
 * share one connection.
 *
 * `onupgradeneeded` is the only place a schema may change, and it fires
 * inside a version-change transaction the browser controls. Bumping
 * DB_VERSION is what triggers it; adding an index without a bump does
 * nothing, which is the classic IndexedDB trap.
 */
function openDatabase(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);

  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (db.objectStoreNames.contains(STORE)) return;

        const store = db.createObjectStore(STORE, { keyPath: "id" });
        // by-chat: every message in one conversation, for targeted deletes.
        store.createIndex("by-chat", "chatId");
        // by-created: ISO timestamps sort lexicographically, so walking this
        // index backwards is exactly "newest first" with no sorting step.
        store.createIndex("by-created", "createdAt");
      };

      request.onsuccess = () => resolve(request.result);

      // Blocked means another tab holds an older version open. Rather than
      // stall the UI waiting for it to close, give up for this page — search
      // is optional, and the next load will succeed.
      request.onblocked = () => resolve(null);
      request.onerror = () => resolve(null);
    });
  }

  return dbPromise;
}

/**
 * Writes messages, one transaction for the batch.
 *
 * `put` rather than `add`, keyed by message id: re-writing a message that
 * already exists overwrites it, which is what makes this safe to call both
 * for new messages and for edits (an assistant bubble filled in after its
 * request resolves).
 */
export async function putMessages(messages: StoredMessage[]): Promise<void> {
  if (messages.length === 0) return;

  const db = await openDatabase();
  if (!db) return;

  await new Promise<void>((resolve) => {
    let tx: IDBTransaction;
    try {
      tx = db.transaction(STORE, "readwrite");
    } catch {
      resolve();
      return;
    }

    const store = tx.objectStore(STORE);
    for (const message of messages) {
      store.put(message);
    }

    // All three resolve rather than reject: a failed archive write is not
    // something any caller can act on.
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
    tx.onabort = () => resolve();
  });
}

/**
 * Finds messages containing `query`, newest first.
 *
 * A cursor walk with a substring test, not a real full-text search: there's
 * no index that can answer "contains", so the scan is linear in archive size
 * and stops as soon as `limit` matches are found. Proper full-text would mean
 * storing a tokenised word list per message and indexing that — a different
 * schema, worth doing when the archive gets big enough to feel this.
 *
 * The early `resolve` on hitting the limit ends the transaction without
 * reading the rest, which is why the newest-first direction matters: it's
 * also the order people want results in.
 */
export async function searchMessages(
  query: string,
  limit: number = DEFAULT_LIMIT
): Promise<StoredMessage[]> {
  const needle = query.trim().toLowerCase();
  if (needle.length < MIN_QUERY_LENGTH) return [];

  const db = await openDatabase();
  if (!db) return [];

  return new Promise((resolve) => {
    const results: StoredMessage[] = [];

    let tx: IDBTransaction;
    try {
      tx = db.transaction(STORE, "readonly");
    } catch {
      resolve(results);
      return;
    }

    // "prev" walks the index in descending order — newest message first.
    const request = tx.objectStore(STORE).index("by-created").openCursor(null, "prev");

    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor || results.length >= limit) {
        resolve(results);
        return;
      }

      const message = cursor.value as StoredMessage;
      if (message.content.toLowerCase().includes(needle)) {
        results.push(message);
      }

      cursor.continue();
    };

    request.onerror = () => resolve(results);
    tx.onabort = () => resolve(results);
  });
}

/** Drops the archive. Pairs with clearPersistedChats(). */
export async function clearMessages(): Promise<void> {
  const db = await openDatabase();
  if (!db) return;

  await new Promise<void>((resolve) => {
    let tx: IDBTransaction;
    try {
      tx = db.transaction(STORE, "readwrite");
    } catch {
      resolve();
      return;
    }

    tx.objectStore(STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
    tx.onabort = () => resolve();
  });
}
