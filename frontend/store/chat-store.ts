/**
 * Global chat state (Zustand), persisted to localStorage.
 *
 * Why a store instead of local component state: chat history needs to be
 * visible in the sidebar (Chat History list) while also being edited from
 * the main chat window and input — that's cross-tree state, which is
 * exactly what a store is for. Keeping it here (rather than in
 * ChatWindow) means the sidebar can read `chats` without prop drilling.
 *
 * This store holds *state and simple mutations only* — it does not call
 * the API. Orchestration (calling ChatService, handling loading/error,
 * committing results into the store) lives in hooks/use-chat.ts, keeping
 * "what the data looks like" separate from "how we fetch it".
 *
 * Persistence (`persist` middleware) survives a page reload: `chats` is
 * written to localStorage on every change and read back on mount.
 * `conversationId` rides along, so a restored chat keeps talking to the
 * *same* backend conversation — the server-side memory layer already has
 * those turns, so the two stay in sync.
 *
 * `activeChatId` is the exception: it lives in *sessionStorage*, not here.
 * The chats belong to the browser, but which one you're looking at belongs to
 * the tab — see lib/active-chat.ts for the reasoning. It stays a normal field
 * on this store; only its storage backend differs, mirrored out by the
 * subscription at the bottom of this file and read back in `merge`.
 *
 * Four things are deliberately not naive about this:
 *
 *   1. Hydration is deferred (`skipHydration`), because Next.js renders
 *      these components on the server where localStorage doesn't exist.
 *      Reading it during render would make the server and client markup
 *      disagree. hooks/use-chat-store-hydration.ts rehydrates after mount.
 *   2. Attachment object URLs are dropped on write — see `stripObjectUrls`.
 *   3. In-flight statuses are settled on read — see `settleInFlightMessage`.
 *   4. The restored `activeChatId` is validated against the restored chats,
 *      because the two stores can now disagree — see `merge`.
 */
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { clearActiveChatId, readActiveChatId, writeActiveChatId } from "@/lib/active-chat";
import { clearMediaCache } from "@/lib/media-cache";
import { clearMessages, putMessages, type StoredMessage } from "@/lib/message-db";
import type { Chat, Message } from "@/types";

/** localStorage key, namespaced so it can't collide with other app state. */
const STORAGE_KEY = "multimodal:chat";

/**
 * Bump when a change to the persisted shape would break older saved data.
 * On mismatch, `migrate` runs (or the stored state is discarded if it can't
 * be upgraded) instead of feeding a stale shape into the app.
 */
const STORAGE_VERSION = 1;

/**
 * Newest-first cap on what gets written. localStorage gives roughly 5MB per
 * origin and fails *silently* when full, which would look like "persistence
 * randomly stopped working" much later. Chats are stored newest-first
 * (startNewChat prepends), so slicing from the front keeps the recent ones.
 */
const MAX_PERSISTED_CHATS = 50;

/**
 * Statuses that only make sense while a request is in flight. A message
 * still wearing one of these in localStorage means the page was reloaded
 * mid-request — the fetch that would have resolved it died with the page.
 */
const IN_FLIGHT_STATUSES: ReadonlySet<Message["status"]> = new Set<Message["status"]>([
  "sending",
  "streaming",
  "searching",
  "generating",
  "transcribing",
  "analyzing",
]);

const INTERRUPTED_NOTICE =
  "This response was interrupted by a page reload. Please ask again.";

function createEmptyChat(): Chat {
  const now = new Date().toISOString();
  return {
    id: crypto.randomUUID(),
    conversationId: null,
    title: "New Chat",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

/**
 * Blanks `Attachment.url` before writing to storage.
 *
 * Those URLs come from `URL.createObjectURL`, which hands back a handle
 * that is only valid for the page session that created it. Persisting the
 * string keeps a pointer to a blob the browser has already released, so
 * after a reload it would render as a broken <img> / dead <audio>.
 *
 * `cacheKey` deliberately survives (the spread keeps it): the URL is dead but
 * the *bytes* are still on disk in the Cache API, so MessageBubble can mint a
 * fresh object URL from that key — see hooks/use-attachment-url.ts. An
 * attachment with neither a live URL nor a resolvable key still falls back to
 * the "media expired" placeholder.
 */
function stripObjectUrls(message: Message): Message {
  if (!message.attachments?.length) return message;
  return {
    ...message,
    attachments: message.attachments.map((attachment) => ({
      ...attachment,
      url: "",
    })),
  };
}

/**
 * Turns a message that was mid-request at reload into a settled error.
 *
 * Without this, a restored "sending" message renders TypingIndicator
 * forever — a spinner with nothing behind it, since no request is running.
 * Keeps any partial content and only supplies copy when the bubble is empty.
 */
function settleInFlightMessage(message: Message): Message {
  if (!message.status || !IN_FLIGHT_STATUSES.has(message.status)) {
    return message;
  }
  return {
    ...message,
    status: "error",
    content: message.content || INTERRUPTED_NOTICE,
  };
}

interface ChatState {
  chats: Chat[];
  activeChatId: string | null;

  activeChat: () => Chat | undefined;

  startNewChat: () => void;
  setActiveChat: (chatId: string) => void;

  appendMessage: (chatId: string, message: Message) => void;
  updateMessage: (chatId: string, messageId: string, patch: Partial<Message>) => void;
  /**
   * Concatenates onto a message's existing content, for answers that arrive
   * in pieces. Separate from `updateMessage` because a streaming consumer
   * holds only the newest delta, not the text so far — making it patch
   * `content` would mean tracking the accumulated string outside the store
   * and re-sending the whole answer on every token.
   */
  appendToMessage: (chatId: string, messageId: string, delta: string) => void;
  setConversationId: (chatId: string, conversationId: string) => void;

  /** Drops every chat and starts a fresh one, clearing localStorage too. */
  clearAllChats: () => void;
}

/**
 * The subset actually written to localStorage — data only, never the methods,
 * and no longer `activeChatId` (that goes to sessionStorage instead).
 *
 * Dropping a field doesn't need a STORAGE_VERSION bump: payloads written by
 * the previous build still carry `activeChatId`, and `merge` simply ignores
 * it. Bumping without supplying a `migrate` would make zustand discard the
 * stored state entirely — i.e. delete everyone's chat history to remove one
 * string.
 */
type PersistedChatState = Pick<ChatState, "chats">;

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => {
      const initialChat = createEmptyChat();

      return {
        chats: [initialChat],
        activeChatId: initialChat.id,

        activeChat: () => get().chats.find((chat) => chat.id === get().activeChatId),

        startNewChat: () => {
          const chat = createEmptyChat();
          set((state) => ({
            chats: [chat, ...state.chats],
            activeChatId: chat.id,
          }));
        },

        setActiveChat: (chatId) => set({ activeChatId: chatId }),

        appendMessage: (chatId, message) =>
          set((state) => ({
            chats: state.chats.map((chat) =>
              chat.id === chatId
                ? {
                    ...chat,
                    messages: [...chat.messages, message],
                    updatedAt: new Date().toISOString(),
                    // First user message becomes the chat title, ChatGPT-style.
                    title:
                      chat.messages.length === 0 && message.role === "user"
                        ? message.content.slice(0, 48)
                        : chat.title,
                  }
                : chat
            ),
          })),

        updateMessage: (chatId, messageId, patch) =>
          set((state) => ({
            chats: state.chats.map((chat) =>
              chat.id === chatId
                ? {
                    ...chat,
                    messages: chat.messages.map((message) =>
                      message.id === messageId ? { ...message, ...patch } : message
                    ),
                  }
                : chat
            ),
          })),

        appendToMessage: (chatId, messageId, delta) =>
          set((state) => ({
            chats: state.chats.map((chat) =>
              chat.id === chatId
                ? {
                    ...chat,
                    messages: chat.messages.map((message) =>
                      message.id === messageId
                        ? { ...message, content: message.content + delta }
                        : message
                    ),
                  }
                : chat
            ),
          })),

        setConversationId: (chatId, conversationId) =>
          set((state) => ({
            chats: state.chats.map((chat) =>
              chat.id === chatId ? { ...chat, conversationId } : chat
            ),
          })),

        clearAllChats: () => {
          const chat = createEmptyChat();
          set({ chats: [chat], activeChatId: chat.id });
        },
      };
    },
    {
      name: STORAGE_KEY,
      version: STORAGE_VERSION,
      storage: createJSONStorage(() => localStorage),

      // Persist data only. The methods are rebuilt by the initializer on
      // every load, so writing them would just bloat storage.
      partialize: (state): PersistedChatState => ({
        chats: state.chats.slice(0, MAX_PERSISTED_CHATS).map((chat) => ({
          ...chat,
          messages: chat.messages.map(stripObjectUrls),
        })),
      }),

      // Runs once, when rehydrate() reads storage back. Repairs anything the
      // stored snapshot can't carry across a reload, and re-joins the two
      // storage tiers: the chats from localStorage, this tab's selected id
      // from sessionStorage.
      //
      // Validating that id is now load-bearing rather than merely defensive.
      // The pointer and the list are written to different places at different
      // times, so they genuinely go out of sync: history cleared in another
      // tab, a chat pushed past MAX_PERSISTED_CHATS, or a hand-edited value.
      // An activeChatId with no matching chat leaves activeChat() undefined,
      // which silently no-ops every send (see the `if (!activeChat) return`
      // guard in useAgent/useChat). Falling back to the newest chat means a
      // stale pointer costs the user a click, not a dead input box.
      merge: (persisted, current): ChatState => {
        const stored = (persisted ?? {}) as Partial<PersistedChatState>;

        const chats = (stored.chats ?? current.chats).map((chat) => ({
          ...chat,
          messages: chat.messages.map(settleInFlightMessage),
        }));

        if (chats.length === 0) {
          const chat = createEmptyChat();
          return { ...current, chats: [chat], activeChatId: chat.id };
        }

        const tabChatId = readActiveChatId();
        const activeChatId =
          tabChatId && chats.some((chat) => chat.id === tabChatId)
            ? tabChatId
            : chats[0].id;

        return { ...current, chats, activeChatId };
      },

      // localStorage doesn't exist during server rendering. Hydrating here
      // would desync server and client markup, so it's triggered explicitly
      // after mount — see hooks/use-chat-store-hydration.ts.
      skipHydration: true,
    }
  )
);

/**
 * Mirrors `activeChatId` out to sessionStorage whenever it changes.
 *
 * A subscription rather than a `writeActiveChatId` call inside each of
 * `startNewChat` / `setActiveChat` / `clearAllChats`: those three are the
 * only writers today, but nothing stops a fourth from being added, and a
 * mirror that can silently fall out of date is worse than no mirror. One
 * listener covers every path by construction.
 *
 * This also fires during rehydration, when `merge` resolves the id — so a
 * fresh tab that fell back to the newest chat records that choice
 * immediately, instead of re-deriving it on every reload.
 */
useChatStore.subscribe((state, previous) => {
  if (state.activeChatId !== previous.activeChatId) {
    writeActiveChatId(state.activeChatId);
  }
});

/**
 * Flattens every message in `chats` into archive rows.
 *
 * Exported because hydration needs the same shaping to backfill chats that
 * were saved before the archive existed — see hooks/use-chat-store-hydration.
 */
export function toStoredMessages(chats: Chat[]): StoredMessage[] {
  const rows: StoredMessage[] = [];
  for (const chat of chats) {
    for (const message of chat.messages) {
      // Empty content is a bubble waiting on a response. Archiving it would
      // put a blank row in the index that no search can ever match, and it
      // gets rewritten with real content moments later anyway.
      if (!message.content) continue;

      // Skip anything still in flight. Without this, a streaming answer
      // rewrites its own archive row on every flush — dozens of IndexedDB
      // writes for one message, all but the last immediately stale. The
      // message lands here once, when its status settles to "sent" (or
      // "error", which keeps a partial answer searchable).
      if (message.status && IN_FLIGHT_STATUSES.has(message.status)) continue;
      rows.push({
        id: message.id,
        chatId: chat.id,
        chatTitle: chat.title,
        role: message.role,
        content: message.content,
        createdAt: message.createdAt,
      });
    }
  }
  return rows;
}

/**
 * Mirrors new and changed messages into the IndexedDB archive.
 *
 * A subscription for the same reason as `activeChatId` above: every code path
 * that touches messages goes through `set`, so one listener can't be
 * bypassed by a mutation added later.
 *
 * The diff is by reference, not by value. Zustand's updates are immutable, so
 * a chat whose `messages` array is identical to the previous one provably
 * didn't change, and a message object that is `===` its predecessor didn't
 * either. That reduces a keystroke-rate subscription to a handful of pointer
 * comparisons, and means the common case — one new message — writes exactly
 * one record rather than re-archiving the entire history.
 */
useChatStore.subscribe((state, previous) => {
  if (state.chats === previous.chats) return;

  const previousChats = new Map(previous.chats.map((chat) => [chat.id, chat]));
  const changed: Chat[] = [];

  for (const chat of state.chats) {
    const before = previousChats.get(chat.id);
    if (before && before.messages === chat.messages) continue;

    if (!before) {
      changed.push(chat);
      continue;
    }

    const beforeMessages = new Map(before.messages.map((message) => [message.id, message]));
    const newOrEdited = chat.messages.filter(
      (message) => beforeMessages.get(message.id) !== message
    );
    if (newOrEdited.length > 0) {
      changed.push({ ...chat, messages: newOrEdited });
    }
  }

  if (changed.length === 0) return;
  void putMessages(toStoredMessages(changed));
});

/**
 * Wipes persisted chat state. Exported separately from `clearAllChats` (which
 * only resets the in-memory store) so a caller can do both:
 * `useChatStore.getState().clearAllChats(); clearPersistedChats();`
 *
 * Clears this tab's selection too — the chats it could point at are gone, and
 * leaving a dangling id behind just makes `merge` do the fallback work on the
 * next load. Same for cached media: nothing references it any more, and it's
 * by far the largest thing this app puts on disk.
 */
export function clearPersistedChats(): void {
  useChatStore.persist.clearStorage();
  clearActiveChatId();
  void clearMediaCache();
  void clearMessages();
}
