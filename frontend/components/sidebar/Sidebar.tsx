/**
 * Sidebar — navigation for chats plus links/placeholders for other features.
 *
 * Reads/writes chat list state via useChatStore directly (list navigation
 * is simple enough not to warrant its own hook), but delegates all message
 * sending to useChat/ChatService elsewhere — this component only ever
 * starts/selects chats, it never talks to the API.
 *
 * "Documents" now links to /documents (Phase 4A: upload + ingestion).
 * "Images" is still rendered disabled — the nav structure for that
 * feature exists so enabling it later is a matter of wiring a route, not
 * restructuring the sidebar.
 *
 * Search replaces the chat list while a query is active rather than sitting
 * beside it. The two answer the same question — "which conversation do I
 * want" — and showing both would have the user scanning two lists to decide.
 * Results come from the IndexedDB archive, not from `chats`, so they include
 * conversations that have aged out of the persisted list.
 */
"use client";

import { useState } from "react";
import Link from "next/link";
import {
  FileText,
  Image as ImageIcon,
  MessageSquare,
  Plus,
  Search,
  Video,
  X,
} from "lucide-react";
import { useChatStore } from "@/store/chat-store";
import { useMessageSearch } from "@/hooks/use-message-search";
import type { StoredMessage } from "@/lib/message-db";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface SidebarProps {
  className?: string;
  onNavigate?: () => void;
}

export function Sidebar({ className, onNavigate }: SidebarProps) {
  const chats = useChatStore((state) => state.chats);
  const activeChatId = useChatStore((state) => state.activeChatId);
  const startNewChat = useChatStore((state) => state.startNewChat);
  const setActiveChat = useChatStore((state) => state.setActiveChat);

  const [query, setQuery] = useState("");
  const { results, isSearching, isActive } = useMessageSearch(query);

  const openChat = (chatId: string) => {
    setActiveChat(chatId);
    onNavigate?.();
  };

  /**
   * A result can name a chat the store no longer holds — the archive keeps
   * messages past MAX_PERSISTED_CHATS. Clicking one of those would call
   * setActiveChat with an id no chat matches, leaving activeChat() undefined
   * and every send silently no-op'ing (the `if (!activeChat) return` guard in
   * useAgent). Rendering it as non-clickable is the honest option until
   * restoring an archived chat is a thing this app can do.
   */
  const liveChatIds = new Set(chats.map((chat) => chat.id));

  return (
    <aside
      className={cn(
        "flex h-full w-72 flex-col border-r border-border bg-sidebar",
        className
      )}
    >
      <div className="p-3">
        <Button
          variant="outline"
          className="w-full justify-start gap-2"
          onClick={() => {
            startNewChat();
            onNavigate?.();
          }}
        >
          <Plus className="size-4" />
          New Chat
        </Button>
      </div>

      <div className="px-3 pb-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-sidebar-foreground/40" />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search messages..."
            aria-label="Search messages across all chats"
            className="w-full rounded-lg border border-border bg-background/60 py-1.5 pl-8 pr-7 text-sm outline-none placeholder:text-sidebar-foreground/40 focus-visible:ring-1 focus-visible:ring-ring"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              aria-label="Clear search"
              className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-sidebar-foreground/50 transition-colors hover:text-sidebar-foreground"
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>
      </div>

      <nav className="flex flex-col gap-0.5 px-3">
        {isActive ? (
          <>
            <SidebarSectionLabel>
              {isSearching ? "Searching..." : `Results (${results.length})`}
            </SidebarSectionLabel>
            <ScrollArea className="max-h-64">
              <div className="flex flex-col gap-0.5 pr-2">
                {!isSearching && results.length === 0 && (
                  <p className="px-2.5 py-2 text-sm text-sidebar-foreground/50">
                    No messages match &ldquo;{query.trim()}&rdquo;.
                  </p>
                )}
                {results.map((result) => (
                  <SearchResult
                    key={result.id}
                    result={result}
                    query={query.trim()}
                    isOpenable={liveChatIds.has(result.chatId)}
                    onOpen={() => openChat(result.chatId)}
                  />
                ))}
              </div>
            </ScrollArea>
          </>
        ) : (
          <>
            <SidebarSectionLabel>Chat History</SidebarSectionLabel>
            <ScrollArea className="max-h-64">
              <div className="flex flex-col gap-0.5 pr-2">
                {chats.map((chat) => (
                  <button
                    key={chat.id}
                    onClick={() => openChat(chat.id)}
                    className={cn(
                      "flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-sidebar-foreground/80 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                      chat.id === activeChatId &&
                        "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                    )}
                  >
                    <MessageSquare className="size-4 shrink-0 opacity-70" />
                    <span className="truncate">{chat.title || "New Chat"}</span>
                  </button>
                ))}
              </div>
            </ScrollArea>
          </>
        )}
      </nav>

      <div className="mt-2 flex flex-col gap-0.5 px-3">
        <SidebarSectionLabel>Library</SidebarSectionLabel>
        <Link
          href="/documents"
          onClick={() => onNavigate?.()}
          className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-sidebar-foreground/80 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
        >
          <FileText className="size-4 shrink-0 opacity-70" />
          <span>Documents</span>
        </Link>
        <Link
          href="/streaming"
          onClick={() => onNavigate?.()}
          className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-sidebar-foreground/80 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
        >
          <Video className="size-4 shrink-0 opacity-70" />
          <span>Live Streaming</span>
        </Link>
        <SidebarDisabledItem icon={ImageIcon} label="Images" />
      </div>

      <div className="mt-auto p-3 text-xs text-sidebar-foreground/50">
        Phase 7 · Real-time streaming
      </div>
    </aside>
  );
}

/**
 * One search hit: which chat it came from, and enough of the message to
 * recognise it.
 *
 * The snippet is centred on the match rather than being the first 80
 * characters, because a hit halfway through a long answer would otherwise
 * show text that doesn't contain the search term at all — which reads as a
 * bug even when the match is real.
 */
function SearchResult({
  result,
  query,
  isOpenable,
  onOpen,
}: {
  result: StoredMessage;
  query: string;
  isOpenable: boolean;
  onOpen: () => void;
}) {
  const snippet = buildSnippet(result.content, query);

  const body = (
    <>
      <span className="flex items-center gap-1.5 text-xs font-medium text-sidebar-foreground/70">
        <MessageSquare className="size-3 shrink-0 opacity-70" />
        <span className="truncate">{result.chatTitle || "New Chat"}</span>
        <span className="ml-auto shrink-0 opacity-60">
          {result.role === "user" ? "You" : "AI"}
        </span>
      </span>
      <span className="line-clamp-2 text-xs text-sidebar-foreground/60">{snippet}</span>
    </>
  );

  if (!isOpenable) {
    return (
      <div
        title="This conversation is no longer in your chat list"
        className="flex flex-col gap-1 rounded-lg px-2.5 py-2 text-left opacity-50"
      >
        {body}
      </div>
    );
  }

  return (
    <button
      onClick={onOpen}
      className="flex flex-col gap-1 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-sidebar-accent"
    >
      {body}
    </button>
  );
}

/** Window of content around the first match, with ellipses where cut. */
function buildSnippet(content: string, query: string, radius = 40): string {
  const index = content.toLowerCase().indexOf(query.toLowerCase());
  if (index === -1) return content.slice(0, radius * 2);

  const start = Math.max(0, index - radius);
  const end = Math.min(content.length, index + query.length + radius);

  return `${start > 0 ? "…" : ""}${content.slice(start, end)}${
    end < content.length ? "…" : ""
  }`;
}

function SidebarSectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="px-2.5 py-1.5 text-xs font-medium uppercase tracking-wide text-sidebar-foreground/50">
      {children}
    </span>
  );
}

function SidebarDisabledItem({
  icon: Icon,
  label,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
}) {
  return (
    <button
      disabled
      title="Coming soon"
      className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-sidebar-foreground/40 cursor-not-allowed"
    >
      <Icon className="size-4 shrink-0" />
      <span>{label}</span>
      <span className="ml-auto rounded-full bg-sidebar-accent px-1.5 py-0.5 text-[10px] font-medium text-sidebar-foreground/50">
        Soon
      </span>
    </button>
  );
}
