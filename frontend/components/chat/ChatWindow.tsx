/**
 * ChatWindow — renders the message list for the active chat, restores where
 * this tab was last scrolled to, and follows new messages from there.
 *
 * Takes `messages` as a prop rather than reading the store itself, so it
 * stays a straightforward list renderer that ChatLayout composes with
 * useChat's output — easy to reason about and easy to test in isolation.
 * Scroll behaviour is delegated to useChatScroll for the same reason: this
 * component decides what the transcript looks like, not how it moves.
 */
"use client";

import { MessageSquareText } from "lucide-react";
import { MessageBubble } from "./MessageBubble";
import { useChatScroll } from "@/hooks/use-chat-scroll";
import type { Message } from "@/types";

interface ChatWindowProps {
  messages: Message[];
  /**
   * False until saved chats have been read back from localStorage. An empty
   * `messages` means "no messages yet" only once this is true — before that
   * it just means the store hasn't been restored, and showing the empty
   * state would flash "start a conversation" at someone mid-conversation.
   */
  isHydrated?: boolean;
  /**
   * Which chat's scroll position to restore. Null keeps scrolling ephemeral —
   * same gating as the draft in ChatInput, see ChatLayout.
   */
  chatId: string | null;
}

export function ChatWindow({ messages, isHydrated = true, chatId }: ChatWindowProps) {
  const { containerRef, handleScroll } = useChatScroll(chatId, messages);

  if (messages.length === 0 && !isHydrated) {
    return <div className="flex-1" aria-busy="true" />;
  }

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 px-6 text-center text-muted-foreground">
        <MessageSquareText className="size-8 opacity-50" />
        <p className="text-sm">Start a conversation with your AI workspace.</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto px-3 sm:px-6"
    >
      <div className="mx-auto flex max-w-3xl flex-col divide-y divide-transparent">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
      </div>
    </div>
  );
}
