/**
 * ChatLayout — top-level composition: Header + Sidebar + main chat area.
 *
 * This is the only component that wires layout-level concerns together
 * (sidebar open/close state for mobile, and passing useAgent's state down
 * to ChatWindow/ChatInput). Individual pieces (Header, Sidebar, ChatWindow,
 * ChatInput) stay independent and reusable — e.g. Sidebar could be reused
 * on a future /documents page without dragging chat logic along with it.
 *
 * Responsive behavior:
 *  - Desktop (md and up): sidebar is always visible, static in the flex row.
 *  - Mobile: sidebar is hidden by default and slides in as an overlay,
 *    toggled from the Header's menu button.
 */
"use client";

import { useState } from "react";
import { Header } from "./Header";
import { AnnouncementBanner } from "./AnnouncementBanner";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { ChatInput } from "@/components/chat/ChatInput";
import { useChat } from "@/hooks/use-chat";
import { useChatStore } from "@/store/chat-store";
import { useChatStoreHydration } from "@/hooks/use-chat-store-hydration";
import { useOutboxListener } from "@/hooks/use-outbox-listener";
import { useAgent } from "@/hooks/use-agent";
import { useVision } from "@/hooks/use-vision";
import { useAudio } from "@/hooks/use-audio";
import { cn } from "@/lib/utils";

export function ChatLayout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [attachedImage, setAttachedImage] = useState<File | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [attachedAudio, setAttachedAudio] = useState<File | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);

  // Pulls saved chats out of localStorage after mount (the store defers
  // hydration so server and client render the same initial markup). The flag
  // keeps ChatWindow from flashing its "start a conversation" empty state at
  // someone who actually has saved history.
  const isHydrated = useChatStoreHydration();

  // Fills in bubbles the service worker delivered after a reconnect. Mounted
  // here because it's the one component guaranteed to be alive whenever a
  // chat is on screen.
  useOutboxListener();

  // Keys this tab's per-chat session state — the unsent draft in ChatInput
  // and the scroll position in ChatWindow. Withheld until hydration finishes:
  // before that the store is still showing the placeholder chat it was
  // created with, and anything written under that id would be orphaned the
  // moment the real history loaded.
  const activeChatId = useChatStore((state) => state.activeChatId);
  const sessionChatId = isHydrated ? activeChatId : null;

  // useChat is kept for `messages` only — the store-backed transcript. Text
  // sends now go through useAgent instead of useChat.sendMessage.
  const { messages } = useChat();
  const { askAgent, isSending } = useAgent();
  const { analyzeImage, isUploading, uploadProgress } = useVision();
  const {
    analyzeAudio,
    isProcessing: isProcessingAudio,
    uploadProgress: audioUploadProgress,
  } = useAudio();

  // Routes a send to vision or audio analysis when a file is attached, and
  // otherwise to the agent — which makes its own choice between searching
  // ingested documents, calling an external API, and answering directly
  // (see backend/agents/assistant_agent.py). This branching is intentionally
  // kept here — the one place that composes every flow — rather than inside
  // ChatInput, which stays unaware of all of them.
  //
  // Attachments still win over the agent: an image or audio file is an
  // explicit signal about which pipeline the user wants, so there's nothing
  // for a router to infer.
  const handleSend = (text: string) => {
    if (attachedImage) {
      const image = attachedImage;
      setAttachedImage(null);
      void analyzeImage(image, text);
      return;
    }
    if (attachedAudio) {
      const audio = attachedAudio;
      setAttachedAudio(null);
      void analyzeAudio(audio, text);
      return;
    }
    void askAgent(text);
  };

  const isBusy = isSending || isUploading || isProcessingAudio;

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-background">
      <Header onToggleSidebar={() => setIsSidebarOpen((open) => !open)} />

      {/* Above the sidebar/main row rather than inside <main>, so it spans
          the full width and pushes both columns down instead of overlapping
          either. Renders nothing once dismissed, so it costs no height. */}
      <AnnouncementBanner />

      <div className="relative flex flex-1 overflow-hidden">
        {/* Desktop sidebar */}
        <Sidebar className="hidden md:flex" />

        {/* Mobile sidebar overlay */}
        {isSidebarOpen && (
          <div
            className="absolute inset-0 z-40 bg-black/40 md:hidden"
            onClick={() => setIsSidebarOpen(false)}
            aria-hidden
          />
        )}
        <Sidebar
          className={cn(
            "absolute inset-y-0 left-0 z-50 flex transition-transform duration-200 md:hidden",
            isSidebarOpen ? "translate-x-0" : "-translate-x-full"
          )}
          onNavigate={() => setIsSidebarOpen(false)}
        />

        <main className="flex flex-1 flex-col overflow-hidden">
          <ChatWindow
            messages={messages}
            isHydrated={isHydrated}
            chatId={sessionChatId}
          />
          {imageError && (
            <p className="mx-auto mb-1 max-w-3xl px-6 text-center text-xs text-destructive">
              {imageError}
            </p>
          )}
          {audioError && (
            <p className="mx-auto mb-1 max-w-3xl px-6 text-center text-xs text-destructive">
              {audioError}
            </p>
          )}
          <ChatInput
            onSend={handleSend}
            disabled={isBusy}
            chatId={sessionChatId}
            attachedImage={attachedImage}
            onSelectImage={(file) => {
              setImageError(null);
              setAttachedImage(file);
            }}
            onRemoveImage={() => setAttachedImage(null)}
            onImageError={setImageError}
            isUploading={isUploading || isProcessingAudio}
            uploadProgress={attachedAudio ? audioUploadProgress : uploadProgress}
            attachedAudio={attachedAudio}
            onSelectAudio={(file) => {
              setAudioError(null);
              setAttachedAudio(file);
            }}
            onRemoveAudio={() => setAttachedAudio(null)}
            onAudioError={setAudioError}
          />
        </main>
      </div>
    </div>
  );
}
