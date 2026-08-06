/**
 * MessageBubble — renders a single Message.
 *
 * Purely presentational: takes a Message and renders it. Markdown
 * rendering is isolated here so switching/upgrading the markdown renderer
 * later never touches ChatWindow or the data layer.
 */
"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bot,
  User,
  AlertCircle,
  Clock,
  FileText,
  Globe,
  ImageOff,
  Sparkles,
} from "lucide-react";
import type { AgentTool, Message } from "@/types";
import { useAttachmentUrl } from "@/hooks/use-attachment-url";
import { cn } from "@/lib/utils";

/**
 * How each agent tool is labelled in the bubble footer. Without this the
 * only visible difference between an answer grounded in the user's PDFs and
 * one improvised from the model's own knowledge is the text itself — which
 * is exactly the case where a silent misroute matters most.
 */
const TOOL_LABELS: Record<AgentTool, { icon: typeof FileText; label: string }> = {
  search_knowledge_base: { icon: FileText, label: "From your documents" },
  call_external_api: { icon: Globe, label: "From a live API" },
  answer_directly: { icon: Sparkles, label: "From general knowledge" },
};

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isError = message.status === "error";
  // Distinct from pending on purpose: nothing is in flight, so animated dots
  // would misrepresent it as work in progress rather than work not yet
  // started. The user needs to know the send is deferred, not slow.
  const isQueued = message.status === "queued";
  const isPending =
    (message.status === "sending" ||
      message.status === "searching" ||
      message.status === "generating" ||
      message.status === "transcribing" ||
      message.status === "analyzing") &&
    !message.content;

  return (
    <div
      className={cn(
        "flex w-full gap-3 py-3",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Bot className="size-4" />
        </div>
      )}

      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm sm:max-w-[70%]",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground",
          isError && "border border-destructive/40 bg-destructive/10 text-destructive"
        )}
      >
        {isPending ? (
          <TypingIndicator status={message.status} />
        ) : isQueued ? (
          <QueuedNotice />
        ) : isError ? (
          <div className="flex items-center gap-2">
            <AlertCircle className="size-4 shrink-0" />
            <span>{message.content}</span>
          </div>
        ) : isUser ? (
          <div className="flex flex-col gap-2">
            {message.attachments?.map((attachment) =>
              attachment.type === "audio" ? (
                <AudioAttachmentPreview key={attachment.id} attachment={attachment} />
              ) : (
                <ImageAttachmentPreview key={attachment.id} attachment={attachment} />
              )
            )}
            <p className="whitespace-pre-wrap">{message.content}</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {message.attachments
              ?.filter((attachment) => attachment.type === "audio")
              .map((attachment) => (
                <AudioAttachmentPreview key={attachment.id} attachment={attachment} showTranscript />
              ))}
            <div className="markdown-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
            {message.sources && message.sources.length > 0 && (
              <SourceList sources={message.sources} />
            )}
            {message.toolUsed && <ToolBadge tool={message.toolUsed} />}
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
          <User className="size-4" />
        </div>
      )}
    </div>
  );
}

function ToolBadge({ tool }: { tool: AgentTool }) {
  const { icon: Icon, label } = TOOL_LABELS[tool];
  return (
    <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
      <Icon className="size-3" />
      <span>{label}</span>
    </div>
  );
}

/**
 * Shown for a message the service worker is holding until the network comes
 * back (see lib/outbox.ts). Static rather than animated: the request isn't
 * running, and a spinner would promise progress that isn't happening.
 */
function QueuedNotice() {
  return (
    <span className="flex items-center gap-2 py-1 text-muted-foreground" role="status">
      <Clock className="size-3.5 shrink-0" />
      <span>Waiting for connection — this will send when you&apos;re back online.</span>
    </span>
  );
}

function TypingIndicator({ status }: { status: Message["status"] }) {
  const label =
    status === "searching"
      ? "Searching documents..."
      : status === "generating"
        ? "Generating answer..."
        : status === "transcribing"
          ? "Transcribing..."
          : status === "analyzing"
            ? "Analyzing..."
            : "Assistant is thinking...";

  return (
    <span className="flex items-center gap-2 py-1 text-muted-foreground" role="status">
      <span className="flex items-center gap-1">
        <Dot />
        <Dot delayClassName="[animation-delay:150ms]" />
        <Dot delayClassName="[animation-delay:300ms]" />
      </span>
      <span>{label}</span>
    </span>
  );
}

/**
 * Renders an attached image, from the live object URL when the upload
 * happened in this page session and from the Cache API after a reload.
 *
 * Split out of the map above because resolving the cache is asynchronous and
 * therefore needs a hook, which can't be called from inside a callback.
 */
function ImageAttachmentPreview({
  attachment,
}: {
  attachment: NonNullable<Message["attachments"]>[number];
}) {
  const src = useAttachmentUrl(attachment);

  if (!src) {
    return <ExpiredImageNotice name={attachment.name} />;
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element -- object URLs aren't compatible with next/image's optimizer
    <img
      src={src}
      alt={attachment.name ?? "Attached image"}
      className="max-h-64 w-full max-w-xs rounded-lg object-cover"
    />
  );
}

function AudioAttachmentPreview({
  attachment,
  showTranscript = false,
}: {
  attachment: NonNullable<Message["attachments"]>[number];
  showTranscript?: boolean;
}) {
  const src = useAttachmentUrl(attachment);

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border/60 bg-background/40 p-2">
      <div className="flex items-center gap-1.5 text-xs font-medium">
        <span aria-hidden>🎵</span>
        <span className="truncate">{attachment.name ?? "audio"}</span>
      </div>
      {src ? (
        <audio src={src} controls className="h-8 w-full max-w-xs" />
      ) : (
        <p className="text-xs text-muted-foreground/70">
          Playback unavailable after reload
        </p>
      )}
      {showTranscript && attachment.transcript && (
        <div className="mt-1 border-t border-border/60 pt-1.5">
          <p className="text-xs font-medium text-muted-foreground/80">Transcript</p>
          <p className="mt-0.5 max-h-32 overflow-y-auto text-xs text-muted-foreground whitespace-pre-wrap">
            {attachment.transcript}
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Stands in for an image whose object URL didn't survive a page reload (the
 * store blanks those on persist — see store/chat-store.ts). Showing the
 * filename keeps the turn readable instead of leaving a broken-image icon
 * above an answer that still refers to "the image you sent".
 */
function ExpiredImageNotice({ name }: { name?: string }) {
  return (
    <div className="flex items-center gap-1.5 rounded-lg border border-border/40 bg-background/30 px-2 py-1.5 text-xs opacity-80">
      <ImageOff className="size-3.5 shrink-0" />
      <span className="truncate">{name ?? "Image"} · not kept after reload</span>
    </div>
  );
}

function SourceList({ sources }: { sources: NonNullable<Message["sources"]> }) {
  return (
    <div className="mt-1 flex flex-col gap-1 border-t border-border/60 pt-2 text-xs text-muted-foreground">
      <span className="font-medium text-muted-foreground/80">Sources</span>
      <ul className="flex flex-col gap-1">
        {sources.map((source) => (
          <li key={source.chunkId} className="flex items-center gap-1.5">
            <FileText className="size-3.5 shrink-0" />
            <span className="truncate">{source.filename}</span>
            <span className="shrink-0 text-muted-foreground/70">
              &middot; Page {source.page}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Dot({ delayClassName }: { delayClassName?: string }) {
  return (
    <span
      className={cn(
        "size-1.5 animate-bounce rounded-full bg-muted-foreground/60",
        delayClassName
      )}
    />
  );
}
