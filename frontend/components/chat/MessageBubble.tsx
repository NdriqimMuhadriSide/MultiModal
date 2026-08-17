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
  ChevronRight,
  Clock,
  FileText,
  Globe,
  Image as ImageIcon,
  ImageOff,
  Layers,
  Sparkles,
} from "lucide-react";
import type { AgentStepView, AgentTool, Message } from "@/types";
import { useAttachmentUrl } from "@/hooks/use-attachment-url";
import { cn } from "@/lib/utils";

/**
 * How each agent tool is labelled in the bubble footer. Without this the
 * only visible difference between an answer grounded in the user's PDFs and
 * one improvised from the model's own knowledge is the text itself — which
 * is exactly the case where a silent misroute matters most.
 *
 * Typed as a total Record so adding a tool to AgentTool without a label here
 * is a compile error rather than a blank badge. ToolBadge still guards at
 * runtime, because the value arrives from the network and the union is only
 * a claim about what the backend sends.
 */
type ToolLabel = { icon: typeof FileText; label: string };

const TOOL_LABELS: Record<AgentTool, ToolLabel> = {
  research_documents: { icon: FileText, label: "From your documents" },
  read_image: { icon: ImageIcon, label: "From your image" },
  call_external_api: { icon: Globe, label: "From a live API" },
  answer_directly: { icon: Sparkles, label: "From general knowledge" },
  // Wording matters: the answer rests on more than one specialist, and
  // naming either alone would tell the reader it rests on less than it does.
  // The trace below the badge is where they can see which.
  multiple_specialists: {
    icon: Layers,
    label: "From your documents and image",
  },
};

/**
 * How each tool reads while it is still running, for the live activity line.
 *
 * A delegating run spends several seconds producing no prose at all, so the
 * bubble would otherwise sit empty until the whole answer landed. Naming the
 * tool the agent is on turns that silence into progress a person can follow.
 *
 * Falls back to the raw tool name rather than hiding: an unlabelled step is
 * still more informative than a blank bubble, and this list is the thing most
 * likely to lag a backend that gained a tool.
 */
const STEP_LABELS: Record<string, string> = {
  research_documents: "Asking the document specialist",
  read_image: "Reading the image",
  search_knowledge_base: "Searching your documents",
  search: "Searching your documents",
  list_documents: "Listing your documents",
  inspect_image: "Looking at the image",
  read_text: "Reading the text in the image",
  call_external_api: "Checking a live source",
  finish: "Writing the answer",
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
  // The `!content` guard is what makes streaming work here without a special
  // case: a streaming bubble shows the indicator only in the gap before its
  // first token, then renders text that keeps growing.
  const isPending =
    (message.status === "sending" ||
      message.status === "streaming" ||
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
            {/*
              While the run is going this is the only thing on screen, so it
              reads as an activity line; once the answer lands it collapses
              to a summary the reader can open. Same data either way — a
              supervised answer is assembled out of work nobody asked to see,
              and this is what makes it checkable.
            */}
            {message.steps && message.steps.length > 0 && (
              <AgentTrace
                steps={message.steps}
                live={message.status === "streaming"}
              />
            )}
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
  // `tool` is typed, but it came off the wire — a backend that adds a tool
  // before this file learns about it would otherwise destructure undefined
  // and take the whole message list down with it. A missing badge is a far
  // better failure than an unrenderable conversation.
  const entry: ToolLabel | undefined = TOOL_LABELS[tool];
  if (!entry) return null;

  const { icon: Icon, label } = entry;
  return (
    <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
      <Icon className="size-3" />
      <span>{label}</span>
    </div>
  );
}

function stepLabel(tool: string): string {
  return STEP_LABELS[tool] ?? tool;
}

/**
 * The supervisor's trace: what it did, and what it asked a specialist to do.
 *
 * Two modes from one component, because they are the same information at
 * different moments. While `live`, the last step is the activity line — the
 * agent is doing that right now, and there is no answer yet to read. Once
 * the answer arrives it becomes a one-line summary that opens on click, so a
 * finished conversation is not a wall of tool calls.
 *
 * `<details>` rather than React state on purpose: the open/closed flag is
 * per-bubble UI with no bearing on anything else, and the browser already
 * tracks it correctly — including for a keyboard user, which a div with an
 * onClick would not.
 */
function AgentTrace({ steps, live }: { steps: AgentStepView[]; live: boolean }) {
  if (live) {
    const current = steps[steps.length - 1];
    return (
      <div
        className="flex items-center gap-2 text-[11px] text-muted-foreground"
        role="status"
      >
        <span className="size-1.5 animate-pulse rounded-full bg-current" />
        <span>{stepLabel(current.tool)}…</span>
      </div>
    );
  }

  // The `finish` step is the answer itself, which is already rendered above
  // it. Counting it here would tell the reader the agent took one more step
  // than it usefully did.
  const worked = steps.filter((step) => step.tool !== "finish");
  if (worked.length === 0) return null;

  return (
    <details className="group text-[11px] text-muted-foreground">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 hover:text-foreground">
        <ChevronRight className="size-3 transition-transform group-open:rotate-90" />
        <span>
          {worked.length} step{worked.length === 1 ? "" : "s"}
        </span>
      </summary>
      <ol className="mt-1.5 flex flex-col gap-1.5 border-l border-border pl-3">
        {worked.map((step, index) => (
          <TraceStep key={index} step={step} />
        ))}
      </ol>
    </details>
  );
}

/**
 * One step, and — when it was a delegation — the steps behind it.
 *
 * Recursive because the backend's trace is: nothing in the agent loop caps
 * how deep delegation goes, so a renderer that handled exactly one level
 * would silently stop showing work the first time that changed.
 */
function TraceStep({ step }: { step: AgentStepView }) {
  return (
    <li className="flex flex-col gap-1">
      <span className="text-foreground/70">{stepLabel(step.tool)}</span>
      {step.children.length > 0 && (
        <ol className="flex flex-col gap-1 border-l border-border pl-3">
          {step.children.map((child, index) => (
            <TraceStep key={index} step={child} />
          ))}
        </ol>
      )}
    </li>
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
            <span className="shrink-0">{source.filename}</span>
            <span className="shrink-0 text-muted-foreground/70">
              &middot; Page {source.page}
            </span>
            {/* The section is the part that tells you *what* was cited, so it
                takes the flexible width and truncates last. */}
            {source.section && (
              <span className="truncate text-muted-foreground/70">
                &middot; {source.section}
              </span>
            )}
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
