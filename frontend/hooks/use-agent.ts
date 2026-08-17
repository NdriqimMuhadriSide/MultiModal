/**
 * useAgent — orchestrates sending a message to the supervisor agent.
 *
 * Same boundary role as useChat/useRag (UI never calls AgentService or
 * mutates the store directly), but replaces the manual "General AI vs My
 * Documents" choice: the backend decides per message whether to answer
 * directly, ask a specialist about the documents or the conversation's
 * image, or call a live external API — and can use more than one in a turn.
 *
 * What the answer rests on comes back in `tool_used` and is stored on the
 * message as `toolUsed`, so the bubble can show where it came from. When a
 * document specialist was involved, its citations follow in `sources` and
 * are stored the way useRag stores them, so the claim on the badge is one
 * the user can actually check.
 *
 * WHY THIS NO LONGER BUFFERS TOKENS
 *
 * The stream used to carry `delta` frames, and this hook batched them on an
 * interval so a long answer did not re-serialize localStorage once per
 * token. There are no deltas any more: a supervisor's answer is whole inside
 * its final action by the time the backend sees it, so it arrives in one
 * `answer` frame and the buffering has nothing left to buffer.
 *
 * What arrives in its place is one `step` frame per completed turn, appended
 * to the message as they land. That is a handful of writes for a whole run
 * rather than hundreds, so it needs no batching of its own — and it means
 * the bubble can show what the agent is *doing* during the several seconds a
 * delegating run spends thinking, which an empty bubble waiting on a first
 * token could not.
 */
"use client";

import { useCallback, useState } from "react";
import { useChatStore } from "@/store/chat-store";
import { AgentService } from "@/services/agent-service";
import { canQueue, enqueue, requestFlush } from "@/lib/outbox";
import { newId } from "@/lib/uuid";
import { ApiError } from "@/types/api";
import type { AgentStepView, Message } from "@/types";

/**
 * True only when the request never reached the server. ApiError carries a
 * null status for exactly that case; anything with a status means the backend
 * answered, and an answer — even a 500 — is not something to retry blindly.
 */
function isNetworkError(err: unknown): boolean {
  return err instanceof ApiError && err.status === null;
}

function createMessage(role: Message["role"], content: string, status?: Message["status"]): Message {
  return {
    id: newId(),
    role,
    content,
    createdAt: new Date().toISOString(),
    status,
  };
}

function toUserFacingError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === null) {
      return "Backend Offline. Check that the API server is running and try again.";
    }
    if (err.status === 503) {
      return "The assistant isn't configured. Check the backend's API keys and try again.";
    }
    if (err.status >= 500) {
      return "The assistant ran into a problem answering that. Please try again.";
    }
    if (err.status === 429) {
      return "Too many requests right now. Please wait a moment and try again.";
    }
    if (err.status >= 400) {
      return "That message couldn't be processed. Please rephrase and try again.";
    }
  }
  return "Something went wrong while contacting the assistant.";
}

export function useAgent() {
  const activeChat = useChatStore((state) => state.activeChat());
  const appendMessage = useChatStore((state) => state.appendMessage);
  const updateMessage = useChatStore((state) => state.updateMessage);
  const setConversationId = useChatStore((state) => state.setConversationId);

  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const askAgent = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !activeChat || isSending) return;

      setError(null);

      const chatId = activeChat.id;
      const userMessage = createMessage("user", trimmed, "sent");
      appendMessage(chatId, userMessage);

      const pendingAssistant = createMessage("assistant", "", "sending");
      appendMessage(chatId, pendingAssistant);

      setIsSending(true);

      // Steps accumulate here rather than in the store, so each arriving
      // frame is one write of the whole list instead of a read-modify-write
      // against persisted state. A run produces a handful of them, so this
      // needs none of the interval batching token deltas did.
      const steps: AgentStepView[] = [];
      let receivedAny = false;

      try {
        let streamError = false;

        for await (const event of AgentService.streamAsk(
          trimmed,
          activeChat.conversationId
        )) {
          if (event.type === "start") {
            if (!activeChat.conversationId) {
              setConversationId(chatId, event.conversation_id);
            }
          } else if (event.type === "step") {
            // The run's live progress. Only the supervisor's own steps are
            // collected: a specialist's (depth > 0) already arrive nested
            // under the delegation that caused them in the parent step's
            // `children`, and appending them here as well would show the
            // same work twice — once as something the supervisor did, and
            // once as something it delegated.
            if (event.depth === 0) {
              steps.push(event.step);
              updateMessage(chatId, pendingAssistant.id, {
                status: "streaming",
                steps: [...steps],
              });
            }
          } else if (event.type === "tool") {
            updateMessage(chatId, pendingAssistant.id, {
              status: "streaming",
              toolUsed: event.tool,
            });
          } else if (event.type === "sources") {
            // Before the answer. The bubble already says where it came
            // from; this is the part that lets the user check it.
            updateMessage(chatId, pendingAssistant.id, {
              sources: event.sources,
            });
          } else if (event.type === "answer") {
            // Whole, in one frame — see this module's header for why there
            // is nothing to stream token by token here.
            receivedAny = true;
            updateMessage(chatId, pendingAssistant.id, {
              content: event.content,
            });
          } else if (event.type === "error") {
            // A provider failure after the response had already committed as
            // 200. Whatever arrived stays on screen — only the status moves.
            streamError = true;
          }
        }

        if (streamError) {
          updateMessage(chatId, pendingAssistant.id, { status: "error" });
          setError("The assistant's response was cut off. Please try again.");
        } else {
          updateMessage(chatId, pendingAssistant.id, { status: "sent" });
        }
      } catch (err) {
        // Offline with a worker to hand it to: park the request instead of
        // failing it. Only for a genuine network error — a 4xx/5xx means the
        // backend answered and replaying it later would change nothing.
        if (isNetworkError(err) && canQueue()) {
          const { url, body } = AgentService.askRequest(trimmed, activeChat.conversationId);
          const queued = await enqueue({
            chatId,
            messageId: pendingAssistant.id,
            url,
            body,
            createdAt: new Date().toISOString(),
          });

          if (queued) {
            updateMessage(chatId, pendingAssistant.id, { status: "queued" });
            void requestFlush();
            // No setError: nothing has gone wrong from the user's side yet,
            // and the bubble already says it's waiting.
            return;
          }
        }

        const message = toUserFacingError(err);
        // Only replace the bubble's text when nothing ever arrived. Once the
        // user has an answer on screen, blanking it to show an error
        // destroys what they were reading; the error line below is enough.
        updateMessage(chatId, pendingAssistant.id, {
          ...(receivedAny ? {} : { content: message }),
          status: "error",
        });
        setError(message);
      } finally {
        setIsSending(false);
      }
    },
    [activeChat, isSending, appendMessage, updateMessage, setConversationId]
  );

  return {
    askAgent,
    isSending,
    error,
  };
}
