/**
 * AgentService — talks to POST /api/v1/agent/ask.
 *
 * Like ChatService/RagService, the only module that knows this endpoint's
 * URL and request/response shape.
 *
 * This endpoint supersedes the manual choice between ChatService (plain
 * LLM) and RagService (document Q&A): the backend's router decides per
 * message which of those to run, plus a live external-API tool, and reports
 * its choice back in `tool_used`. Both older services are still used
 * directly by the streaming/document flows, so neither is removed.
 */
import { apiClient } from "@/lib/api-client";
import { API_V1_URL } from "@/lib/config";
import type { AgentAskRequest, AgentAskResponse } from "@/types";

export const AgentService = {
  /**
   * The same request as `ask`, described rather than sent.
   *
   * Exists for the service worker outbox, which replays a failed send from
   * outside the page and therefore can't go through apiClient. Building it
   * here keeps this module the only one that knows the endpoint's URL and
   * body shape, which would otherwise be duplicated into lib/outbox.ts.
   */
  askRequest(
    message: string,
    conversationId?: string | null
  ): { url: string; body: AgentAskRequest } {
    return {
      url: `${API_V1_URL}/agent/ask`,
      body: { message, conversation_id: conversationId ?? null },
    };
  },

  /**
   * Sends a message to the agent, which routes it to exactly one tool
   * (knowledge base / external API / direct answer) and returns the answer.
   * `conversationId` is omitted for the first message in a new chat; the
   * backend assigns one and returns it in the response.
   */
  ask(
    message: string,
    conversationId?: string | null,
    signal?: AbortSignal
  ): Promise<AgentAskResponse> {
    const body: AgentAskRequest = {
      message,
      conversation_id: conversationId ?? null,
    };
    return apiClient.post<AgentAskResponse, AgentAskRequest>("/agent/ask", body, {
      signal,
    });
  },
};
