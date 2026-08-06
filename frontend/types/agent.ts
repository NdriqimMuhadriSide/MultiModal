/**
 * Types for the agent domain.
 *
 * Mirrors backend/app/schemas/agent.py's AgentAskRequest/AgentAskResponse.
 * Snake_case on the wire (like types/chat.ts, unlike types/rag.ts) because
 * the backend schema for this endpoint declares no Pydantic aliases.
 *
 * The agent is the routing entry point: rather than the caller choosing
 * between /chat and /rag/chat, the backend asks the LLM which tool applies
 * and reports its choice back in `tool_used`.
 */

// ---------------------------------------------------------------------------
// Wire types — must match backend/app/schemas/agent.py
// ---------------------------------------------------------------------------

/** POST /api/v1/agent/ask request body */
export interface AgentAskRequest {
  message: string;
  conversation_id?: string | null;
}

/**
 * Which tool the agent's router picked. Mirrors the `tool_used` values set
 * by the tool nodes in backend/agents/assistant_agent.py.
 */
export type AgentTool =
  | "search_knowledge_base"
  | "call_external_api"
  | "answer_directly";

/** POST /api/v1/agent/ask response body */
export interface AgentAskResponse {
  message: string;
  answer: string;
  tool_used: AgentTool;
  conversation_id: string;
}
