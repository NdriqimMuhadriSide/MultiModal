/**
 * Types for the agent domain.
 *
 * Mirrors backend/app/schemas/agent.py's AgentAskRequest/AgentAskResponse.
 * Snake_case on the wire (like types/chat.ts, unlike types/rag.ts) because
 * the backend schema for this endpoint declares no Pydantic aliases — with
 * the exception of the nested citations and trace steps, which keep the
 * camelCase aliases they have elsewhere because they are literally those
 * models reused.
 *
 * The agent is the general entry point: rather than the caller choosing
 * between /chat, /rag/chat and /vision/ask, the backend runs a supervisor
 * that answers directly when it can and delegates to a specialist when it
 * cannot — sometimes to more than one in a single turn.
 */
import type { RAGChatSource } from "./rag";

// ---------------------------------------------------------------------------
// Wire types — must match backend/app/schemas/agent.py
// ---------------------------------------------------------------------------

/** POST /api/v1/agent/ask request body */
export interface AgentAskRequest {
  message: string;
  conversation_id?: string | null;
}

/**
 * What the supervisor's answer rests on, in one word, for the badge on the
 * bubble. Mirrors the `ToolUsed` values in
 * backend/agents/supervisor_agent.py — every member of that Literal must
 * appear here, or the bubble has no label to render for it.
 *
 * `answer_directly` means no specialist was used at all, which is the common
 * and correct case for general knowledge. `multiple_specialists` is reported
 * when more than one contributed: naming either alone would tell the reader
 * the answer rests on less than it does. The detail is in `steps`.
 */
export type AgentTool =
  | "answer_directly"
  | "research_documents"
  | "read_image"
  | "call_external_api"
  | "multiple_specialists";

/**
 * One step of the supervisor's trace, mirroring
 * backend/app/schemas/agent_trace.py's AgentStepModel.
 *
 * `children` are the steps a delegated specialist took to produce this
 * step's observation — empty for every ordinary tool call. Recursive rather
 * than one flat level because nothing in the backend loop caps how deep
 * delegation goes, so a renderer that assumed a depth would be wrong the
 * first time that changed.
 */
export interface AgentStepView {
  thought: string;
  tool: string;
  /** The arguments the agent chose, as JSON text. */
  toolInput: string;
  observation: string;
  children: AgentStepView[];
}

/** Why a run ended. `finished` is the good case. */
export type AgentStopReason = "finished" | "step_limit" | "parse_failures";

/** POST /api/v1/agent/ask response body */
export interface AgentAskResponse {
  message: string;
  answer: string;
  tool_used: AgentTool;
  conversation_id: string;
  /**
   * The chunks a delegated document answer was grounded in — the same
   * `RAGChatSource` POST /rag/chat returns, deliberately reused rather than
   * redeclared: it is the same chunk from the same pipeline, and the bubble
   * renders both with the same component.
   *
   * Empty when nothing was cited. Index i is the passage labelled [E(i+1)]
   * in the answer, and stays so across every specialist a turn delegated to,
   * because they all number from one shared ledger on the backend.
   */
  sources: RAGChatSource[];
  /** The supervisor's trace, with each delegation's sub-steps nested. */
  steps: AgentStepView[];
  stopped_because: AgentStopReason;
}

/**
 * One event from POST /api/v1/agent/ask/stream, mirroring the payloads built
 * in backend/app/services/agent_service.py.
 *
 * Order is `start`, then any number of `step`s, then `tool`, then an optional
 * `sources`, then `answer`, then exactly one terminator — `done` or `error`,
 * never both.
 *
 * STEPS, NOT TOKEN DELTAS
 *
 * This stream used to carry `delta` events. It cannot any more, and the
 * reason is not a regression: a supervisor's answer is already whole inside
 * its final action by the time the backend loop sees it, so there is no
 * token stream left to forward. What arrives instead is one frame per
 * completed step, which suits the work better — a delegating run is several
 * seconds of thinking with no prose at all, and "asking the document
 * specialist about the refund window" is a more useful thing to show than an
 * empty bubble.
 *
 * `depth` is 0 for the supervisor's own steps and 1 for a specialist's, so
 * the client can indent a delegated step rather than presenting a
 * specialist's search as something the supervisor did itself.
 *
 * `tool` arrives after the steps rather than before them, unlike the old
 * router's: with delegation the honest label depends on what the whole turn
 * ended up using.
 */
export type AgentStreamEvent =
  | { type: "start"; conversation_id: string }
  | { type: "step"; index: number; depth: number; step: AgentStepView }
  | { type: "tool"; tool: AgentTool }
  | { type: "sources"; sources: RAGChatSource[] }
  | { type: "answer"; content: string }
  | { type: "done"; stopped_because: AgentStopReason }
  | { type: "error"; detail: string };
