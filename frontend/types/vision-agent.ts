/**
 * Types for the vision agent domain.
 *
 * Mirrors backend/app/schemas/vision_agent.py and the frame shapes in
 * backend/app/api/v1/endpoints/vision_agent.py.
 *
 * Distinct from types/vision.ts, which describes POST /vision/analyze — one
 * call to the vision model, one string back. This endpoint runs a loop that
 * decides *how* to read the image (vision model, character recognition, or
 * both) and can check what it finds against the ingested documents. The
 * extra fields here all exist because that process needs to be visible: a
 * reader cannot sanity-check an answer assembled over four hidden tool
 * calls unless they can see the calls.
 *
 * Casing follows the backend's aliases: the envelope keys of stream frames
 * are snake_case (matching the existing agent stream), while anything
 * derived from a Pydantic model is camelCase.
 */
import type { RAGChatSource } from "./rag";

/** One completed turn of the agent's loop. */
export interface AgentStep {
  thought: string;
  tool: string;
  /** The arguments the agent chose, as JSON text — every tool differs. */
  toolInput: string;
  observation: string;
}

/**
 * Why the loop ended. Not decoration: an answer synthesised after the step
 * budget ran out was built from partial work, and showing it identically to
 * a completed one hides that from the reader.
 */
export type StoppedBecause = "finished" | "step_limit" | "parse_failures";

/** POST /api/v1/vision/ask response body */
export interface VisionAgentResponse {
  question: string;
  answer: string;
  conversationId: string;
  steps: AgentStep[];
  /**
   * Knowledge-base passages only, never the image — an image has no chunk
   * id, filename or page to cite. Empty on a question answered purely from
   * the picture, which is the honest result rather than a gap.
   */
  sources: RAGChatSource[];
  stoppedBecause: StoppedBecause;
  /**
   * Figures in the answer that character recognition did not confirm.
   *
   * A caution, not a verdict. A correctly derived per-head amount and a
   * limit quoted from a retrieved policy both land here legitimately. What
   * it gives the reader is the one distinction the answer text cannot be
   * trusted to make on its own: which numbers came off the image.
   */
  unverifiedValues: string[];
}

/**
 * One frame from POST /api/v1/vision/ask/stream.
 *
 * Order: `start`, then one `step` per completed turn, then an optional
 * `sources`, then exactly one `answer`, then an optional `unverified`, then
 * a terminator — `done` or `error`, never both.
 *
 * Steps rather than tokens, because a step is the unit worth watching. A
 * run produces no prose at all until the end, so token streaming would send
 * nothing for twenty seconds and then everything; a step frame lands the
 * moment each tool returns.
 *
 * `unverified` arrives *after* `answer`, unlike the chat agent's citations
 * which precede the first delta. It cannot come earlier — the check reads
 * the finished answer.
 */
export type VisionAgentStreamEvent =
  | { type: "start"; conversation_id: string }
  | { type: "step"; index: number; step: AgentStep }
  | { type: "sources"; sources: RAGChatSource[] }
  | { type: "answer"; content: string }
  | { type: "unverified"; values: string[] }
  | { type: "done"; stopped_because: StoppedBecause }
  | { type: "error"; detail: string };

/**
 * Human-readable labels for the tools the agent can run, used by the trace
 * UI. Keyed by the tool names registered in
 * backend/agents/vision_agent.py — a name missing here falls back to the
 * raw value rather than rendering blank.
 */
export const VISION_TOOL_LABELS: Record<string, string> = {
  inspect_image: "Looking at the image",
  read_text: "Reading the text",
  search_knowledge_base: "Checking the documents",
  finish: "Writing the answer",
};
