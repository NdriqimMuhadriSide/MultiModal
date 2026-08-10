"""
Prompt management layer.

Centralizes every prompt the assistant sends to the model. Keeping prompts
here - instead of inlined as string literals inside `ai/llm_service.py` -
means:

- One place to review, tune, and diff prompt wording over time.
- Prompts can be unit tested / linted separately from request-handling code.
- The same prompt constant can be reused across multiple services (chat,
  vision, RAG, agents) once they're added, instead of copy-pasted.

Naming convention: ALL_CAPS constants for static prompts, e.g. SYSTEM_PROMPT.
As more prompts are added (e.g. RAG_SYSTEM_PROMPT, VISION_SYSTEM_PROMPT,
SUMMARIZATION_PROMPT), keep them in this module or split into sibling files
in this package (e.g. prompts/rag_prompts.py) if it grows large.
"""

SYSTEM_PROMPT = (
    "You are a professional AI assistant. "
    "Answer clearly. "
    "If information is unknown, say so. "
    "Do not hallucinate."
)

# The RAG system prompt and prompt template moved to prompts/rag_prompts.py
# (RAG_SYSTEM_PROMPT / format_rag_prompt) - kept as its own module since the
# RAG pipeline is its own layer with its own prompt lifecycle.


# Used by agents/assistant_agent.py's decision node. Describes the three
# tools the agent can choose between and asks it to respond with a single
# routing keyword. Kept deliberately narrow (one word out) rather than
# open-ended, so the routing step is cheap and reliable to parse - the
# actual answer generation happens in a separate LLM call after routing.
AGENT_ROUTER_PROMPT_TEMPLATE = (
    "You are a routing component inside an AI agent. Given the user's "
    "message, decide which single tool should handle it:\n\n"
    "- KNOWLEDGE_BASE: the message asks about internal documents, policies, "
    "or information that would come from ingested company documents "
    "(e.g. refund policy, product docs, internal procedures).\n"
    "- EXTERNAL_API: the message asks for CURRENT, LIVE weather for a named "
    "location. Only weather - nothing else qualifies.\n"
    "- DIRECT_ANSWER: the message is general knowledge, conversation, or "
    "reasoning that does not require looking anything up. This includes "
    "facts like populations, history, and definitions.\n\n"
    "{history_block}"
    "Respond with exactly one word: KNOWLEDGE_BASE, EXTERNAL_API, or "
    "DIRECT_ANSWER. No punctuation, no explanation.\n\n"
    "User message: {message}"
)

# Recent turns are injected as plain text *inside* the routing prompt rather
# than passed as real conversation history to the LLM. Sending them as chat
# turns makes the model answer the follow-up conversationally instead of
# emitting one routing keyword; as inert context it can still resolve
# pronouns ("what is its population?") without being drawn into replying.
_ROUTER_HISTORY_BLOCK_TEMPLATE = (
    "For context, here are the most recent turns of the conversation. Use "
    "them only to understand what the user's message refers to (e.g. what "
    '"it" means) - do not answer them:\n{turns}\n\n'
)

# Enough to resolve a pronoun without burying the instructions.
_ROUTER_HISTORY_TURNS = 4


def format_router_prompt(
    message: str, history: list[dict[str, str]] | None = None
) -> str:
    """
    Fill AGENT_ROUTER_PROMPT_TEMPLATE with the user's message, and with the
    tail of `history` (oldest first, same shape the memory layer produces)
    when there is any.
    """
    history_block = ""
    if history:
        turns = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in history[-_ROUTER_HISTORY_TURNS:]
        )
        history_block = _ROUTER_HISTORY_BLOCK_TEMPLATE.format(turns=turns)

    return AGENT_ROUTER_PROMPT_TEMPLATE.format(
        message=message, history_block=history_block
    )


# Used by ai/vision_service.py for the POST /vision/analyze endpoint.
# Structured as an explicit 3-step instruction so the model doesn't skip
# straight to an answer without grounding it in what's actually visible.
VISION_SYSTEM_PROMPT = (
    "You are a professional AI assistant that analyzes images. "
    "When given an image and a question, respond in three clearly labeled parts: "
    "1) Description: describe what is visible in the image. "
    "2) Extracted information: pull out any useful details (text, error messages, "
    "labels, numbers, UI elements, data). "
    "3) Answer: directly answer the user's question using only the image and the "
    "information you extracted. "
    "If something is unclear, unreadable, or not visible in the image, say so. "
    "Do not hallucinate details that are not actually present in the image."
)


# Used by agents/assistant_agent.py's second hop, after the knowledge-base
# tool searched the ingested documents and retrieved nothing.
#
# The disclosure instruction is the whole point. Without it the model answers
# from its own knowledge in the same voice it uses for a document-grounded
# answer, and a user reading "expense claims must be submitted within 30 days"
# has no way to tell whether that came from their handbook or from the
# model's impression of what handbooks usually say. One is a fact about their
# company; the other is a plausible guess.
KB_FALLBACK_PROMPT_TEMPLATE = (
    "The user asked a question and the ingested documents were searched, but "
    "nothing relevant was found in them.\n\n"
    "Answer from your own general knowledge instead. Begin by stating plainly "
    "that this was not found in their documents and that the answer is general "
    "knowledge rather than something from their materials. If you do not know "
    "the answer either, say so.\n\n"
    "Question: {message}"
)


def format_kb_fallback_prompt(message: str) -> str:
    """Fill KB_FALLBACK_PROMPT_TEMPLATE for a knowledge-base miss."""
    return KB_FALLBACK_PROMPT_TEMPLATE.format(message=message)
