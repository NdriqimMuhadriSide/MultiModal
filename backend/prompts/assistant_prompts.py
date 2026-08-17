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
