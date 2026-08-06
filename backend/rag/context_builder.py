"""
Context builder.

Responsibility: turn a list of retrieved chunks into the plain-text context
block that gets embedded in the LLM prompt. This is a deliberately tiny,
single-purpose module - formatting is the *only* thing it does - kept
separate from rag/retriever.py (which decides *which* chunks are relevant)
and prompts/rag_prompts.py (which decides how the context is wrapped with
system instructions and the question) so each concern can change
independently: reformatting how a chunk is labeled doesn't touch retrieval
logic, and rewording the system prompt doesn't touch context formatting.

Layering rule: this module only depends on rag/retriever.py's
RetrievedChunk type. It has no knowledge of ChromaDB, embeddings, or the
LLM - pure data-to-string formatting.
"""
from rag.retriever import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:
    """
    Join retrieved chunks into a single context block, each one labeled
    with its source document and page so the model (and a human reading
    logs) can see where each piece of context came from.

    Example, for one retrieved chunk:

        Document: employee_policy.pdf
        Page: 12
        Content: Employees receive 25 vacation days...

    Returns an empty string if `chunks` is empty - callers (rag/rag_service.py)
    treat an empty context as "nothing relevant was retrieved" and skip the
    LLM call entirely rather than prompting it with no grounding material.
    """
    blocks = [
        f"Document: {chunk.filename}\nPage: {chunk.page}\nContent: {chunk.text}"
        for chunk in chunks
    ]
    return "\n\n".join(blocks)
