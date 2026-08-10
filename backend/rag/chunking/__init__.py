"""
Chunking strategies.

Responsibility: decide where a document is cut, and what each piece is used
for. One strategy is selected per deployment via CHUNKING_STRATEGY, and every
one of them produces the same `ChunkDraft` records, so ingestion, embedding,
storage and retrieval are the same code whichever is chosen.

    recursive        paragraph -> sentence -> word, tables kept whole (default)
    semantic         embed sentences, cut where the subject changes
    sentence_window  embed one sentence, return it with its neighbours
    parent_document  embed small children, return the whole parent
    propositional    an LLM rewrites passages into standalone facts

...and CONTEXTUAL_RETRIEVAL layers an LLM-written line of context on top of
whichever is chosen, because that changes what is embedded rather than where
the cuts fall.

There is no best one. They trade the same three things against each other:

    cost        recursive is free; semantic embeds every sentence;
                propositional and contextual call an LLM per passage
    index size  sentence_window and parent_document store several entries
                per passage, and parent_document stores the passage on each
    precision   narrower embedded text matches more sharply, which is the
                point of all four alternatives

Defaulting to `recursive` is deliberate: it is the only one with no runtime
dependency on a model, and it is a strong baseline that the others beat on
particular documents rather than in general.
"""
from app.core.config import settings
from rag.chunking.base import ChunkBudget, ChunkDraft, ChunkingStrategy
from rag.chunking.contextual import ContextualEnrichment, document_context
from rag.chunking.parent_document import ParentDocumentChunking
from rag.chunking.propositional import PropositionalChunking
from rag.chunking.recursive import RecursiveChunking
from rag.chunking.semantic import SemanticChunking
from rag.chunking.sentence_window import SentenceWindowChunking

__all__ = [
    "ChunkBudget",
    "ChunkDraft",
    "ChunkingStrategy",
    "ContextualEnrichment",
    "STRATEGY_NAMES",
    "build_strategy",
    "document_context",
]

STRATEGY_NAMES = (
    "recursive",
    "semantic",
    "sentence_window",
    "parent_document",
    "propositional",
)


def build_strategy(name: str, embedding_service=None, llm_service=None) -> ChunkingStrategy:
    """
    Construct the named strategy.

    The embedding and LLM services are passed in rather than imported so a
    strategy that doesn't need them never touches them - selecting `recursive`
    must not load a model or require an API key, which is what keeps the
    default deployment dependency-free.

    Raises:
        ValueError: if `name` isn't a known strategy, or its dependency is
            missing. Failing here beats silently falling back to a different
            strategy than the one that was configured.
    """
    if name == "recursive":
        return RecursiveChunking()

    if name == "semantic":
        if embedding_service is None:
            raise ValueError("The 'semantic' chunking strategy needs an embedding service.")
        return SemanticChunking(
            embedding_service,
            breakpoint_percentile=settings.semantic_breakpoint_percentile,
        )

    if name == "sentence_window":
        return SentenceWindowChunking(window_sentences=settings.sentence_window_size)

    if name == "parent_document":
        return ParentDocumentChunking(child_tokens=settings.parent_child_tokens)

    if name == "propositional":
        if llm_service is None:
            raise ValueError("The 'propositional' chunking strategy needs an LLM service.")
        return PropositionalChunking(llm_service)

    raise ValueError(
        f"Unknown chunking strategy '{name}'. Valid options: {', '.join(STRATEGY_NAMES)}."
    )
