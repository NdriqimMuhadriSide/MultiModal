"""
What every chunking strategy produces, and what it gets to work with.

The one idea that makes the alternative strategies possible is that **the text
that gets embedded need not be the text that gets returned**. Everything up to
now has assumed they're the same string; several of the better-performing
retrieval techniques are exactly the observation that they shouldn't be:

    sentence-window   embed one sentence, return it with its neighbours
    parent-document   embed a small child, return the whole parent
    propositional     embed an atomic fact, return the passage it came from
    contextual        embed the chunk with an LLM-written preamble

They all pull in the same direction, because the two texts are being judged by
different things. What's embedded is scored by a similarity function that gets
sharper as the text gets *narrower* - one clear idea makes one clear vector.
What's returned is read by a language model that answers better as the text
gets *wider*, because a sentence with no context around it is hard to reason
from. A single string has to compromise between the two; a pair doesn't.

`ChunkDraft` is that pair, plus the key that stops the trick backfiring: when
several children point at one parent, retrieval would otherwise return the same
passage three times and fill top-k with duplicates.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rag.layout import Block


@dataclass
class ChunkDraft:
    """One chunk, before ingestion attaches ids and document metadata."""

    # Stored in the vector store and handed back at retrieval - the text a
    # language model will read when answering from this chunk.
    text: str
    # What actually gets embedded. None means "same as text", which is the
    # ordinary case and keeps simple strategies simple.
    embed_text: str | None = None
    # Identifies the larger passage this chunk points at, when several chunks
    # point at the same one. Retrieval keeps only the best-scoring chunk per
    # key, so a parent is returned once however many of its children matched.
    # None for strategies where every chunk stands alone.
    parent_key: str | None = None

    def embedded(self) -> str:
        return self.embed_text if self.embed_text is not None else self.text


@dataclass
class ChunkBudget:
    """
    The size limit a strategy has to respect, and the yardstick for it.

    `measure` is the embedding model's tokenizer at ingestion time, so `size`
    is in the units the model's context window is actually expressed in - see
    rag/text_splitter.py. Passed around rather than read from settings so a
    strategy stays testable with plain character counts.
    """

    size: int
    overlap: int
    measure: Callable[[str], int] = len


class ChunkingStrategy(Protocol):
    """
    Turns one section's blocks into chunks.

    Called per section run rather than per document, so a chunk never straddles
    a heading and every strategy inherits that for free (see
    rag/structure.py for why that boundary matters).
    """

    name: str

    def split(self, blocks: list[Block], budget: ChunkBudget) -> list[ChunkDraft]:
        ...
