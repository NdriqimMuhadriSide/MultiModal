"""
Recursive chunking - the default strategy.

Prose is split by rag/text_splitter.py, walking paragraph -> line -> sentence
-> word and cutting at the most structural boundary that fits. Tables and code
blocks are kept whole, or split by row with their header repeated, because a
general-purpose splitter destroys them (see rag/chunking/blocks.py).

This is the strategy to beat. It costs nothing beyond the tokenizer, it never
calls a model, and it is right about the thing that matters most - chunk
boundaries land where a reader would put them. The alternatives in this package
buy accuracy on specific failure modes with embeddings, LLM calls, or a much
larger index; none of them makes this one obsolete.
"""
from rag.chunking.base import ChunkBudget, ChunkDraft
from rag.chunking.blocks import Segment, prose_runs, segment_blocks
from rag.layout import Block
from rag.text_splitter import split_text


class RecursiveChunking:
    name = "recursive"

    def split(self, blocks: list[Block], budget: ChunkBudget) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for group in prose_runs(segment_blocks(blocks, budget)):
            if isinstance(group, Segment):
                drafts.append(ChunkDraft(text=group.text))
                continue
            drafts.extend(
                ChunkDraft(text=chunk.text)
                for chunk in split_text(
                    "\n\n".join(segment.text for segment in group),
                    chunk_size=budget.size,
                    chunk_overlap=budget.overlap,
                    measure=budget.measure,
                )
            )
        return drafts
