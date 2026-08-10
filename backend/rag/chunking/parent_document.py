"""
Parent-document retrieval: embed small children, return the big parent.

The same tension sentence-window addresses, attacked from the other end. Rather
than starting at a sentence and widening, this starts from an ordinary chunk -
the parent - and cuts it into small children purely to be embedded. The
children compete in the search; whichever wins hands back its parent.

    parent   a full 250-token passage, returned to the model
    children 60-token slices of it, one vector each

Why children match better than their parent would
-------------------------------------------------
An embedding is a single point, so a passage covering four points lands at
their average - somewhere between all four and close to none of them. Cut the
same passage into four, and each piece lands squarely on its own idea. A
question about the third idea now has something to match sharply, instead of a
blurred vector that also half-describes three things the question didn't ask
about.

The difference from sentence-window is what defines the parent. There, the
returned text is a moving window centred on each sentence, so every hit returns
a slightly different span. Here the parent is fixed, so several children return
*exactly* the same passage - which makes deduplication mandatory rather than
merely tidy, and makes the returned text predictable.

What it costs
-------------
Every child stores a copy of its parent's text, so the store grows by roughly
the parent size times the number of children. Keeping parents in a separate
table and joining at retrieval would avoid that, at the price of a second store
to keep consistent with Chroma - a trade this project has repeatedly declined
in favour of one place to look.
"""
from rag.chunking.base import ChunkBudget, ChunkDraft
from rag.chunking.blocks import Segment, prose_runs, segment_blocks
from rag.layout import Block
from rag.text_splitter import split_text

# Child size, in the budget's units. Small enough to hold roughly one idea -
# which is the entire point - and comfortably inside any embedding model.
DEFAULT_CHILD_TOKENS = 64


class ParentDocumentChunking:
    name = "parent_document"

    def __init__(self, child_tokens: int = DEFAULT_CHILD_TOKENS):
        self._child_tokens = max(1, child_tokens)

    def split(self, blocks: list[Block], budget: ChunkBudget) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        parent_index = 0

        for group in prose_runs(segment_blocks(blocks, budget)):
            parents = (
                [group.text]
                if isinstance(group, Segment)
                else [
                    chunk.text
                    for chunk in split_text(
                        "\n\n".join(segment.text for segment in group),
                        chunk_size=budget.size,
                        chunk_overlap=budget.overlap,
                        measure=budget.measure,
                    )
                ]
            )

            for parent in parents:
                drafts.extend(self._children(parent, f"p{parent_index}", budget))
                parent_index += 1

        return drafts

    def _children(self, parent: str, key: str, budget: ChunkBudget) -> list[ChunkDraft]:
        """
        Cut a parent into the pieces that will be embedded on its behalf.

        No overlap between children: they are never read by anyone, only
        matched, and overlapping them would just create near-duplicate vectors
        competing for the same slot. The parent is what gets read, and it is
        already whole.
        """
        children = split_text(
            parent,
            chunk_size=min(self._child_tokens, budget.size),
            chunk_overlap=0,
            measure=budget.measure,
        )
        if not children:
            return []

        return [
            ChunkDraft(text=parent, embed_text=child.text, parent_key=key)
            for child in children
        ]
