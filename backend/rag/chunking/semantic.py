"""
Semantic chunking: let the text decide where it changes subject.

Every other strategy cuts on *form* - a blank line, a sentence end, a token
count. Form is a proxy for meaning and usually a good one, but it is blind to
the case that matters: a paragraph break that isn't a topic change, and a topic
change that isn't a paragraph break. Two paragraphs elaborating one idea get
split; a single long paragraph that pivots halfway through does not.

The method
----------
Embed each sentence, then walk the sequence measuring how far each sentence sits
from the one before it. Adjacent sentences about the same thing land close
together in embedding space; the distance spikes exactly where the subject
turns. Cut at the spikes.

    s1 ---- s2 --- s3 ------------- s4 --- s5
                          ^ the gap here is the topic change

"Spike" is relative, not absolute: a distance of 0.3 is unremarkable in a
rambling document and enormous in a tightly-argued one. So the threshold is a
percentile of the distances *in this passage* rather than a fixed number, which
is what lets the same setting work on a legal contract and a blog post.

What it costs
-------------
One embedding per sentence, at ingestion. That is a real cost - a 40-page
report is a few thousand sentences - and it buys boundaries that are better
placed but not always better *sized*: the resulting groups still have to be
packed to the token budget afterwards, and a group that survives that packing
untouched is where the benefit actually lands.
"""
import math

from rag.chunking.base import ChunkBudget, ChunkDraft
from rag.chunking.blocks import Segment, prose_runs, segment_blocks
from rag.layout import Block
from rag.sentences import split_sentences
from rag.text_splitter import split_text

# Cut where the distance between consecutive sentences is in the top N% for
# this passage. 95 is deliberately conservative: a false cut costs a split
# idea, while a missed cut only leaves behaviour where the recursive splitter
# would have been anyway.
DEFAULT_BREAKPOINT_PERCENTILE = 95

# Below this many sentences there is nothing to measure - two points give one
# distance, and a percentile of one number is that number.
MIN_SENTENCES_TO_ANALYSE = 4


class SemanticChunking:
    name = "semantic"

    def __init__(self, embedding_service, breakpoint_percentile: int = DEFAULT_BREAKPOINT_PERCENTILE):
        self._embedding_service = embedding_service
        self._percentile = breakpoint_percentile

    def split(self, blocks: list[Block], budget: ChunkBudget) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for group in prose_runs(segment_blocks(blocks, budget)):
            if isinstance(group, Segment):
                drafts.append(ChunkDraft(text=group.text))
                continue

            text = "\n\n".join(segment.text for segment in group)
            for passage in self._semantic_groups(text):
                # The groups are chosen by meaning and have no idea about the
                # token budget, so anything oversized still goes through the
                # ordinary splitter. Groups that fit pass through untouched -
                # those are the ones this strategy exists to produce.
                drafts.extend(
                    ChunkDraft(text=chunk.text)
                    for chunk in split_text(
                        passage,
                        chunk_size=budget.size,
                        chunk_overlap=budget.overlap,
                        measure=budget.measure,
                    )
                )
        return drafts

    def _semantic_groups(self, text: str) -> list[str]:
        """Group consecutive sentences, breaking where the subject turns."""
        sentences = [sentence for sentence in split_sentences(text) if sentence.strip()]
        if len(sentences) < MIN_SENTENCES_TO_ANALYSE:
            return [text] if text.strip() else []

        embedded = self._embedding_service.embed_texts(
            [sentence.strip() for sentence in sentences]
        )
        distances = [
            1.0 - _cosine(first.embedding, second.embedding)
            for first, second in zip(embedded, embedded[1:])
        ]
        threshold = _percentile(distances, self._percentile)

        groups: list[str] = []
        current = [sentences[0]]
        for sentence, distance in zip(sentences[1:], distances):
            if distance > threshold:
                groups.append("".join(current))
                current = []
            current.append(sentence)
        groups.append("".join(current))
        return [group for group in groups if group.strip()]


def _cosine(first: list[float], second: list[float]) -> float:
    """
    Dot product, which *is* cosine similarity here: the embedding service
    normalises its vectors, so their magnitudes are already 1.
    """
    return sum(a * b for a, b in zip(first, second))


def _percentile(values: list[float], percentile: int) -> float:
    """
    The value below which `percentile`% of `values` fall, by linear
    interpolation.

    Hand-rolled rather than pulled from numpy or statistics.quantiles: the
    input is a handful of floats, and the exact interpolation convention
    matters less than being able to read what the threshold means.
    """
    if not values:
        return math.inf
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
