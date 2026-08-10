"""
Sentence-window retrieval: embed one sentence, return it with its neighbours.

The problem with any fixed-size chunk is that it is one string doing two jobs.
Similarity search wants it *narrow*: a 250-token chunk covering four different
points produces a vector that is the average of four things and a sharp match
for none of them. The language model reading it wants it *wide*: a sentence
retrieved on its own is often unanswerable, because the subject was named two
sentences earlier.

Splitting the two texts apart resolves the tension outright. The embedding is
built from a single sentence - one idea, one clean vector - and what comes back
is that sentence plus the ones either side of it, so the model gets the
paragraph it needed to make sense of the hit.

    embedded:  "Sites were resampled after the calibration failure."
    returned:  "...Three sites were excluded in March. Sites were resampled
                after the calibration failure. The revised totals appear in
                Table 4..."

What it costs
-------------
The index gets much bigger - one entry per sentence instead of one per few
hundred tokens - and neighbouring sentences overlap heavily in what they
return, so the same passage can be reached many ways. That is what `parent_key`
is for: retrieval keeps the best-scoring sentence per window and drops the
rest, so top-k holds distinct passages rather than five views of one paragraph.
"""
from rag.chunking.base import ChunkBudget, ChunkDraft
from rag.chunking.blocks import Segment, prose_runs, segment_blocks
from rag.layout import Block
from rag.sentences import split_sentences
from rag.text_splitter import split_text

# Sentences of context on each side of the embedded one. Three each way is
# usually a paragraph, which is about what it takes to resolve a pronoun or an
# implied subject without dragging in an unrelated topic.
DEFAULT_WINDOW_SENTENCES = 3


class SentenceWindowChunking:
    name = "sentence_window"

    def __init__(self, window_sentences: int = DEFAULT_WINDOW_SENTENCES):
        self._window = max(0, window_sentences)

    def split(self, blocks: list[Block], budget: ChunkBudget) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for index, group in enumerate(prose_runs(segment_blocks(blocks, budget))):
            if isinstance(group, Segment):
                # A table has no sentences to window over; it stands alone.
                drafts.append(ChunkDraft(text=group.text))
                continue

            text = "\n\n".join(segment.text for segment in group)
            drafts.extend(self._windows(text, budget, run_index=index))
        return drafts

    def _windows(self, text: str, budget: ChunkBudget, run_index: int) -> list[ChunkDraft]:
        sentences = [sentence.strip() for sentence in split_sentences(text) if sentence.strip()]
        if not sentences:
            return []

        # Neighbouring sentences produce near-identical windows, so they are
        # grouped into regions and retrieval keeps the best-scoring sentence
        # per region. Without this, top-k fills with five views of one
        # paragraph and the other four slots are wasted.
        region = max(1, self._window)

        return [
            ChunkDraft(
                text=self._window_text(sentences, position, budget),
                embed_text=sentence,
                parent_key=f"w{run_index}:{position // region}",
            )
            for position, sentence in enumerate(sentences)
        ]

    def _window_text(self, sentences: list[str], position: int, budget: ChunkBudget) -> str:
        """
        Widen outwards from the embedded sentence for as long as it fits.

        Grown rather than sliced to a fixed span and trimmed, because trimming
        would cut a sentence in half at answer time - and because the sentence
        that was embedded is the one that must survive. It is placed first and
        every widening step is checked against the budget, so it can never be
        the part that gets dropped.
        """
        if budget.measure(sentences[position]) > budget.size:
            # A single sentence longer than the whole budget - rare, but a
            # 300-token legal clause is one sentence. Fall back to the ordinary
            # splitter rather than returning something the model would truncate.
            pieces = split_text(
                sentences[position],
                chunk_size=budget.size,
                chunk_overlap=0,
                measure=budget.measure,
            )
            return pieces[0].text if pieces else sentences[position]

        start = stop = position  # inclusive on both ends
        window = sentences[position]
        for _ in range(self._window):
            for widened_start, widened_stop in (
                (start - 1, stop),  # take one from the left...
                (start, stop + 1),  # ...then one from the right
            ):
                if widened_start < 0 or widened_stop >= len(sentences):
                    continue
                candidate = " ".join(sentences[widened_start : widened_stop + 1])
                if budget.measure(candidate) <= budget.size:
                    start, stop, window = widened_start, widened_stop, candidate
        return window
