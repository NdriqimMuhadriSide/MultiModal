"""
Propositional chunking: break a passage into standalone atomic facts.

Every other strategy cuts text at boundaries that already exist in it -
paragraphs, sentences, topic shifts. This one rewrites it. A language model
turns each passage into a list of self-contained statements, each resolving its
own references:

    passage       "The trial began in March. It was halted six weeks later
                   after three sites reported calibration failures."

    propositions  "The trial began in March 2026."
                  "The trial was halted six weeks after it began."
                  "Three sites reported calibration failures."

The second sentence of the passage is unsearchable on its own - "it" could be
anything, and nothing in the sentence says what was halted. As propositions,
each fact carries its own subject, so each one embeds to a vector that means
what it says.

What is embedded, and what comes back
-------------------------------------
The proposition is embedded; the *source passage* is what gets returned. The
propositions are a rewrite, and a rewrite is exactly what you don't want an
answer grounded in - a model reading "The trial was halted six weeks after it
began" has lost the hedging, the caveats and the numbers around it. So they act
as search keys onto the real text, the same shape as parent-document
retrieval, with the LLM standing in for the splitter.

What it costs
-------------
One LLM call per passage at ingestion - by far the most expensive strategy
here, and the only one whose output is non-deterministic. It fails softly: a
passage the model refuses, garbles, or times out on falls back to being one
ordinary chunk, so a flaky API degrades the index rather than corrupting it.
"""
import logging

from prompts.chunking_prompts import (
    PROPOSITION_SYSTEM_PROMPT,
    format_proposition_prompt,
)
from rag.chunking.base import ChunkBudget, ChunkDraft
from rag.chunking.blocks import Segment, prose_runs, segment_blocks
from rag.layout import Block
from rag.text_splitter import split_text

logger = logging.getLogger(__name__)

# A "proposition" shorter than this is a fragment the model emitted by mistake
# - a stray bullet, a heading it echoed back - not a fact.
MIN_PROPOSITION_CHARS = 20

# More than this from one passage means the model started enumerating clauses
# rather than facts, which floods the index with near-duplicates.
MAX_PROPOSITIONS_PER_PASSAGE = 20


class PropositionalChunking:
    name = "propositional"

    def __init__(self, llm_service, section_context: str = ""):
        self._llm_service = llm_service
        self._section_context = section_context

    def split(self, blocks: list[Block], budget: ChunkBudget) -> list[ChunkDraft]:
        context = " > ".join(blocks[0].section_path) if blocks and blocks[0].section_path else ""
        drafts: list[ChunkDraft] = []
        passage_index = 0

        for group in prose_runs(segment_blocks(blocks, budget)):
            if isinstance(group, Segment):
                # A table is already a set of atomic facts, one per row, and
                # rewriting it through a model would only lose precision.
                drafts.append(ChunkDraft(text=group.text))
                continue

            passages = split_text(
                "\n\n".join(segment.text for segment in group),
                chunk_size=budget.size,
                chunk_overlap=budget.overlap,
                measure=budget.measure,
            )
            for passage in passages:
                drafts.extend(
                    self._propositions(passage.text, f"q{passage_index}", context, budget)
                )
                passage_index += 1

        return drafts

    def _propositions(
        self, passage: str, key: str, context: str, budget: ChunkBudget
    ) -> list[ChunkDraft]:
        try:
            response = self._llm_service.generate_response(
                format_proposition_prompt(passage, context),
                system_prompt=PROPOSITION_SYSTEM_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001 - never fail an upload over this
            logger.warning("Proposition extraction failed, keeping the passage whole: %s", exc)
            return [ChunkDraft(text=passage)]

        propositions = _parse(response, budget)
        if not propositions:
            # The model found nothing factual, or returned something unusable.
            # Either way the passage is still worth indexing as it stands.
            return [ChunkDraft(text=passage)]

        return [
            ChunkDraft(text=passage, embed_text=proposition, parent_key=key)
            for proposition in propositions
        ]


def _parse(response: str, budget: ChunkBudget) -> list[str]:
    """
    Pull propositions out of the model's reply, discarding anything that
    doesn't look like one.

    The prompt asks for one per line with no decoration, but models add bullets
    and numbering anyway, so those are stripped rather than treated as a
    failure. Everything else here is a filter: this text goes straight into the
    index, and a malformed line becomes a permanently bad search key.
    """
    propositions: list[str] = []
    for line in response.splitlines():
        cleaned = line.strip().lstrip("-*•0123456789.) ").strip()
        if len(cleaned) < MIN_PROPOSITION_CHARS:
            continue
        if budget.measure(cleaned) > budget.size:
            continue
        propositions.append(cleaned)
    return propositions[:MAX_PROPOSITIONS_PER_PASSAGE]
