"""
Contextual retrieval: an LLM-written line of context on every chunk.

Anthropic's technique, and one of the largest published gains from a change
this small. The problem it names is the one this whole package keeps circling:
a chunk is embedded alone, but it was written to be read in place.

    "The company's revenue grew by 3% over the previous quarter."

Which company? Which quarter? The chunk cannot answer either, so a search for
"how did ACME do in Q2 2023" has nothing to match. Every other strategy here
attacks that by changing *where the cut falls*. This one leaves the cut alone
and adds what the cut removed:

    "This chunk is from ACME Corp's Q2 2023 SEC filing, in the section on
     quarterly performance. The company's revenue grew by 3% over the
     previous quarter."

The difference from the section header
--------------------------------------
Chunks already carry their heading path ("[2. Methods > 2.1 Sampling]"), which
is the same idea done mechanically and for free. This goes further in the one
way that matters: a heading says where the chunk *sits*, while a written line
can say what it is *about* - naming the entity, the period and the subject,
none of which the heading necessarily contains. It costs an LLM call per chunk
to get that.

Applied as a wrapper, not a strategy
------------------------------------
It changes what gets embedded, not where the boundaries fall, so it composes
with every strategy in this package rather than replacing one. Recursive plus
context, sentence-window plus context, and so on.

Budgeting
---------
The line is part of the embedded text, so room has to be reserved for it before
splitting - the same arithmetic as the section header, and the same failure if
skipped. The reservation is a fixed allowance, and the generated line is
clamped to it: an LLM asked for one sentence occasionally writes four, and the
alternative to clamping is the silent truncation this layer exists to prevent.
"""
import logging

from prompts.chunking_prompts import CONTEXTUAL_SYSTEM_PROMPT, format_contextual_prompt
from rag.chunking.base import ChunkBudget, ChunkDraft
from rag.layout import Block

logger = logging.getLogger(__name__)

# How much of the document to show the model when asking it to situate a chunk.
# Enough to cover a title, an abstract and the opening of the body - which is
# where a document says what it is - without paying to send the whole thing on
# every call.
DOCUMENT_CONTEXT_CHARS = 2000


class ContextualEnrichment:
    """
    Wraps another strategy, prefixing each chunk with a written line of context.

    Holds no chunking logic of its own: `split` delegates, then enriches.
    """

    def __init__(self, strategy, llm_service, reserved_tokens: int, document_context: str = ""):
        self._strategy = strategy
        self._llm_service = llm_service
        self._reserved_tokens = reserved_tokens
        self._document_context = document_context
        self.name = f"{strategy.name}+contextual"

    def split(self, blocks: list[Block], budget: ChunkBudget) -> list[ChunkDraft]:
        drafts = self._strategy.split(blocks, budget)
        return [self._enrich(draft, budget) for draft in drafts]

    def _enrich(self, draft: ChunkDraft, budget: ChunkBudget) -> ChunkDraft:
        context = self._context_line(draft.embedded(), budget)
        if not context:
            return draft

        # Prepended to both: to the embedded text because that is the point,
        # and to the returned text because the model answering benefits from
        # the same orientation the searcher did. Both fit, because the caller
        # reserved room for the line before splitting.
        return ChunkDraft(
            text=f"{context}\n\n{draft.text}",
            embed_text=f"{context}\n\n{draft.embedded()}",
            parent_key=draft.parent_key,
        )

    def _context_line(self, chunk_text: str, budget: ChunkBudget) -> str:
        try:
            response = self._llm_service.generate_response(
                format_contextual_prompt(self._document_context, chunk_text),
                system_prompt=CONTEXTUAL_SYSTEM_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001 - never fail an upload over this
            logger.warning("Contextual enrichment failed for a chunk: %s", exc)
            return ""

        line = " ".join(response.split()).strip('"').strip()
        if not line:
            return ""

        if budget.measure(line) > self._reserved_tokens:
            # Asked for one sentence, given four. Keeping the first sentence is
            # better than truncating mid-word, and better than dropping the
            # line entirely - the opening sentence is the one that names the
            # subject.
            first = line.split(". ")[0].strip()
            if budget.measure(first) > self._reserved_tokens:
                logger.warning("Contextual line too long even truncated; skipping it.")
                return ""
            line = first if first.endswith(".") else f"{first}."
        return line


def document_context(pages) -> str:
    """
    A synopsis of the document to show the model alongside each chunk.

    The opening of a document is where it introduces itself - title, date,
    parties, abstract - so the first couple of thousand characters carry most
    of what is needed to situate anything later in it. Sending the whole
    document on every chunk would be the thorough version and is not worth what
    it costs.
    """
    collected: list[str] = []
    length = 0
    for page in pages:
        for line in page.text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            collected.append(stripped)
            length += len(stripped)
            if length >= DOCUMENT_CONTEXT_CHARS:
                return "\n".join(collected)
    return "\n".join(collected)
