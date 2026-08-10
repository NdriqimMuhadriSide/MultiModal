"""
Query expansion (multi-query retrieval).

Responsibility: turn one question into several phrasings of itself, so
retrieval gets more than one chance to find the passage that answers it.

The problem it addresses is that a single query is a single sample of how
the answer might be worded, and retrieval is unforgiving about that. A user
asking "how do I stop paying for this" and a handbook saying "subscriptions
may be cancelled from the billing page" share no content word at all: BM25
has nothing to match, and the embedding has to carry the entire burden of a
paraphrase it may or may not have learned. Ask the same thing three more
ways - "how do I cancel my subscription", "where do I end my plan" - and the
odds that at least one phrasing lands near the document's own wording go up
sharply.

Each phrasing is retrieved for independently and all the rankings are fused
(rag/fusion.py). That fusion step is what makes this safe rather than merely
broader: a chunk found by one variant and nothing else gets one vote, while
a chunk every phrasing agrees on rises to the top. Expansion widens the net;
fusion is what stops the extra catch from being noise.

The original question is always retrieved for as well, and always first.
This is not a nicety - it is the safeguard. Expansion runs through a
language model, and a model asked to reword "what causes ERR-4021" may
helpfully produce "what causes upload errors", which has thrown away the one
literal token the BM25 half needed. Keeping the original in the mix means
the worst an unhelpful rewrite can do is fail to add anything.

Costs, plainly: one LLM call per question, on the critical path, before
retrieval can start. That is far and away the most expensive thing in the
query path - the reranker adds ~15ms, this adds a network round-trip - which
is why MULTI_QUERY_ENABLED is off by default and CONTEXTUAL_RETRIEVAL, the
other LLM-at-runtime feature, is too.
"""
import logging

from prompts.query_expansion_prompts import format_query_expansion_prompt

logger = logging.getLogger(__name__)

# Lines this long are not questions - they are a model ignoring the "no
# preamble" instruction and writing a paragraph. Retrieving for one would
# spend a slot on something that matches everything weakly.
MAX_VARIANT_CHARS = 300


class QueryExpander:
    """Generates alternative phrasings of a question with an LLM."""

    def __init__(self, llm_service, count: int = 3) -> None:
        if count < 1:
            raise ValueError("count must be a positive integer.")
        self._llm_service = llm_service
        self._count = count

    def expand(self, question: str) -> list[str]:
        """
        Return up to `count` alternative phrasings of `question`.

        Never includes `question` itself - the caller owns the original and
        always retrieves for it, so returning it here would double its weight
        in fusion by accident.

        Returns an empty list rather than raising when the LLM call fails.
        This is a deliberate exception to the project's usual "fail loudly
        rather than silently do something different" rule, and the difference
        is what the failure means: a missing API key is a misconfiguration
        that should be fixed, but a timeout on the expansion call is a
        transient event during a question that can still be answered. The
        degraded result - retrieval on the original question alone - is
        exactly the non-expanded behaviour, so the fallback is a known-good
        path rather than an unknown one. It is logged at warning level so a
        provider that is failing every time is visible rather than merely
        slow-and-worse.

        Raises:
            ValueError: if `question` is empty.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty.")

        prompt = format_query_expansion_prompt(question, count=self._count)
        try:
            raw = self._llm_service.generate_response(prompt)
        except Exception as exc:  # noqa: BLE001 - see docstring
            logger.warning(
                "Query expansion failed, retrieving on the original question only: %s", exc
            )
            return []

        return self._parse(raw, question)

    def _parse(self, raw: str, question: str) -> list[str]:
        """
        Pull clean questions out of whatever the model actually returned.

        Models are inconsistent about the "no numbering" instruction, so the
        parsing is forgiving about leading bullets and digits. It is not
        forgiving about duplicates: a variant identical to the original (or
        to another variant) would get its chunks counted twice in fusion,
        which is an argument from repetition rather than from agreement
        between genuinely different phrasings.
        """
        seen = {_normalise(question)}
        variants: list[str] = []

        for line in (raw or "").splitlines():
            candidate = _strip_list_marker(line.strip())
            if not candidate or len(candidate) > MAX_VARIANT_CHARS:
                continue
            key = _normalise(candidate)
            if key in seen:
                continue
            seen.add(key)
            variants.append(candidate)
            if len(variants) == self._count:
                break

        return variants


def _strip_list_marker(line: str) -> str:
    """Remove a leading "1.", "2)", "-" or "*" that the model added anyway."""
    stripped = line.lstrip("-*• \t")
    index = 0
    while index < len(stripped) and stripped[index].isdigit():
        index += 1
    if index and index < len(stripped) and stripped[index] in ".):":
        stripped = stripped[index + 1 :]
    return stripped.strip()


def _normalise(text: str) -> str:
    """Case- and punctuation-insensitive key for duplicate detection."""
    return "".join(character for character in text.lower() if character.isalnum() or character == " ").strip()


def get_query_expander():
    """
    Build a QueryExpander from the shared LLM service singleton.

    Not cached: it holds no loaded model and no state worth reusing, only a
    reference to the LLM service (which is itself the cached singleton).
    """
    from app.core.config import settings
    from ai.llm_service import get_llm_service

    return QueryExpander(get_llm_service(), count=settings.multi_query_count)
