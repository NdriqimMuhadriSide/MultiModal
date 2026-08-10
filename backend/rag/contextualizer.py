"""
Query contextualization.

Responsibility: turn a question that only makes sense inside a conversation
into one that makes sense on its own, before it reaches retrieval.

This closes a real hole rather than adding a refinement. Retrieval has no
memory: the embedding model sees one string, and the BM25 index matches the
terms in one string. So a user who asks

    "how much annual leave do I build up each month?"
    "how far ahead do I have to request it?"

sends retrieval the six words of the second question. "it" embeds to nothing
in particular, "request" and "ahead" match half the corpus, and the chunk
about giving fourteen days' notice for annual leave is competing on those
terms alone against every other chunk containing a time period. The
conversation held the answer to what "it" was, and retrieval never saw the
conversation.

The fix is to reconstruct the standalone question first:

    "how far ahead do I have to request annual leave?"

...and retrieve for that. Note what this is *not*: it is not passing history
to the LLM at answer time, which the chat path already does and which does
nothing for retrieval - by then the chunks have already been chosen. The
rewrite has to happen before the search, or it happens too late to matter.

Cost is one LLM call, and only on turns that have a conversation behind them
- the first question of every conversation skips it entirely, because there
is nothing for a reference to point at. That guard is exact rather than
heuristic: no history means no possible resolution, not "probably fine".

Failure degrades to the original question, for the same reason expansion's
does (see rag/query_expansion.py): the un-rewritten question is the
behaviour that existed before this module, so the fallback is a known-good
path rather than an unknown one.
"""
import logging

from prompts.contextualize_prompts import format_contextualize_prompt

logger = logging.getLogger(__name__)

# A rewrite longer than this is not a question - it is a model that decided
# to answer, or to explain its reasoning. Searching for it would retrieve
# whatever the model's own prose happens to resemble, which is worse than
# searching for the original question.
MAX_REWRITE_CHARS = 400


class QueryContextualizer:
    """Rewrites a follow-up question into a standalone one using conversation history."""

    def __init__(self, llm_service) -> None:
        self._llm_service = llm_service

    def contextualize(self, question: str, history: list[dict[str, str]] | None) -> str:
        """
        Return a self-contained version of `question`.

        Returns `question` itself when there is no history to resolve against,
        when the model declines to change anything, or when the call fails.

        Raises:
            ValueError: if `question` is empty.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty.")

        if not history:
            # The first turn of a conversation cannot contain a reference to
            # an earlier one. Skipping here is what keeps this free for
            # single-question use - /rag/ask never pays for it at all.
            return question

        prompt = format_contextualize_prompt(question, history)
        try:
            raw = self._llm_service.generate_response(prompt)
        except Exception as exc:  # noqa: BLE001 - see module docstring
            logger.warning(
                "Query contextualization failed, retrieving on the original question: %s",
                exc,
            )
            return question

        rewritten = _first_line(raw)
        if not rewritten or len(rewritten) > MAX_REWRITE_CHARS:
            logger.warning(
                "Query contextualization returned an unusable rewrite; keeping the original."
            )
            return question

        if rewritten != question:
            logger.info("Contextualized %r -> %r", question, rewritten)
        return rewritten


def _first_line(raw: str) -> str:
    """
    The rewritten question, out of whatever the model returned.

    Takes the first non-empty line rather than the whole response: the prompt
    asks for one line, and when a model adds "Here is the rewritten question:"
    it is on a line of its own, so the question is still the first line that
    looks like one. Surrounding quotes are stripped because models like to add
    them around a rewrite and they would become search terms.
    """
    for line in (raw or "").splitlines():
        candidate = line.strip().strip('"').strip("'").strip()
        if candidate and not candidate.endswith(":"):
            return candidate
    return ""


def get_query_contextualizer():
    """Build a QueryContextualizer from the shared LLM service singleton."""
    from ai.llm_service import get_llm_service

    return QueryContextualizer(get_llm_service())
