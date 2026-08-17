"""
Conversation compaction - what happens to a turn that falls out of the
history window.

Before this, the answer was "nothing". Every service loads the last
settings.conversation_history_limit messages and sends those; message 11 of
a conversation was still on disk, still visible in the transcript the UI
renders from GET /chat/{id}/history, and permanently invisible to the
model. The failure that produces is a specific and confusing one: the user
can read, on screen, the message the assistant just contradicted.

So the turns that drop out are summarised, and the summary is sent with
every subsequent prompt:

    [summary of everything up to message N]   <- one system message
    message N+1 ... message N+k               <- the window, verbatim

WHAT IS SUMMARISED, EXACTLY

Two markers decide the slice, and between them they guarantee no message is
in both halves or in neither:

    covered_through_id   the summary's high-water mark. Everything at or
                         below it is already represented.
    oldest in window     the first message still being sent verbatim.

Anything strictly between them is uncompacted history, and that is what a
pass consumes. A pass rewrites the whole summary rather than appending to
it - see prompts/compaction_prompts.py for why a chain of summaries is the
wrong shape.

WHEN IT RUNS

After the assistant's turn is stored, and only once `trigger` messages have
built up in that gap. That placement is a compromise, and worth being
honest about: it costs an LLM call inside a request, so roughly one turn in
every `trigger`/2 exchanges is slower. The alternatives were worse for this
codebase - compacting *before* building the prompt delays the answer the
user is waiting on, and a background worker means a queue this project
doesn't have. When one exists, this is the first thing that should move to
it.

FAILURE IS NOT THE TURN'S PROBLEM

`compact` never raises. A provider timeout during summarisation must not
turn a successful answer into a 502 - the user got what they asked for, and
the worst case of a skipped pass is that the same turns are summarised on
the next one. Everything here is idempotent for exactly that reason.
"""
import logging

from ai.llm_service import LLMService
from memory.conversation_memory import ConversationMemory
from prompts.compaction_prompts import (
    COMPACTION_SYSTEM_PROMPT,
    format_compaction_prompt,
)

logger = logging.getLogger(__name__)

# How a stored summary is introduced to the model. The "system" role is
# deliberate: this is not something either party said, and labelling it
# `user` or `assistant` would invite the model to answer it or to treat it
# as its own previous words. The agents that render history as text (see
# prompts/agent_loop_prompts.py) print it as "system: ...", which reads
# correctly there too.
SUMMARY_ROLE = "system"
SUMMARY_PREFIX = "Summary of earlier turns in this conversation:"


class ConversationCompactor:
    """Summarises turns as they fall out of the history window."""

    def __init__(
        self,
        memory: ConversationMemory,
        llm_service: LLMService,
        history_limit: int = 10,
        trigger: int = 10,
        max_words: int = 200,
    ) -> None:
        self._memory = memory
        self._llm_service = llm_service
        self._history_limit = history_limit
        self._trigger = trigger
        self._max_words = max_words

    def summary_prefix(self, conversation_id: str) -> list[dict[str, str]]:
        """
        The summary message to put in front of the history window, as a
        zero- or one-element list.

        A list rather than an optional value so callers can write
        `prefix + window` without a branch - which matters because there
        are six of them and a missed branch is a silently forgetful
        conversation.
        """
        summary = self._memory.get_summary(conversation_id)
        if summary is None:
            return []

        return [
            {
                "role": SUMMARY_ROLE,
                "content": f"{SUMMARY_PREFIX}\n{summary.summary}",
            }
        ]

    def compact(self, conversation_id: str) -> None:
        """
        Summarise whatever has fallen out of the window since the last pass,
        if there is enough of it to be worth a call.

        Never raises. See the module docstring.
        """
        try:
            self._compact(conversation_id)
        except Exception:  # noqa: BLE001 - a failed pass must not fail the turn
            logger.warning(
                "Compaction failed for conversation %s; it will be retried on the "
                "next turn.",
                conversation_id,
                exc_info=True,
            )

    def _compact(self, conversation_id: str) -> None:
        window = self._memory.get_history(conversation_id, limit=self._history_limit)
        if len(window) < self._history_limit:
            # The whole conversation still fits in the window, so nothing
            # has fallen out of it yet and there is nothing to compact.
            return

        existing = self._memory.get_summary(conversation_id)
        uncompacted = self._memory.get_messages_between(
            conversation_id,
            after_id=existing.covered_through_id if existing else 0,
            before_id=window[0].message_id,
        )
        if len(uncompacted) < self._trigger:
            return

        prompt = format_compaction_prompt(
            turns=[
                {"role": message.role, "content": message.content}
                for message in uncompacted
            ],
            previous_summary=existing.summary if existing else None,
            max_words=self._max_words,
        )
        # temperature=0: this is a transformation of text that already
        # exists, not a piece of writing. Sampling noise here shows up as
        # detail that drifts a little further from what was said on every
        # pass, and each pass reads the last one's output.
        summary = self._llm_service.generate_response(
            prompt, system_prompt=COMPACTION_SYSTEM_PROMPT, temperature=0
        )
        if not summary.strip():
            # An empty summary would replace a good one with nothing.
            # Leaving the high-water mark where it is means these turns are
            # simply retried next time.
            logger.warning(
                "Compaction produced an empty summary for conversation %s; keeping "
                "the previous one.",
                conversation_id,
            )
            return

        self._memory.save_summary(
            conversation_id,
            summary=summary.strip(),
            covered_through_id=uncompacted[-1].message_id,
        )


def get_conversation_compactor() -> ConversationCompactor | None:
    """
    Build a compactor from shared singletons, or None when compaction is
    switched off.

    None rather than a no-op object because the services treat the
    compactor as optional anyway - it is the one collaborator whose absence
    is a supported configuration rather than a mistake, and their tests
    construct them without one.
    """
    from ai.llm_service import get_llm_service
    from app.core.config import settings
    from memory.conversation_memory import get_conversation_memory

    if not settings.conversation_compaction_enabled:
        return None

    return ConversationCompactor(
        memory=get_conversation_memory(),
        llm_service=get_llm_service(),
        history_limit=settings.conversation_history_limit,
        trigger=settings.conversation_compaction_trigger,
        max_words=settings.conversation_summary_max_words,
    )
