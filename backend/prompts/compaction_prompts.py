"""
Prompts for conversation compaction.

One job: take the turns that have fallen out of the history window - plus
whatever summary already covers the turns before those - and produce a
single replacement summary.

The instructions are shaped by what the summary is *for*. It is not a
recap for a person to read; it is the only trace of those turns the model
will ever see again, so the qualities that matter are unusual:

    - Facts over narrative. "The user's deploy target is Vercel" survives
      being read out of order; "they then explained their setup" does not.
    - The user's own words for names, numbers, ids and file paths. These
      are exactly what a later turn refers back to, and a paraphrase of an
      identifier is a lost identifier.
    - Open threads kept. A question the assistant asked and the user hasn't
      answered is the single most useful thing to carry forward, and the
      easiest for a summariser to drop as "not a fact".
    - No invention. A summary is downstream of everything, so anything
      invented here is indistinguishable from something the user said, for
      the rest of the conversation.
"""

COMPACTION_SYSTEM_PROMPT = (
    "You compress conversation history. You output only the compressed "
    "record - no preamble, no commentary, no offer to help."
)

_TEMPLATE = (
    "Below is part of a conversation between a user and an assistant that "
    "is about to fall out of the assistant's memory. Rewrite it as a "
    "compact record the assistant can rely on later.\n\n"
    "{previous_block}"
    "TURNS TO COMPRESS:\n"
    "{turns}\n\n"
    "Rules:\n"
    "- Keep facts, decisions, preferences, and anything the user asked for "
    "that has not been delivered yet.\n"
    "- Keep names, numbers, identifiers, file paths and quoted strings "
    "exactly as they appeared. Do not paraphrase them.\n"
    "- Keep any question left unanswered, and say who asked it.\n"
    "- Drop pleasantries, restatements, and the assistant's reasoning about "
    "how it answered.\n"
    "- Write plain statements, not a narrative. No 'the user then asked'.\n"
    "- Invent nothing. If something is unclear, leave it out.\n"
    "- Stay under {max_words} words.\n\n"
    "Output the record and nothing else."
)

_PREVIOUS_BLOCK = (
    "This is the record so far, covering everything before the turns "
    "below. Fold it into your answer - your output replaces it "
    "entirely, so anything you leave out is forgotten:\n"
    "{previous_summary}\n\n"
)


def format_compaction_prompt(
    turns: list[dict[str, str]],
    previous_summary: str | None = None,
    max_words: int = 200,
) -> str:
    """
    Build the prompt that turns `turns` (oldest first) into a summary.

    `previous_summary` is the existing record, if there is one. It is
    included rather than kept alongside because the alternative - a chain
    of summaries - grows without bound and eventually costs more than the
    turns it replaced.
    """
    rendered = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in turns
    )
    previous_block = (
        _PREVIOUS_BLOCK.format(previous_summary=previous_summary)
        if previous_summary
        else ""
    )
    return _TEMPLATE.format(
        previous_block=previous_block, turns=rendered, max_words=max_words
    )
