"""
Prompts for the two chunking strategies that use a language model.

Both ask the model to do something a splitter cannot: understand the passage.
Both are written to fail safely, because they run over every chunk of every
upload and a bad answer must degrade to "no enrichment", never to a corrupted
index. That shapes the instructions - short outputs, a fixed shape, and an
explicit escape hatch when the passage doesn't suit the task.
"""

PROPOSITION_SYSTEM_PROMPT = """\
You break passages into propositions: standalone factual statements that can be \
understood with no other context.

Rules:
1. Each proposition states exactly one fact.
2. Resolve every reference. Replace pronouns and phrases like "the company", \
"this method" or "the above" with the specific thing they refer to, taken from \
the passage.
3. Use only information present in the passage. Never infer, generalise or add \
outside knowledge.
4. Keep the passage's own wording and numbers wherever possible.
5. Output one proposition per line. No numbering, no bullets, no preamble, no \
commentary.
6. If the passage contains no factual statements (a heading, a fragment, \
boilerplate), output nothing at all.\
"""


def format_proposition_prompt(passage: str, context: str = "") -> str:
    """
    Ask for the propositions in one passage.

    `context` is the document title and section path when known. It is what
    makes rule 2 possible: a passage saying "the trial was halted" cannot be
    made standalone from its own words, but can be from its heading.
    """
    header = f"Document context: {context}\n\n" if context else ""
    return f"{header}Passage:\n{passage}\n\nPropositions:"


CONTEXTUAL_SYSTEM_PROMPT = """\
You write a single short sentence that situates an excerpt within the document \
it came from, so the excerpt can be understood and searched on its own.

Rules:
1. One sentence, at most 25 words.
2. Say what the excerpt is about and where it sits in the document - the \
subject, the entity, the time period, the section it belongs to.
3. Use only the document context and the excerpt provided. Never invent \
details.
4. Output the sentence and nothing else: no preamble, no quotation marks, no \
explanation.\
"""


def format_contextual_prompt(document_context: str, excerpt: str) -> str:
    """Ask for the one-line context to prepend to a chunk before embedding."""
    return (
        f"Document context:\n{document_context}\n\n"
        f"Excerpt:\n{excerpt}\n\n"
        "Sentence situating the excerpt:"
    )
