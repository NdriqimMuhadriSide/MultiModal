"""
Query expansion prompt.

Used by rag/query_expansion.py to turn one question into several, so a
document that phrases an answer differently from the question still gets
found. The prompt is doing more work than it looks like, and every clause in
it is load-bearing:

    "different wording"      the whole point - a variant that repeats the
                             question retrieves the same chunks and wastes
                             a slot in the fusion
    "keep exact identifiers" the critical one. Retrieval here is hybrid, and
                             the BM25 half matches literal tokens. A variant
                             that helpfully rewrites "ERR-4021" as "the
                             upload error" has destroyed the only signal
                             that found the right chunk - so the model is
                             told, twice, to carry codes through verbatim
    "one per line, no
     numbering, no preamble" the output is parsed, not read. A model that
                             answers "Sure! Here are 3 queries:" costs a
                             variant slot to a line that retrieves nothing

Deliberately not asking for sub-questions or a decomposition of a
multi-part question. That is a different technique with a different failure
mode (sub-answers that have to be recombined), and mixing the two into one
prompt gets a model that does neither reliably.
"""

QUERY_EXPANSION_SYSTEM_PROMPT = (
    "You rewrite a user's search question into alternative phrasings for a "
    "document search engine."
)

QUERY_EXPANSION_TEMPLATE = (
    "{system_prompt}\n\n"
    "Write {count} alternative versions of the question below. Rules:\n"
    "- Each must ask for the same information, in different wording.\n"
    "- Keep every exact identifier unchanged: error codes, product names, "
    "numbers, abbreviations. Never paraphrase a code such as ERR-4021.\n"
    "- Use terms a document might use, not only the user's terms.\n"
    "- Output exactly {count} lines, one question per line.\n"
    "- No numbering, no bullet points, no preamble, no explanation.\n\n"
    "Question:\n"
    "{question}"
)


def format_query_expansion_prompt(question: str, count: int) -> str:
    """Fill QUERY_EXPANSION_TEMPLATE for `count` alternative phrasings."""
    return QUERY_EXPANSION_TEMPLATE.format(
        system_prompt=QUERY_EXPANSION_SYSTEM_PROMPT,
        question=question,
        count=count,
    )
