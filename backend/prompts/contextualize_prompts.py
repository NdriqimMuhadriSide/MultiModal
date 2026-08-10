"""
Query contextualization prompt.

Used by rag/contextualizer.py to turn a follow-up question into one that
stands on its own, so retrieval has something to search for.

The distinction from prompts/query_expansion_prompts.py is worth stating,
because both are "an LLM rewrites the query":

    expansion          one standalone question -> several phrasings of it.
                       Widens what retrieval can match. Breadth.
    contextualization  a dependent question + the conversation -> one
                       standalone question. Recovers what the user meant.
                       Resolution.

Expansion on "and if I never use them?" would produce three equally
meaningless rewrites, because the missing information is not in the question
at all - it is in the turn before it. Only contextualization can put it back.

Every rule in the prompt is load-bearing:

    "self-contained"     the output is going to a search index that has no
                         memory. "it", "them", "that one" retrieve nothing.
    "return it
     unchanged"          most questions are already standalone, and rewriting
                         a good question can only damage it. The instruction
                         to leave those alone is what makes it safe to run
                         this on every turn.
    "keep identifiers"   same hazard as expansion: retrieval is hybrid, and a
                         rewrite that turns ERR-4021 into "the upload error"
                         has deleted the BM25 half's only signal.
    "do not answer it"   a model handed a question and a conversation will
                         cheerfully answer instead of rewriting. That output
                         then gets embedded and searched for, which retrieves
                         whatever the model's own answer happens to resemble.
"""

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question so that it can be understood on its own, "
    "without the conversation it came from."
)

CONTEXTUALIZE_TEMPLATE = (
    "{system_prompt}\n\n"
    "Conversation so far:\n"
    "{turns}\n\n"
    "Follow-up question:\n"
    "{question}\n\n"
    "Rewrite the follow-up question so it can be understood on its own.\n"
    "Rules:\n"
    "- Change as few words as possible. Replace only the words that point at "
    "the conversation - \"it\", \"them\", \"that\", \"those\", \"then\" - with the "
    "specific thing they refer to. Leave every other word alone.\n"
    "- Do NOT add any fact, number, detail or explanation from the earlier "
    "answers. The rewrite must not become longer than the original question "
    "plus the few words needed to name its subject.\n"
    "- If the question already stands on its own, return it exactly unchanged.\n"
    "- Keep every exact identifier unchanged: error codes, product names, "
    "numbers, abbreviations. Never introduce an identifier that the follow-up "
    "question did not itself mention.\n"
    "- Do not answer the question. Do not explain. Output the rewritten "
    "question only, on one line.\n\n"
    "Example:\n"
    "  conversation: user asked about annual leave accrual\n"
    "  follow-up: \"how far ahead do I have to request it?\"\n"
    "  rewrite: \"how far ahead do I have to request annual leave?\""
)

# Enough to resolve a reference without burying the instructions - the same
# window prompts/assistant_prompts.py uses for routing, and for the same
# reason: what "it" refers to is almost always in the last turn or two.
CONTEXTUALIZE_HISTORY_TURNS = 4


def format_contextualize_prompt(question: str, history: list[dict[str, str]]) -> str:
    """
    Fill CONTEXTUALIZE_TEMPLATE with the tail of `history` (oldest first, the
    shape the memory layer produces) and the follow-up question.
    """
    turns = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[-CONTEXTUALIZE_HISTORY_TURNS:]
    )
    return CONTEXTUALIZE_TEMPLATE.format(
        system_prompt=CONTEXTUALIZE_SYSTEM_PROMPT,
        turns=turns,
        question=question,
    )
