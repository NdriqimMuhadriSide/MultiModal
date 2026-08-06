"""
RAG prompt templates.

Split out of prompts/assistant_prompts.py into its own module, since the
RAG pipeline (rag/rag_service.py) is its own layer with its own prompt
lifecycle - this is the only file that should need to change if the RAG
system prompt wording is tuned, independent of the general assistant or
vision prompts.

{context} is filled in by rag/context_builder.py's build_context() output
(chunks labeled with Document/Page/Content); {question} is the user's
original question. The explicit "never invent information" / "say you do
not know" instructions are what prevent the model from falling back on its
own general knowledge and hallucinating an answer that isn't grounded in
the retrieved documents.
"""

RAG_SYSTEM_PROMPT = (
    "You are an AI knowledge assistant.\n"
    "Answer only using the provided context.\n"
    "If the answer is not available, say that you do not know.\n"
    "Never invent information."
)

RAG_PROMPT_TEMPLATE = (
    "{system_prompt}\n\n"
    "Context:\n"
    "{context}\n\n"
    "Question:\n"
    "{question}"
)


def format_rag_prompt(context: str, question: str) -> str:
    """Fill RAG_PROMPT_TEMPLATE with the RAG system prompt, context, and question."""
    return RAG_PROMPT_TEMPLATE.format(
        system_prompt=RAG_SYSTEM_PROMPT, context=context, question=question
    )
