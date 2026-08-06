"""
Audio analysis prompts.

Split out into its own module, same reasoning as prompts/rag_prompts.py:
the audio pipeline is its own layer with its own prompt lifecycle, so its
wording can be tuned here without touching prompts/assistant_prompts.py or
any other feature's prompts.

The LLM analysis step (ai/llm_service.py, reused unchanged - see
app/services/audio_service.py) operates purely on the transcript text; it
has no awareness the text came from audio. format_audio_analysis_prompt()
is what turns a transcript + a question into the single user message sent
to that shared LLM service.

Default question: if the user doesn't provide one, the pipeline defaults
to a general-purpose summary instruction (DEFAULT_AUDIO_QUESTION) rather
than requiring a question on every request - matching the spec's "the user
may also provide their own custom question" (i.e. a question is optional).
"""

AUDIO_ANALYSIS_SYSTEM_PROMPT = (
    "You are a professional AI assistant that analyzes audio transcripts "
    "(e.g. meetings, interviews, voice notes). "
    "Base your answer only on the transcript provided - do not invent "
    "details that are not present in it. "
    "If the transcript is too short, unclear, or garbled to answer the "
    "question, say so explicitly rather than guessing. "
    "Write clearly and, where relevant, use short headings or bullet points "
    "(e.g. for action items or decisions)."
)

DEFAULT_AUDIO_QUESTION = (
    "Summarize this audio. Include the key points discussed, any decisions "
    "made, and any action items or next steps mentioned."
)

AUDIO_ANALYSIS_PROMPT_TEMPLATE = (
    "{system_prompt}\n\n"
    "Transcript:\n"
    "{transcript}\n\n"
    "Question:\n"
    "{question}"
)


def format_audio_analysis_prompt(transcript: str, question: str | None) -> str:
    """
    Fill AUDIO_ANALYSIS_PROMPT_TEMPLATE with the transcript and question.

    `question` falls back to DEFAULT_AUDIO_QUESTION when empty/None, so
    callers (app/services/audio_service.py) can always pass through
    whatever the client sent without checking for emptiness themselves.
    """
    effective_question = question.strip() if question and question.strip() else DEFAULT_AUDIO_QUESTION
    return AUDIO_ANALYSIS_PROMPT_TEMPLATE.format(
        system_prompt=AUDIO_ANALYSIS_SYSTEM_PROMPT,
        transcript=transcript,
        question=effective_question,
    )
