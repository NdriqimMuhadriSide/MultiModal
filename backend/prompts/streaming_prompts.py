"""
Streaming vision prompts.

Split into its own module, same reasoning as prompts/rag_prompts.py and
prompts/audio_prompts.py: the streaming pipeline is its own layer with its
own prompt lifecycle, tunable here without touching any other feature's
prompts.

STREAM_OBSERVATION_SYSTEM_PROMPT asks the vision model to return a strict
JSON object (not free-form prose) - unlike ai/vision_service.py's regular
VISION_SYSTEM_PROMPT (three labeled prose sections), because
processors/streaming/vision_stream.py needs machine-parseable structured
fields (objects/text_detected/ui_elements/activities/warnings/
important_changes) to populate the API response's `observations` list and
to feed the context ring buffer in stream_processor.py - free-form prose
would require fragile re-parsing to extract those fields.
"""

STREAM_OBSERVATION_SYSTEM_PROMPT = (
    "You are a real-time vision analysis system watching a live camera or "
    "screen-share feed, one sampled frame at a time. "
    "Analyze the image and respond with ONLY a single valid JSON object - "
    "no markdown, no code fences, no commentary before or after it - "
    "matching exactly this shape:\n\n"
    "{\n"
    '  "objects": ["short noun phrases for physical/visual objects visible"],\n'
    '  "text_detected": ["any readable text, error messages, or labels in the image"],\n'
    '  "ui_elements": ["visible UI elements if this looks like a screen share, e.g. '
    '\'error dialog\', \'terminal window\', \'browser tab\'"],\n'
    '  "activities": ["short phrases describing what appears to be happening"],\n'
    '  "warnings": ["anything that looks like an error, risk, or problem worth flagging"],\n'
    '  "important_changes": ["notable differences from the recent context below, if any"]\n'
    "}\n\n"
    "Leave a field as an empty list if nothing applies - never omit a field, "
    "and never invent objects/text/activities that are not actually visible. "
    "If the image is blank, static, or unclear, say so honestly in "
    '"activities" rather than guessing.'
)

# {recent_context} is the short-term context ring buffer (recent frames'
# observations for this session, oldest first) built by
# processors/streaming/stream_processor.py - giving the model continuity
# across frames (e.g. "Frame 1: person enters room. Frame 2: person sits
# at desk.") so it can reason about change over time, not just describe
# one frame in isolation. {question} is optional - format_stream_prompt()
# falls back to a generic "describe what is happening" instruction when
# the user didn't ask anything specific for this frame.
STREAM_ANALYSIS_PROMPT_TEMPLATE = (
    "Recent observations from this session (oldest first, may be empty for "
    "the first frame):\n"
    "{recent_context}\n\n"
    "Question about the current frame: {question}"
)

DEFAULT_STREAM_QUESTION = "Describe what is currently visible and note anything noteworthy."


def format_stream_prompt(recent_context: str, question: str | None) -> str:
    """
    Fill STREAM_ANALYSIS_PROMPT_TEMPLATE with the session's recent context
    and the (optional) user question for the current frame.
    """
    effective_question = question.strip() if question and question.strip() else DEFAULT_STREAM_QUESTION
    return STREAM_ANALYSIS_PROMPT_TEMPLATE.format(
        recent_context=recent_context or "(none yet - this is the first frame)",
        question=effective_question,
    )
