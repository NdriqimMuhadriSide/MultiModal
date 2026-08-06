"""
Streaming vision analysis.

Responsibility: analyze one sampled frame and return a structured
observation - the "Vision layer" in the streaming architecture, sitting
between the Frame Processing layer (stream_processor.py) and the LLM layer
(ai/vision_service.py, reused unchanged - no second OpenAI/Groq client is
created here or anywhere in this package).

Unlike ai/vision_service.py's default prose output (three labeled
sections), this module asks the model for strict JSON
(prompts/streaming_prompts.STREAM_OBSERVATION_SYSTEM_PROMPT) and parses it
into a StreamObservation - the shape stream_processor.py needs to populate
the API response's `observations` list and to build recent-context strings
for future frames. Parsing failures (the model didn't return valid JSON) are
handled gracefully by falling back to a single-field observation containing
the raw text, rather than raising and failing the whole frame - a
malformed model response is far more likely than a validation bug, and a
degraded-but-present result is more useful to the caller than an error.
"""
import json
from dataclasses import dataclass, field

from ai.vision_service import VisionService
from prompts.streaming_prompts import STREAM_OBSERVATION_SYSTEM_PROMPT, format_stream_prompt


@dataclass
class StreamObservation:
    """Structured observations extracted from one analyzed frame."""

    objects: list[str] = field(default_factory=list)
    text_detected: list[str] = field(default_factory=list)
    ui_elements: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    important_changes: list[str] = field(default_factory=list)
    raw_text: str = ""

    def as_flat_list(self) -> list[str]:
        """
        Flatten every observed field into one ordered list of short
        strings - this is what the API response's `observations: [...]`
        field is built from (see app/schemas/stream.py), and what
        stream_processor.py stores in a session's context ring buffer as
        a compact one-line summary for future frames to reason over.
        """
        return [
            *self.objects,
            *self.text_detected,
            *self.ui_elements,
            *self.activities,
            *self.warnings,
            *self.important_changes,
        ]

    def summary_line(self) -> str:
        """A compact, single-line summary for the context ring buffer (e.g. 'Frame 3: ...')."""
        flat = self.as_flat_list()
        if flat:
            return "; ".join(flat)
        return self.raw_text or "(no notable observations)"


class VisionStreamAnalyzer:
    """Analyzes one sampled frame and returns a structured StreamObservation."""

    def __init__(self, vision_service: VisionService) -> None:
        self._vision_service = vision_service

    def analyze_frame(
        self, frame_bytes: bytes, mime_type: str, recent_context: str, question: str | None
    ) -> StreamObservation:
        """
        Analyze one frame, using `recent_context` (prior frames'
        summaries for this session, oldest first) to let the model reason
        about change over time, and `question` to focus the analysis if
        the user asked something specific about the current frame.

        Raises:
            ValueError: if frame_bytes is empty or mime_type is unsupported
                (propagated from ai.vision_service).
            RuntimeError: if the vision API call fails (propagated from
                ai.vision_service).
        """
        prompt = format_stream_prompt(recent_context=recent_context, question=question)

        raw_response = self._vision_service.analyze_image(
            image_bytes=frame_bytes,
            mime_type=mime_type,
            question=prompt,
            system_prompt=STREAM_OBSERVATION_SYSTEM_PROMPT,
        )

        return _parse_observation(raw_response)


def _parse_observation(raw_response: str) -> StreamObservation:
    """
    Parse the model's JSON response into a StreamObservation.

    Falls back to a raw_text-only observation (rather than raising) if the
    model didn't return valid/expected JSON - see this module's docstring
    for why that's the right default here.
    """
    try:
        data = json.loads(_strip_code_fence(raw_response))
    except (json.JSONDecodeError, TypeError):
        return StreamObservation(raw_text=raw_response.strip())

    if not isinstance(data, dict):
        return StreamObservation(raw_text=raw_response.strip())

    return StreamObservation(
        objects=_string_list(data.get("objects")),
        text_detected=_string_list(data.get("text_detected")),
        ui_elements=_string_list(data.get("ui_elements")),
        activities=_string_list(data.get("activities")),
        warnings=_string_list(data.get("warnings")),
        important_changes=_string_list(data.get("important_changes")),
        raw_text=raw_response.strip(),
    )


def _string_list(value: object) -> list[str]:
    """Coerce a JSON field into a list[str], tolerating a missing/wrong-typed field."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _strip_code_fence(text: str) -> str:
    """
    Strip a ```json ... ``` or ``` ... ``` wrapper if the model added one
    despite being asked not to - some models habitually wrap JSON output
    in a markdown code fence even when told not to.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return stripped
