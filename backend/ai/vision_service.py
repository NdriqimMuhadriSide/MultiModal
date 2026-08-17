"""
Vision service — the only file in the codebase that sends images to the
multimodal (vision-capable) model.

Responsibilities:
1. Load the API key / vision model name / base URL from environment
   variables (via app settings).
2. Create and hold the API client (same OpenAI-compatible client used by
   ai/llm_service.py - Groq exposes vision through the same chat.completions
   API, just with an image content block added to the message).
3. Accept raw image bytes + a user question.
4. Encode the image as a base64 data URL and send it, together with the
   question and the vision system prompt, to the vision model.
5. Return the generated text response (description + extracted info +
   answer, per the system prompt).

This module has no knowledge of FastAPI, HTTP, or file upload handling -
it just takes bytes in and returns text out. Multipart parsing lives in the
API layer; business rules (if any) live in the service layer.
"""
import base64
import re
from functools import lru_cache

from openai import OpenAI

from app.core.config import settings
from prompts.assistant_prompts import VISION_SYSTEM_PROMPT

# Maps common image MIME types to the subtype used in a base64 data URL.
# Anything not in this set is rejected before reaching the model.
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
}

# Some vision-capable models (e.g. Groq's qwen3.6 series) are "thinking"
# models that prepend a <think>...</think> reasoning block before the
# actual answer. That's internal scratch work, not meant for the API
# response, so it's stripped out before returning the text.
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_block(text: str) -> str:
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


class VisionService:
    """Wraps an OpenAI-compatible client for image + text (multimodal) completions."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file before "
                "calling the vision service."
            )

        # Same policy, and for the same reasons, as ai/llm_service.py - see
        # the comment there. A vision request carries an image and so is
        # slower than a text one, but not by an order of magnitude, and it
        # runs from the same threadpool.
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._model = model

    def analyze_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        question: str,
        system_prompt: str = VISION_SYSTEM_PROMPT,
        temperature: float | None = None,
    ) -> str:
        """
        Send an image and a question to the vision model and return the
        generated analysis.

        `system_prompt` defaults to the standard three-part prose format
        (VISION_SYSTEM_PROMPT) used by POST /vision/analyze. Callers with a
        different output contract - e.g. processors/streaming/vision_stream.py,
        which needs strict structured JSON instead of prose - can pass their
        own system prompt without this class needing a second client or a
        duplicated analyze method. This is the only supported way to
        customize model behavior here; everything else (encoding, the
        client, error handling) stays identical for every caller.

        `temperature` is unset by default. agents/vision_agent.py passes 0:
        it reads the reply for facts about a document, and a description
        that varies run to run makes the agent's answer unreproducible for
        no benefit. A caller wanting prose about a photograph should leave
        it alone.

        Raises:
            ValueError: if the question is empty, the image is empty, or the
                mime type is not supported.
            RuntimeError: if the API call fails.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty.")

        if not image_bytes:
            raise ValueError("image must not be empty.")

        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(
                f"Unsupported image type '{mime_type}'. Supported types: "
                f"{', '.join(sorted(SUPPORTED_IMAGE_MIME_TYPES))}."
            )

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_image}"

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                # Omitted entirely when None, so POST /vision/analyze and the
                # streaming analyzer keep sending exactly the request they
                # sent before this parameter existed.
                **({} if temperature is None else {"temperature": temperature}),
            )
        except Exception as exc:  # noqa: BLE001 - surface as a domain error
            raise RuntimeError(f"Vision request failed: {exc}") from exc

        raw_content = completion.choices[0].message.content or ""
        return _strip_think_block(raw_content)


@lru_cache
def get_vision_service() -> VisionService:
    """Return a cached VisionService instance built from app settings (Groq)."""
    return VisionService(
        api_key=settings.groq_api_key,
        model=settings.groq_vision_model,
        base_url=settings.groq_base_url,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
    )
