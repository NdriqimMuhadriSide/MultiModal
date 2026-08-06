"""
LLM service — the only file in the codebase that talks to the LLM provider.

Responsibilities:
1. Load the API key / model name / base URL from environment variables
   (via app settings).
2. Create and hold the API client.
3. Accept plain text input.
4. Send it to the chat completion model.
5. Return the generated text response.

Currently configured for Groq (free tier, no billing required for a
learning project). Groq exposes an OpenAI-compatible API, so we reuse the
`openai` SDK and just point it at Groq's base URL instead of OpenAI's.
Switching back to real OpenAI later only means changing the constructor
args passed in `get_llm_service()` - the rest of the app is unaffected.

Every request is sent with a system prompt (see `prompts/assistant_prompts.py`)
that sets the assistant's behavior/persona, followed by the user's message.
This module has no knowledge of FastAPI, HTTP, or business rules. It is a
thin, reusable wrapper so the rest of the app never imports `openai` directly.
"""
from functools import lru_cache

from openai import OpenAI

from app.core.config import settings
from prompts.assistant_prompts import SYSTEM_PROMPT


class LLMService:
    """Wraps an OpenAI-compatible client for simple text-in / text-out completions."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file before "
                "calling the LLM service."
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def generate_response(
        self, user_message: str, history: list[dict[str, str]] | None = None
    ) -> str:
        """
        Send a user message to the model, prefixed with the shared system
        prompt and (optionally) prior conversation turns, and return the
        generated text.

        `history` is a list of {"role": "user"|"assistant", "content": str}
        dicts, oldest first - this is exactly the shape
        memory/conversation_memory.py's ConversationMemory.get_history
        produces once converted, so the memory layer can be inserted here
        without this class needing to know anything about how/where
        history is stored.

        Raises:
            ValueError: if `user_message` is empty.
            RuntimeError: if the API call fails.
        """
        if not user_message or not user_message.strip():
            raise ValueError("user_message must not be empty.")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a domain error
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        return completion.choices[0].message.content or ""


@lru_cache
def get_llm_service() -> LLMService:
    """Return a cached LLMService instance built from app settings (Groq)."""
    return LLMService(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
    )
