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
from collections.abc import Iterator
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
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> str:
        """
        Send a user message to the model, prefixed with a system prompt and
        (optionally) prior conversation turns, and return the generated text.

        `system_prompt` defaults to the shared assistant prompt. Callers with a
        different output contract - rag/chunking/propositional.py wants one
        fact per line, rag/chunking/contextual.py wants a single sentence -
        pass their own, exactly as ai/vision_service.py already allows. This is
        the only supported way to change the model's behaviour here; the
        client, error handling and message assembly stay identical for every
        caller.

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
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(user_message, history, system_prompt),
            )
        except Exception as exc:  # noqa: BLE001 - surface as a domain error
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        return completion.choices[0].message.content or ""

    def stream_response(
        self, user_message: str, history: list[dict[str, str]] | None = None
    ) -> Iterator[str]:
        """
        Same request as `generate_response`, yielded token by token instead
        of returned whole.

        Kept as a separate method rather than a `stream=True` flag on
        `generate_response`, because the two have genuinely different return
        types (`str` vs a generator) and different failure timing. Callers
        that want the complete text - the agent's router, which asks for a
        one-word classification, and the non-streaming /chat endpoint the
        offline outbox replays through - should keep using
        `generate_response`; there is nothing to gain from streaming a
        response nobody displays incrementally.

        Failure timing is the subtle part. `generate_response` either raises
        or returns a complete answer. Here the request is only *started*
        when the first item is pulled, so a connection error surfaces mid
        iteration, after the caller has likely already sent HTTP headers.
        Consumers therefore have to handle a RuntimeError raised partway
        through, not just at the call site.

        Raises:
            ValueError: if `user_message` is empty (raised eagerly, before
                any iteration, since it needs no network call to detect).
            RuntimeError: if the API call fails - possibly mid-stream.
        """
        messages = self._build_messages(user_message, history)

        def iterator() -> Iterator[str]:
            try:
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    stream=True,
                )
                for chunk in stream:
                    # The final chunk carries finish_reason and an empty
                    # delta; some providers also send a leading role-only
                    # chunk. Both have `content` of None, not "".
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            except Exception as exc:  # noqa: BLE001 - surface as a domain error
                raise RuntimeError(f"LLM stream failed: {exc}") from exc

        return iterator()

    def _build_messages(
        self,
        user_message: str,
        history: list[dict[str, str]] | None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> list[dict[str, str]]:
        """
        Assemble the message list both request paths send: system prompt,
        then prior turns, then the new message.

        Validation lives here so streaming rejects an empty message at the
        call site rather than on first iteration - by which point the HTTP
        response has already begun and a 400 is no longer possible.
        """
        if not user_message or not user_message.strip():
            raise ValueError("user_message must not be empty.")

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages


@lru_cache
def get_llm_service() -> LLMService:
    """Return a cached LLMService instance built from app settings (Groq)."""
    return LLMService(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
    )
