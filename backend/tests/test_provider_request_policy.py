"""
The retry and timeout policy the three provider clients are built with.

Both values have SDK defaults, and the point of these tests is that this
project's choices are *its* choices rather than whatever the SDK ships
next. An openai release changing DEFAULT_TIMEOUT should not silently change
how long a hung request holds a threadpool worker here.
"""
import pytest

from ai.llm_service import LLMService
from ai.transcription_service import TranscriptionService
from ai.vision_service import VisionService

SERVICES = [LLMService, VisionService, TranscriptionService]


@pytest.mark.parametrize("service_class", SERVICES)
def test_a_missing_key_is_still_rejected_before_any_client_is_built(service_class):
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        service_class(api_key="", model="a-model")


@pytest.mark.parametrize("service_class", SERVICES)
def test_retries_are_passed_to_the_client(service_class):
    service = service_class(api_key="k", model="a-model", max_retries=5)

    assert service._client.max_retries == 5


@pytest.mark.parametrize("service_class", SERVICES)
def test_the_timeout_is_passed_to_the_client(service_class):
    service = service_class(api_key="k", model="a-model", timeout=12.5)

    assert service._client.timeout == 12.5


@pytest.mark.parametrize("service_class", SERVICES)
def test_retries_can_be_switched_off_entirely(service_class):
    # 0 has to mean "one attempt", not "fall back to the SDK default" - a
    # setting that silently ignores its own zero is worse than no setting.
    service = service_class(api_key="k", model="a-model", max_retries=0)

    assert service._client.max_retries == 0


def test_the_default_timeouts_reflect_what_each_call_actually_does():
    chat = LLMService(api_key="k", model="a-model")
    vision = VisionService(api_key="k", model="a-model")
    transcription = TranscriptionService(api_key="k", model="a-model")

    # Chat and vision answer in seconds. Transcription's duration scales
    # with the *upload* - a 25MB recording is minutes of audio - so cutting
    # it off at the chat timeout would fail every long file by design.
    assert chat._client.timeout == 60.0
    assert vision._client.timeout == 60.0
    assert transcription._client.timeout == 300.0


def test_the_defaults_are_not_the_sdks():
    from openai._constants import DEFAULT_TIMEOUT

    # The SDK's read timeout is 600 seconds. These calls are made from
    # synchronous routes in FastAPI's threadpool, so inheriting that means
    # one hung connection holds a worker for ten minutes - and retries
    # multiply it. This assertion exists to fail loudly if someone drops
    # the explicit argument.
    assert LLMService(api_key="k", model="a-model")._client.timeout != DEFAULT_TIMEOUT


def test_the_factories_use_the_configured_policy(monkeypatch):
    from app.core import config
    from ai import llm_service, transcription_service, vision_service

    monkeypatch.setattr(config.settings, "groq_api_key", "test-key")
    monkeypatch.setattr(config.settings, "llm_max_retries", 4)
    monkeypatch.setattr(config.settings, "llm_timeout_seconds", 30.0)
    monkeypatch.setattr(config.settings, "transcription_timeout_seconds", 120.0)
    # These are @lru_cache'd, so a service built by an earlier test would
    # otherwise be handed back with the old policy on it.
    llm_service.get_llm_service.cache_clear()
    vision_service.get_vision_service.cache_clear()
    transcription_service.get_transcription_service.cache_clear()

    try:
        assert llm_service.get_llm_service()._client.max_retries == 4
        assert llm_service.get_llm_service()._client.timeout == 30.0
        assert vision_service.get_vision_service()._client.timeout == 30.0
        assert transcription_service.get_transcription_service()._client.timeout == 120.0
    finally:
        llm_service.get_llm_service.cache_clear()
        vision_service.get_vision_service.cache_clear()
        transcription_service.get_transcription_service.cache_clear()
