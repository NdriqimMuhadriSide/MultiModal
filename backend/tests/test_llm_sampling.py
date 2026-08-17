"""
Tests for the sampling parameter on the AI layer.

The point of these is the *absence* of a parameter as much as its presence.
LLMService and VisionService are shared by /chat, the RAG answer step, the
chunking helpers, the streaming analyzer and both agents, and only the
agents asked for determinism. Every other caller has to keep sending the
request it sent before this existed - which means no `temperature` key at
all, not `temperature` set to some default.
"""
from ai.llm_service import LLMService
from ai.vision_service import VisionService


class RecordingClient:
    """Stands in for the OpenAI client, capturing the kwargs of each call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Message:
            content = "a reply"

        class _Choice:
            message = _Message()
            delta = _Message()

        class _Completion:
            choices = [_Choice()]

        # The streaming call iterates the return value; the blocking one
        # reads `.choices` off it directly.
        return [_Completion()] if kwargs.get("stream") else _Completion()


def _llm() -> tuple[LLMService, RecordingClient]:
    service = LLMService(api_key="k", model="m")
    client = RecordingClient()
    service._client = client
    return service, client


def _vision() -> tuple[VisionService, RecordingClient]:
    service = VisionService(api_key="k", model="m")
    client = RecordingClient()
    service._client = client
    return service, client


# ---- The default must stay "send nothing" ----------------------------------


def test_generate_response_sends_no_temperature_by_default():
    service, client = _llm()
    service.generate_response("hello")

    assert "temperature" not in client.calls[0]


def test_analyze_image_sends_no_temperature_by_default():
    service, client = _vision()
    service.analyze_image(b"\x89PNG", "image/png", "what is this?")

    assert "temperature" not in client.calls[0]


def test_stream_response_sends_no_temperature_by_default():
    service, client = _llm()
    list(service.stream_response("hello"))

    assert "temperature" not in client.calls[0]


# ---- And an explicit value must get through --------------------------------


def test_generate_response_forwards_an_explicit_temperature():
    service, client = _llm()
    service.generate_response("hello", temperature=0.0)

    assert client.calls[0]["temperature"] == 0.0


def test_analyze_image_forwards_an_explicit_temperature():
    service, client = _vision()
    service.analyze_image(b"\x89PNG", "image/png", "q", temperature=0.0)

    assert client.calls[0]["temperature"] == 0.0


def test_stream_response_forwards_an_explicit_temperature():
    service, client = _llm()
    list(service.stream_response("hello", temperature=0.2))

    assert client.calls[0]["temperature"] == 0.2
    # The streaming flag is still set - the new kwarg must not displace it.
    assert client.calls[0]["stream"] is True
