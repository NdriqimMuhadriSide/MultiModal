"""
Tests for the vision & OCR agent (agents/vision_agent.py).

The loop machinery is covered in test_research_agent.py - it is the same
AgentLoop - so nothing here re-tests parsing or termination. What is tested
is the part that is this agent's own: which reading tool it may call, what
happens when character recognition is unavailable or finds nothing, and the
caching that stops it paying twice for the same look at the image.

No real model, no real Tesseract: the LLM's replies are scripted, the
vision service is a fake, and rag.ocr is monkeypatched.
"""
import pytest

from agents.vision_agent import VisionAgent
from rag import ocr
from rag.retriever import RetrievedChunk

PNG = "image/png"


class FakeLLMService:
    """Replays scripted replies; raises if the loop takes an unscripted turn."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []
        self.temperatures: list[float | None] = []

    def generate_response(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        self.prompts.append(user_message)
        self.system_prompts.append(system_prompt)
        self.temperatures.append(temperature)
        if not self.replies:
            raise AssertionError(
                f"LLM called {len(self.prompts)} times but only "
                f"{len(self.prompts) - 1} replies were scripted."
            )
        return self.replies.pop(0)


class FakeVisionService:
    def __init__(self, answer: str = "A restaurant receipt, total near the bottom.") -> None:
        self.answer = answer
        self.calls: list[tuple[bytes, str, str]] = []
        self.temperatures: list[float | None] = []

    def analyze_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        question: str,
        temperature: float | None = None,
    ) -> str:
        self.calls.append((image_bytes, mime_type, question))
        self.temperatures.append(temperature)
        return self.answer


class FakeRetriever:
    def __init__(self, results: list[list[RetrievedChunk]] | None = None) -> None:
        self.results = list(results or [])
        self.queries: list[str] = []

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.queries.append(question)
        return self.results.pop(0) if self.results else []


def _chunk(chunk_id: str, text: str = "Meals are capped at 50 per person.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, filename="expenses.pdf", page=4, score=0.77
    )


def _agent(llm, vision=None, retriever=None, **overrides) -> VisionAgent:
    return VisionAgent(
        llm_service=llm,
        vision_service=vision or FakeVisionService(),
        retriever=retriever or FakeRetriever(),
        **{"max_steps": 5, "search_top_k": 3, **overrides},
    )


def _inspect(question: str) -> str:
    return (
        f'Thought: Let me look.\nAction: {{"tool": "inspect_image", '
        f'"input": {{"question": "{question}"}}}}'
    )


def _read() -> str:
    return 'Thought: I need exact digits.\nAction: {"tool": "read_text", "input": {}}'


def _search(query: str) -> str:
    return (
        f'Thought: Checking the policy.\nAction: {{"tool": "search_knowledge_base", '
        f'"input": {{"query": "{query}"}}}}'
    )


def _finish(answer: str) -> str:
    return f'Thought: Done.\nAction: {{"tool": "finish", "input": {{"answer": "{answer}"}}}}'


@pytest.fixture
def ocr_returns(monkeypatch):
    """Make rag.ocr behave as if Tesseract is present and returns `text`."""

    def _set(text: str):
        monkeypatch.setattr(ocr, "is_available", lambda: True)
        monkeypatch.setattr(ocr, "ocr_image", lambda image_bytes: text)

    return _set


# ---- Input validation ------------------------------------------------------


def test_run_rejects_an_empty_image():
    with pytest.raises(ValueError, match="image must not be empty"):
        _agent(FakeLLMService([])).run("what is this?", b"", PNG)


def test_run_rejects_an_unsupported_image_type():
    """Rejected up front, not on whichever step happens to call the vision model."""
    with pytest.raises(ValueError, match="Unsupported image type"):
        _agent(FakeLLMService([])).run("what is this?", b"bytes", "image/tiff")


def test_run_rejects_an_empty_question():
    with pytest.raises(ValueError, match="must not be empty"):
        _agent(FakeLLMService([])).run("  ", b"bytes", PNG)


# ---- The reading decision --------------------------------------------------


def test_the_canonical_run_inspects_then_reads_then_answers(ocr_returns):
    ocr_returns("TOTAL      84.50\nDATE  2026-03-11")
    vision = FakeVisionService()
    llm = FakeLLMService(
        [_inspect("what kind of document is this?"), _read(), _finish("The total is 84.50.")]
    )

    result = _agent(llm, vision).run("what is the total?", b"img", PNG)

    assert [step.tool for step in result.steps] == ["inspect_image", "read_text", "finish"]
    assert vision.calls[0][2] == "what kind of document is this?"
    assert "TOTAL      84.50" in result.steps[1].observation
    assert result.answer == "The total is 84.50."
    assert result.stopped_because == "finished"


def test_read_text_is_cached_for_the_run(ocr_returns, monkeypatch):
    """It takes no arguments, so a second call is necessarily identical."""
    calls = []

    monkeypatch.setattr(ocr, "is_available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_image", lambda image_bytes: calls.append(1) or "INVOICE 22"
    )

    llm = FakeLLMService([_read(), _read(), _finish("done")])
    result = _agent(llm).run("q", b"img", PNG)

    assert len(calls) == 1
    assert "You already ran read_text" in result.steps[1].observation


def test_missing_tesseract_gives_an_actionable_observation(monkeypatch):
    """Not an empty result - the model has to know the difference."""
    monkeypatch.setattr(ocr, "is_available", lambda: False)
    monkeypatch.setattr(
        ocr, "unavailable_reason", lambda: "Tesseract is not installed."
    )

    llm = FakeLLMService(
        [_read(), _finish("no idea"), _inspect("describe it"), _finish("a receipt")]
    )
    result = _agent(llm).run("what is the total?", b"img", PNG)

    observation = result.steps[0].observation
    assert "Tesseract is not installed." in observation
    assert "Use inspect_image instead" in observation
    # A failed read is not a look: finishing straight after it is nudged.
    assert "have not looked at the image" in result.steps[1].observation
    assert result.answer == "a receipt"


def test_an_empty_ocr_result_is_explained_not_reported_blank(ocr_returns):
    """A photo or a diagram - which calls for a different tool, not a retry."""
    ocr_returns("   \n  ")
    llm = FakeLLMService(
        [_read(), _finish("blank"), _inspect("describe it"), _finish("a photograph")]
    )
    result = _agent(llm).run("what does it say?", b"img", PNG)

    assert "found no printed text" in result.steps[0].observation
    assert "do not quote exact values" in result.steps[0].observation
    # An empty grid is not a look either - the agent obtained no content.
    assert "have not looked at the image" in result.steps[1].observation
    assert result.answer == "a photograph"


def test_long_ocr_output_is_truncated_with_a_pointer(ocr_returns):
    ocr_returns("X" * 4000)
    llm = FakeLLMService([_read(), _finish("done")])
    result = _agent(llm).run("q", b"img", PNG)

    observation = result.steps[0].observation
    assert len(observation) < 3200
    assert "was cut for length" in observation


def test_repeating_an_image_question_is_refused(ocr_returns):
    """Unlike a repeated search, this would be a paid provider round trip."""
    vision = FakeVisionService()
    llm = FakeLLMService(
        [_inspect("what is this?"), _inspect("What   IS this?"), _finish("done")]
    )
    result = _agent(llm, vision).run("q", b"img", PNG)

    assert len(vision.calls) == 1
    assert "You already asked the image" in result.steps[1].observation


def test_inspect_image_without_a_question_is_explained():
    llm = FakeLLMService(
        [
            'Action: {"tool": "inspect_image", "input": {}}',
            _inspect("what is this?"),
            _finish("done"),
        ]
    )
    result = _agent(llm).run("q", b"img", PNG)

    assert "needs a question" in result.steps[0].observation
    assert result.answer == "done"


# ---- Cross-modal grounding -------------------------------------------------


def test_the_agent_can_check_the_image_against_the_knowledge_base(ocr_returns):
    """The reason this is more than an image reader."""
    ocr_returns("TOTAL 84.50")
    retriever = FakeRetriever([[_chunk("c1")]])
    llm = FakeLLMService(
        [_read(), _search("meal expense limit"), _finish("84.50 exceeds the 50 cap [E1].")]
    )

    result = _agent(llm, retriever=retriever).run("is this within policy?", b"img", PNG)

    assert retriever.queries == ["meal expense limit"]
    assert "[E1]" in result.steps[1].observation
    assert [source.chunk_id for source in result.sources] == ["c1"]
    assert result.answer == "84.50 exceeds the 50 cap [E1]."


def test_an_image_only_question_produces_no_citations(ocr_returns):
    """Nothing from the picture gets a source record - there is nothing to cite."""
    ocr_returns("HELLO")
    llm = FakeLLMService([_read(), _finish("It says HELLO.")])
    result = _agent(llm).run("what does it say?", b"img", PNG)

    assert result.sources == []


# ---- Policy and per-run state ----------------------------------------------


def test_answering_without_looking_is_nudged_once_then_accepted():
    llm = FakeLLMService([_finish("a receipt, probably"), _finish("not an image question")])
    result = _agent(llm).run("q", b"img", PNG)

    assert "have not looked at the image" in result.steps[0].observation
    assert result.answer == "not an image question"


def test_answering_after_looking_is_not_nudged(ocr_returns):
    ocr_returns("TEXT")
    llm = FakeLLMService([_read(), _finish("grounded answer")])
    result = _agent(llm).run("q", b"img", PNG)

    assert len(result.steps) == 2
    assert result.answer == "grounded answer"


def test_the_prompt_tells_the_model_an_image_is_attached():
    llm = FakeLLMService([_finish("x"), _finish("x")])
    _agent(llm).run("q", b"img", PNG)

    assert "An image of type image/png is attached" in llm.prompts[0]


def test_each_run_clears_the_ocr_cache(ocr_returns, monkeypatch):
    """The agent is a reusable singleton - one image must not leak into the next."""
    seen = []
    monkeypatch.setattr(ocr, "is_available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_image", lambda image_bytes: seen.append(image_bytes) or image_bytes.decode()
    )

    agent = _agent(FakeLLMService([_read(), _finish("one"), _read(), _finish("two")]))
    agent.run("q", b"FIRST", PNG)
    second = agent.run("q", b"SECOND", PNG)

    assert seen == [b"FIRST", b"SECOND"]
    assert "SECOND" in second.steps[0].observation


def test_a_failing_vision_call_becomes_an_observation(ocr_returns):
    class ExplodingVision(FakeVisionService):
        def analyze_image(self, image_bytes, mime_type, question, temperature=None):
            raise RuntimeError("Vision request failed: upstream 500")

    llm = FakeLLMService([_inspect("what is this?"), _finish("could not see it")])
    result = _agent(llm, ExplodingVision()).run("q", b"img", PNG)

    assert "The inspect_image tool failed" in result.steps[0].observation
    assert result.answer == "could not see it"


# ---- ocr.ocr_image guards --------------------------------------------------


def test_ocr_image_returns_empty_for_empty_bytes():
    assert ocr.ocr_image(b"") == ""


def test_ocr_image_returns_empty_when_tesseract_is_missing(monkeypatch):
    monkeypatch.setattr(ocr, "is_available", lambda: False)
    assert ocr.ocr_image(b"not really an image") == ""


def test_ocr_image_returns_empty_for_undecodable_bytes(monkeypatch):
    """An unreadable upload costs its text, not the request."""
    monkeypatch.setattr(ocr, "is_available", lambda: True)
    assert ocr.ocr_image(b"definitely not a PNG") == ""


# ---- Sampling (stage 3: LLM parameters) ------------------------------------


def test_the_loop_and_the_vision_call_both_run_deterministically(ocr_returns):
    """
    Every reply the loop makes is parsed against a grammar, so sampling noise
    surfaces as a parse failure that looks like a prompt problem.
    """
    ocr_returns("TOTAL 84.50")
    vision = FakeVisionService()
    llm = FakeLLMService([_inspect("what is this?"), _read(), _finish("done")])

    _agent(llm, vision).run("q", b"img", PNG)

    assert llm.temperatures == [0.0, 0.0, 0.0]
    assert vision.temperatures == [0.0]


def test_the_temperature_is_configurable_per_agent(ocr_returns):
    ocr_returns("TEXT")
    vision = FakeVisionService()
    llm = FakeLLMService([_read(), _finish("done")])

    _agent(llm, vision, temperature=0.4).run("q", b"img", PNG)

    assert llm.temperatures == [0.4, 0.4]


# ---- Numeric traceability (SC-10) ------------------------------------------


def test_figures_backed_by_ocr_are_not_flagged(ocr_returns):
    ocr_returns("TOTAL      84.50\nDATE  2026-03-11")
    llm = FakeLLMService([_read(), _finish("The total is 84.50 on 2026-03-11.")])

    result = _agent(llm).run("what is the total?", b"img", PNG)

    assert result.unverified_values == []


def test_a_figure_the_model_invented_is_reported(ocr_returns):
    """The check the agent's whole claim over /vision/analyze rests on."""
    ocr_returns("TOTAL      84.50")
    llm = FakeLLMService([_read(), _finish("The total is 92.00.")])

    result = _agent(llm).run("what is the total?", b"img", PNG)

    assert result.unverified_values == ["92.00"]


def test_figures_are_all_unverified_when_ocr_never_ran(ocr_returns):
    """An answer full of numbers from an image nobody ran recognition over."""
    vision = FakeVisionService("It looks like a receipt for about 84.50.")
    llm = FakeLLMService([_inspect("what is this?"), _finish("The total is 84.50.")])

    result = _agent(llm, vision).run("what is the total?", b"img", PNG)

    assert result.unverified_values == ["84.50"]
