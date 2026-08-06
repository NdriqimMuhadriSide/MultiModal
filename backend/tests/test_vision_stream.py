import json

from processors.streaming.vision_stream import StreamObservation, VisionStreamAnalyzer


class FakeVisionService:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_call: dict | None = None

    def analyze_image(self, image_bytes, mime_type, question, system_prompt):
        self.last_call = {
            "image_bytes": image_bytes,
            "mime_type": mime_type,
            "question": question,
            "system_prompt": system_prompt,
        }
        return self._response


def _valid_json_response(**overrides) -> str:
    payload = {
        "objects": ["laptop", "coffee mug"],
        "text_detected": [],
        "ui_elements": [],
        "activities": ["person typing"],
        "warnings": [],
        "important_changes": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_analyze_frame_parses_structured_json_response():
    fake_service = FakeVisionService(_valid_json_response())
    analyzer = VisionStreamAnalyzer(vision_service=fake_service)

    observation = analyzer.analyze_frame(
        frame_bytes=b"fake frame",
        mime_type="image/jpeg",
        recent_context="",
        question="What's on the desk?",
    )

    assert observation.objects == ["laptop", "coffee mug"]
    assert observation.activities == ["person typing"]
    assert observation.warnings == []


def test_analyze_frame_passes_streaming_system_prompt_not_default_vision_prompt():
    from prompts.streaming_prompts import STREAM_OBSERVATION_SYSTEM_PROMPT

    fake_service = FakeVisionService(_valid_json_response())
    analyzer = VisionStreamAnalyzer(vision_service=fake_service)

    analyzer.analyze_frame(
        frame_bytes=b"fake frame", mime_type="image/jpeg", recent_context="", question=None
    )

    assert fake_service.last_call["system_prompt"] == STREAM_OBSERVATION_SYSTEM_PROMPT


def test_analyze_frame_embeds_recent_context_and_question_in_prompt():
    fake_service = FakeVisionService(_valid_json_response())
    analyzer = VisionStreamAnalyzer(vision_service=fake_service)

    analyzer.analyze_frame(
        frame_bytes=b"fake frame",
        mime_type="image/jpeg",
        recent_context="Frame 1: person enters room.",
        question="What changed?",
    )

    prompt = fake_service.last_call["question"]
    assert "Frame 1: person enters room." in prompt
    assert "What changed?" in prompt


def test_analyze_frame_falls_back_to_raw_text_for_invalid_json():
    fake_service = FakeVisionService("This is not JSON at all.")
    analyzer = VisionStreamAnalyzer(vision_service=fake_service)

    observation = analyzer.analyze_frame(
        frame_bytes=b"fake", mime_type="image/jpeg", recent_context="", question=None
    )

    assert observation.objects == []
    assert observation.raw_text == "This is not JSON at all."


def test_analyze_frame_strips_markdown_code_fence():
    fenced_response = "```json\n" + _valid_json_response() + "\n```"
    fake_service = FakeVisionService(fenced_response)
    analyzer = VisionStreamAnalyzer(vision_service=fake_service)

    observation = analyzer.analyze_frame(
        frame_bytes=b"fake", mime_type="image/jpeg", recent_context="", question=None
    )

    assert observation.objects == ["laptop", "coffee mug"]


def test_analyze_frame_tolerates_missing_fields_in_json():
    fake_service = FakeVisionService(json.dumps({"objects": ["chair"]}))
    analyzer = VisionStreamAnalyzer(vision_service=fake_service)

    observation = analyzer.analyze_frame(
        frame_bytes=b"fake", mime_type="image/jpeg", recent_context="", question=None
    )

    assert observation.objects == ["chair"]
    assert observation.warnings == []
    assert observation.activities == []


def test_as_flat_list_combines_all_fields_in_order():
    observation = StreamObservation(
        objects=["laptop"],
        text_detected=["Error 404"],
        ui_elements=["dialog box"],
        activities=["typing"],
        warnings=["low battery"],
        important_changes=["new window opened"],
    )

    assert observation.as_flat_list() == [
        "laptop",
        "Error 404",
        "dialog box",
        "typing",
        "low battery",
        "new window opened",
    ]


def test_summary_line_falls_back_to_raw_text_when_no_fields_present():
    observation = StreamObservation(raw_text="nothing notable")

    assert observation.summary_line() == "nothing notable"
