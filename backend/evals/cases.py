"""
The labelled cases the vision agent is evaluated against.

Each case pairs an image whose contents are known by construction (see
tests/image_builder.py) with a question, the facts a correct answer must
state, and the tools the agent should and should not have used.

The tool expectations are the unusual part and the most important. Accuracy
alone cannot distinguish an agent that *decided* how to read an image from
one that runs every tool on everything - both score the same. Since the
decision is this agent's whole reason to exist over a single vision call,
it gets measured directly.

Kept deliberately small. Every case is a live run costing several provider
calls, and a set nobody runs because it takes twenty minutes measures
nothing.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

from tests.image_builder import (
    BuiltImage,
    bar_chart,
    blank_page,
    clean_receipt,
    low_res_receipt,
    photograph,
    rotated_receipt,
)


@dataclass
class EvalCase:
    id: str
    question: str
    build: Callable[[], BuiltImage]
    # Facts the answer must state, keyed for readable failures. Empty when
    # the correct outcome is a refusal.
    expect: dict[str, str] = field(default_factory=dict)
    # Tools the run must have invoked.
    require_tools: set[str] = field(default_factory=set)
    # Tools it must NOT have invoked - what makes this a decision rather
    # than a pipeline.
    forbid_tools: set[str] = field(default_factory=set)
    # True when declining is the right answer.
    expect_refusal: bool = False
    why: str = ""


def _receipt_truth(keys: tuple[str, ...]) -> dict[str, str]:
    truth = clean_receipt().truth
    return {key: truth[key] for key in keys}


CASES: list[EvalCase] = [
    EvalCase(
        id="clean-total",
        question="What is the total on this receipt, and what date was it issued?",
        build=clean_receipt,
        expect=_receipt_truth(("total", "date")),
        require_tools={"read_text"},
        why=(
            "The core case. Both facts are exact character strings, so the "
            "run must go through recognition rather than quoting the vision "
            "model's reading of them."
        ),
    ),
    EvalCase(
        id="low-res-total",
        question="What is the total on this receipt?",
        build=low_res_receipt,
        expect=_receipt_truth(("total",)),
        require_tools={"read_text"},
        why=(
            "The degradation behind constraint C-8: at this size recognition "
            "drops the decimal point and reads 8450. Scoring this case is "
            "how you find out whether the agent notices and says so, or "
            "reports 8450 as a total."
        ),
    ),
    EvalCase(
        id="rotated-total",
        question="What is the total on this receipt?",
        build=rotated_receipt,
        expect=_receipt_truth(("total",)),
        require_tools={"read_text"},
        why="The commonest real-world degradation: photographed at an angle.",
    ),
    EvalCase(
        id="photo-no-total",
        question="What is the total on this receipt?",
        build=photograph,
        expect_refusal=True,
        why=(
            "The most valuable case in the set. There is no receipt and no "
            "total, so any number in the answer is invented. This is where a "
            "confident wrong answer costs the most."
        ),
    ),
    EvalCase(
        id="blank-page",
        question="What does this document say?",
        build=blank_page,
        expect_refusal=True,
        why="The degenerate input. Both readers return nothing; the agent must survive it.",
    ),
    EvalCase(
        id="chart-highest",
        question="Which quarter had the highest revenue in this chart?",
        build=bar_chart,
        expect={"highest": "Q3"},
        require_tools={"inspect_image"},
        why=(
            "Bar heights are a visual judgement that recognition cannot "
            "make. The agent has to reach for the vision model here."
        ),
    ),
    EvalCase(
        id="document-type",
        question="What kind of document is this? Do not quote any figures.",
        build=clean_receipt,
        expect={"kind": "receipt"},
        require_tools={"inspect_image"},
        forbid_tools={"search_knowledge_base"},
        why=(
            "Knowing when NOT to reach for a tool is half the decision. This "
            "question is about the image alone, so a corpus search is wasted "
            "latency and a wasted step."
        ),
    ),
]
