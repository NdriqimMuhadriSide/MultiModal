"""
Tests for the evaluation graders (evals/grading.py).

The runner itself cannot be tested without spending provider quota, so the
part that decides pass or fail is tested here instead. A grader nobody has
checked is worse than no grader: it produces numbers that look like
measurements.
"""
from dataclasses import dataclass

from evals.grading import (
    is_refusal,
    states_value,
    tool_choice_ok,
    tools_used,
    value_accuracy,
)


@dataclass
class FakeStep:
    tool: str


# ---- states_value ----------------------------------------------------------


def test_a_stated_value_is_recognised_through_formatting():
    assert states_value("The total is £84.50.", "84.50")
    assert states_value("It comes to 1,234.50 in all.", "1234.50")


def test_a_value_is_not_satisfied_by_digits_inside_another_number():
    """The same trap value_check.py had to be fixed for."""
    assert not states_value("The total is 84.50.", "50")


def test_a_missing_value_is_not_recognised():
    assert not states_value("The total is 92.00.", "84.50")


def test_non_numeric_expectations_match_on_text():
    assert states_value("This is a restaurant receipt.", "receipt")
    assert not states_value("This is an invoice.", "receipt")


def test_value_accuracy_counts_hits_over_expectations():
    answer = "Total 84.50, issued 2026-03-11."
    assert value_accuracy(answer, {"total": "84.50", "date": "2026-03-11"}) == (2, 2)
    assert value_accuracy(answer, {"total": "84.50", "vat": "14.08"}) == (1, 2)


def test_value_accuracy_of_no_expectations_is_zero_of_zero():
    assert value_accuracy("anything", {}) == (0, 0)


# ---- is_refusal ------------------------------------------------------------


def test_a_plain_decline_is_a_refusal():
    assert is_refusal("This image contains no printed text, so I cannot read a total.")


def test_a_hedged_guess_is_not_a_refusal():
    """The failure the numeric half exists to catch."""
    assert not is_refusal(
        "I couldn't read it clearly, but the total appears to be 84.50."
    )


def test_a_confident_answer_is_not_a_refusal():
    assert not is_refusal("The total is 84.50.")


def test_a_decline_mentioning_a_single_digit_is_still_a_refusal():
    """"no. 1 priority" style noise must not disqualify a genuine decline."""
    assert is_refusal("There is no text here at all - it is a photograph.")


# ---- tool choice -----------------------------------------------------------


def test_tools_used_lists_names_in_order_ignoring_parse_failures():
    steps = [FakeStep("inspect_image"), FakeStep(""), FakeStep("read_text")]
    assert tools_used(steps) == ["inspect_image", "read_text"]


def test_a_required_tool_that_did_not_run_fails():
    ok, reason = tool_choice_ok([FakeStep("inspect_image")], {"read_text"}, set())
    assert not ok
    assert "did not use read_text" in reason


def test_a_forbidden_tool_that_ran_fails():
    ok, reason = tool_choice_ok(
        [FakeStep("inspect_image"), FakeStep("search_knowledge_base")],
        set(),
        {"search_knowledge_base"},
    )
    assert not ok
    assert "should not have used search_knowledge_base" in reason


def test_the_right_route_passes():
    ok, reason = tool_choice_ok(
        [FakeStep("read_text"), FakeStep("finish")],
        {"read_text"},
        {"search_knowledge_base"},
    )
    assert ok
    assert reason == "ok"
