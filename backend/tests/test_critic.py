"""
Tests for the reviewer (agents/critic.py).

Two things carry the risk here, and neither is the happy path:

- Parsing. The critic is asked for one line of JSON and, like every other
  model output in this project, will sometimes produce something else. The
  verdict has to survive that.
- Failing open. Every unreadable, empty or failed review must approve, and
  must say it did not review. A reviewer that blocks answers when it is
  broken is worse than no reviewer at all.
"""
from agents.critic import Critic, _parse_verdict
from tests.test_research_agent import FakeLLMService

FINDINGS = [("the document specialist", "Refunds are accepted within 30 days.")]


def review(reply: str, findings=None):
    llm = FakeLLMService([reply])
    critic = Critic(llm)
    return critic.review("refund window?", findings or FINDINGS, "30 days [E1].")


# ---- Parsing ---------------------------------------------------------------


def test_a_plain_approval_is_read_as_approved_and_reviewed():
    verdict = review('{"verdict": "approve"}')

    assert verdict.approved is True
    assert verdict.reviewed is True
    assert verdict.problem == ""


def test_a_revision_carries_the_problem_through():
    verdict = review('{"verdict": "revise", "problem": "the 14-day figure is invented"}')

    assert verdict.approved is False
    assert verdict.reviewed is True
    assert verdict.problem == "the 14-day figure is invented"


def test_a_fenced_verdict_is_read():
    """Told not to, models fence JSON anyway."""
    verdict = review('```json\n{"verdict": "approve"}\n```')

    assert verdict.approved is True
    assert verdict.reviewed is True


def test_prose_around_the_verdict_does_not_break_it():
    verdict = review(
        'Having checked the draft against the evidence:\n'
        '{"verdict": "revise", "problem": "no support for the 14-day claim"}\n'
        "I hope that helps."
    )

    assert verdict.approved is False
    assert verdict.problem == "no support for the 14-day claim"


def test_adjacent_vocabulary_is_accepted_for_both_verdicts():
    """
    Models trained on other review harnesses reach for those harnesses'
    words. Treating a near-miss as unparseable would silently approve a
    draft the critic meant to reject.
    """
    assert review('{"verdict": "approved"}').approved is True
    assert review('{"verdict": "PASS"}').approved is True

    rejected = review('{"verdict": "reject", "problem": "unsupported"}')
    assert rejected.approved is False
    assert rejected.reviewed is True


def test_reason_is_accepted_as_an_alias_for_problem():
    verdict = review('{"verdict": "revise", "reason": "the total is not in evidence"}')

    assert verdict.approved is False
    assert verdict.problem == "the total is not in evidence"


# ---- Failing open ----------------------------------------------------------


def test_a_reply_with_no_json_approves_unreviewed():
    verdict = review("Looks good to me.")

    assert verdict.approved is True
    assert verdict.reviewed is False


def test_malformed_json_approves_unreviewed():
    verdict = review('{"verdict": "approve",}')

    assert verdict.approved is True
    assert verdict.reviewed is False


def test_an_unrecognised_verdict_word_approves_unreviewed():
    """
    Bouncing a draft on a verdict nobody can interpret is the worst of both
    outcomes - a step spent, and no idea what to change.
    """
    verdict = review('{"verdict": "maybe", "problem": "not sure"}')

    assert verdict.approved is True
    assert verdict.reviewed is False


def test_a_rejection_with_no_problem_approves_unreviewed():
    """
    Not actionable. The supervisor would be told to fix something without
    being told what, and would most likely resubmit the same draft.
    """
    verdict = review('{"verdict": "revise"}')

    assert verdict.approved is True
    assert verdict.reviewed is False


def test_a_provider_failure_approves_unreviewed():
    class FailingLLM:
        def generate_response(self, *args, **kwargs):
            raise RuntimeError("LLM request failed: upstream timeout")

    verdict = Critic(FailingLLM()).review("q", FINDINGS, "draft")

    assert verdict.approved is True
    assert verdict.reviewed is False


def test_a_bug_in_the_critic_path_is_not_swallowed():
    """
    Fail-open covers provider failures, which LLMService wraps into
    RuntimeError. It must not also hide a programming error - catching
    broadly here once made a critic that never ran look like one that
    approved.
    """
    class BuggyLLM:
        def generate_response(self, *args, **kwargs):
            raise TypeError("generate_response() got an unexpected keyword")

    try:
        Critic(BuggyLLM()).review("q", FINDINGS, "draft")
    except TypeError:
        return
    raise AssertionError("a TypeError from the critic path should propagate")


# ---- Not running at all ----------------------------------------------------


def test_no_findings_means_no_review_and_no_call():
    """
    "Is this supported by the evidence?" is meaningless with no evidence, and
    skipping it keeps a directly-answered question at its original latency.
    """
    llm = FakeLLMService([])  # would raise if called
    verdict = Critic(llm).review("what is the capital of France?", [], "Paris.")

    assert verdict.approved is True
    assert verdict.reviewed is False
    assert llm.prompts == []


# ---- The prompt it builds --------------------------------------------------


def test_the_draft_and_the_evidence_both_reach_the_prompt():
    llm = FakeLLMService(['{"verdict": "approve"}'])
    Critic(llm).review("refund window?", FINDINGS, "You have 30 days [E1].")

    prompt = llm.prompts[0]
    assert "refund window?" in prompt
    assert "Refunds are accepted within 30 days." in prompt
    assert "You have 30 days [E1]." in prompt
    # The draft goes last: models attend most reliably to the end of a
    # prompt, and putting the evidence after it produces a review of the
    # evidence instead.
    assert prompt.index("You have 30 days [E1].") > prompt.index(
        "Refunds are accepted within 30 days."
    )


def test_the_review_is_deterministic():
    llm = FakeLLMService(['{"verdict": "approve"}'])
    Critic(llm, temperature=0.0).review("q", FINDINGS, "draft")

    assert llm.temperatures == [0.0]


# ---- The parser, directly --------------------------------------------------


def test_parse_verdict_handles_a_non_object_payload():
    assert _parse_verdict('["approve"]').reviewed is False


def test_parse_verdict_handles_an_empty_reply():
    assert _parse_verdict("").reviewed is False
