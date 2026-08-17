"""
The critic — reviews a draft answer against the evidence it was built from.

WHERE IT SITS

Not a tool the supervisor may choose to call. A critic that is optional gets
skipped, and gets skipped hardest on exactly the answers that most need it -
the confident ones. It is a gate on `finish`: the supervisor produces a draft,
the critic sees it before the user does, and an objection is fed back as the
next observation.

That last part needs no new machinery. `ActionRejected` (agents/agent_loop.py)
already means "the tool declined and here is what to do instead", and the loop
already turns it into an observation costing one step. A critic's objection is
precisely that, so the gate is ten lines rather than a new control flow.

WHY IT IS ONE CALL AND NOT A LOOP

Every other agent here is a ReAct loop, and this one deliberately is not.
A loop earns its cost when the next action depends on the last observation -
when there is something to *do*. The critic has one question ("does this
answer follow from this evidence?") and everything needed to answer it is
already in the prompt. Giving it tools would let it re-search the corpus,
which is a different job (fact-checking against the source) at several times
the price, and one the supervisor's own specialists have already done.

So: a judge, in the LLM-as-judge sense. It is a distinct role in the system
rather than a distinct kind of machinery.

WHY IT FAILS OPEN

An unparseable verdict, an empty reply, a provider error - all approve. A
reviewer that blocks answers when it is broken is worse than no reviewer:
the supervisor's work is already done and correct-or-not, and refusing to
deliver it because a *second* model could not produce JSON turns a
degradation into an outage. The run log records that review did not happen,
which is where that belongs.

WHEN IT DOES NOT RUN AT ALL

When nothing was gathered. "Is this answer supported by the evidence?" has no
meaning for a question answered from the model's own knowledge, and the
critic would be reviewing a draft against an empty page. Skipping it there
also keeps the common case - a direct answer to a conversational question -
at exactly the latency it had before this module existed.
"""
import json
import logging
from dataclasses import dataclass

from agents.agent_loop import _extract_json_object
from agents.run_log import record_run
from prompts.critic_prompts import (
    CRITIC_SYSTEM_PROMPT,
    format_critic_prompt,
    format_evidence,
)

logger = logging.getLogger(__name__)

# Recognised spellings of the two verdicts. Models reach for adjacent
# vocabulary - "approved", "pass", "ok" - and treating a near-miss as a
# parse failure would silently approve a draft the critic meant to reject.
_APPROVE = {"approve", "approved", "ok", "pass", "accept", "accepted", "yes"}
_REVISE = {"revise", "reject", "rejected", "fail", "revision", "no"}


@dataclass
class Verdict:
    """
    What the critic decided.

    `problem` is written for the supervisor, not for a developer - it becomes
    the next observation in the loop, so it has to say what to do rather than
    only what is wrong. Empty on approval.
    """

    approved: bool
    problem: str = ""
    # False when the critic did not actually produce a usable verdict and
    # this is the fail-open default. Distinct from `approved` because "was
    # checked and passed" and "could not be checked" are different facts, and
    # only one of them is worth telling a reader.
    reviewed: bool = True


APPROVED = Verdict(approved=True)
NOT_REVIEWED = Verdict(approved=True, reviewed=False)


class Critic:
    """
    Reviews a draft against the findings it was built from.

    Stateless: every `review` is one call with everything it needs. The
    supervisor owns the "how many times may it push back" policy, because
    that is a property of a run rather than of the reviewer.
    """

    def __init__(self, llm_service, temperature: float | None = 0.0) -> None:
        self._llm_service = llm_service
        self._temperature = temperature

    def review(
        self, question: str, findings: list[tuple[str, str]], draft: str
    ) -> Verdict:
        """
        Judge `draft` against `findings`. Never raises.

        Returns NOT_REVIEWED when there was nothing to check against, or when
        the critic failed - see this module's docstring for why both approve.
        """
        if not findings:
            return NOT_REVIEWED

        with record_run("critic") as run:
            try:
                raw = self._llm_service.generate_response(
                    format_critic_prompt(question, format_evidence(findings), draft),
                    system_prompt=CRITIC_SYSTEM_PROMPT,
                    temperature=self._temperature,
                )
            except RuntimeError:
                # RuntimeError specifically, not Exception. LLMService already
                # wraps every provider failure into one, so this catches the
                # whole class of things fail-open exists for - a timeout, a
                # rate limit, a malformed response - while leaving a bug in
                # this module to surface as a bug. Catching broadly here once
                # hid a test's own assertion and made a critic that never ran
                # look like one that approved.
                logger.warning("Critic call failed; approving unreviewed.", exc_info=True)
                run.stopped_because = "error"
                return NOT_REVIEWED

            verdict = _parse_verdict(raw)
            run.record_tool("approve" if verdict.approved else "revise")
            run.stopped_because = "finished" if verdict.reviewed else "unparseable"
            return verdict


def _parse_verdict(raw: str) -> Verdict:
    """
    Read the critic's reply, tolerating the ways a model deviates from JSON.

    Reuses agents/agent_loop.py's brace scanner rather than a regex, for the
    reasons documented there: it survives prose after the object and nesting
    inside it, both of which a reviewer asked for one line of JSON still
    produces.

    Anything it cannot read approves, and says it was not reviewed.
    """
    candidate = _extract_json_object(raw or "")
    if candidate is None:
        return NOT_REVIEWED

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return NOT_REVIEWED

    if not isinstance(payload, dict):
        return NOT_REVIEWED

    verdict = payload.get("verdict")
    if not isinstance(verdict, str):
        return NOT_REVIEWED

    normalised = verdict.strip().lower()
    if normalised in _APPROVE:
        return APPROVED

    if normalised not in _REVISE:
        # A word for neither. Approving is the safe reading: the alternative
        # is bouncing a draft on a verdict nobody can interpret.
        return NOT_REVIEWED

    problem = payload.get("problem") or payload.get("reason") or ""
    if not isinstance(problem, str) or not problem.strip():
        # A rejection with no reason is not actionable - the supervisor would
        # be told to fix something without being told what, and would most
        # likely resubmit the same draft. Treated as unreviewed.
        return NOT_REVIEWED

    return Verdict(approved=False, problem=problem.strip())
