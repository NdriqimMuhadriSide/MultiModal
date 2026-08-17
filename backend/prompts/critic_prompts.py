"""
Who the critic is, and what it is allowed to object to.

One prompt, not two. Unlike the loop agents (see
prompts/supervisor_prompts.py) the critic never takes an action, so there is
no Thought/Action contract to state and no synthesis mode to contradict it.

WHAT IT MAY OBJECT TO, AND WHY THE LIST IS SHORT

A critic that can object to anything objects to everything. Each rejection
costs a step and a full extra turn of the supervisor's loop, so the bar is
not "could this be better" - it is "would a reader be misled". The four
faults below are the ones that make an answer actively wrong rather than
merely improvable:

    unsupported     a claim the evidence does not contain. The failure mode
                    the whole retrieval stack exists to prevent, arriving at
                    the last possible moment.
    miscited        an [E#] label that was never shown, or one attached to a
                    claim it does not support. Worse than no citation: it
                    invites a reader to stop checking.
    contradicted    the answer says the opposite of what a specialist
                    reported.
    silent gap      part of the question was never researched and the answer
                    does not say so. A half-answer that admits its gap is
                    useful; one that hides it is not.

And the explicit non-list matters as much. Style, length, tone, hedging and
formatting are all off limits, because an answer sent back for rewording
costs exactly as much as one sent back for inventing a number.
"""

CRITIC_SYSTEM_PROMPT = (
    "You are a reviewer. You are shown a question, the evidence that was "
    "gathered to answer it, and a draft answer. You decide whether the draft "
    "is supported by the evidence.\n"
    "\n"
    "You are not the author. You do not rewrite the answer, and you do not "
    "answer the question yourself.\n"
    "\n"
    "Reply with a single JSON object and nothing else:\n"
    '  {"verdict": "approve"}\n'
    "or\n"
    '  {"verdict": "revise", "problem": "<what is wrong and what to do>"}\n'
    "\n"
    "Ask for a revision ONLY when one of these is true:\n"
    "- The draft states something the evidence does not contain.\n"
    "- The draft cites an [E1]-style label that does not appear in the "
    "evidence, or attaches one to a claim it does not support.\n"
    "- The draft contradicts the evidence.\n"
    "- The draft answers part of the question that was never investigated, "
    "without saying that part is unverified.\n"
    "\n"
    "Approve everything else. In particular, do NOT ask for a revision "
    "because of:\n"
    "- wording, tone, length, formatting or structure\n"
    "- an answer being shorter or plainer than you would have written\n"
    "- a correct statement that the evidence does not happen to mention but "
    "that is obviously general knowledge\n"
    "- arithmetic the draft performed correctly on numbers that ARE in the "
    "evidence\n"
    "- the answer saying the evidence does not cover something, when it "
    "does not\n"
    "\n"
    "When you do ask for a revision, `problem` is read by the author as its "
    "next instruction. Say which claim is unsupported and what to do about "
    "it - drop it, hedge it, or state that it was not verified. Be specific: "
    '"the 30-day figure is not in the evidence" is useful, "some claims are '
    'unsupported" is not.'
)


def format_critic_prompt(question: str, evidence: str, draft: str) -> str:
    """
    Build the critic's single call.

    The draft goes last. Models attend most reliably to the end of a prompt,
    and the draft is the thing being judged - putting the evidence after it
    reliably produces a review of the evidence instead.
    """
    return (
        f"The question that was asked:\n{question}\n"
        "\n"
        "The evidence that was gathered:\n"
        f"{evidence}\n"
        "\n"
        "The draft answer to review:\n"
        f"{draft}\n"
        "\n"
        "Reply with your JSON verdict now."
    )


def format_evidence(findings: list[tuple[str, str]]) -> str:
    """
    Render what the specialists reported, as the critic reads it.

    Only what came *back* - not the thoughts, tool names or action JSON that
    make up the rest of the scratchpad. The critic is judging whether the
    answer follows from the findings, and how the supervisor decided to go
    looking for them is not evidence about anything.
    """
    if not findings:
        return "(nothing was gathered - no specialist was consulted)"

    return "\n\n".join(
        f"From {source}:\n{report}" for source, report in findings
    )
