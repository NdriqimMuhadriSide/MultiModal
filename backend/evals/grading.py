"""
Graders for the vision agent evaluation.

Pure functions over an answer string and a trace, deliberately - so the
scoring can be unit-tested without spending a single provider call. A
grader nobody has checked is worse than no grader: it produces numbers that
look like measurements.

Every grader here is a heuristic, and each one says where it is weak. That
matters more than usual, because the temptation with an eval harness is to
read its output as truth rather than as an approximation of a judgement a
person would make.
"""
import re

# A value is "stated" if its digits appear in the answer, ignoring
# formatting. The same normalisation agents/value_check.py uses, and for the
# same reason: 84.50, £84.50 and 84,50 are the same claim, and an evaluation
# that marked two of those wrong would be measuring punctuation.
_VALUE = re.compile(r"\d+(?:[.,:/\-]\d+)*")

# Patterns a model reaches for when it declines.
#
# Regexes rather than substrings after the first live run marked a correct
# refusal wrong: the reply said "no visible text", and a literal "no text"
# does not appear in that. Any qualifier can sit between the negation and
# the noun, so the qualifier has to be part of the pattern.
_HEDGES = re.compile(
    r"""
    no\s+\w*\s*(text|content|writing|information|data|receipt|total|figures)
  | (does\s?not|doesn't|is\s?not|isn't|cannot|can't|could\s?not|couldn't)
      \s+(contain|show|say|include|read|appear|a\s)
  | (unable|not\s+able)\s+to
  | (un|not\s+)?(readable|legible)
  | blank
  | empty
  | nothing\s+(at\s+all|to\s+read|visible|here)
  | photograph|photo\b|illustration|abstract
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _digits(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def states_value(answer: str, expected: str) -> bool:
    """
    True if `answer` states `expected`, ignoring formatting differences.

    Matches per token rather than against the answer's concatenated digits,
    so "50" is not satisfied by an answer that only mentions "84.50" - the
    same trap agents/value_check.py had to be fixed for.
    """
    target = _digits(expected)
    if not target:
        # A non-numeric expectation (a vendor name, a quarter label).
        return expected.lower() in answer.lower()

    return any(
        _digits(match.group()) == target for match in _VALUE.finditer(answer)
    )


def value_accuracy(answer: str, expected: dict[str, str]) -> tuple[int, int]:
    """Return `(stated, total)` over the expected facts."""
    if not expected:
        return 0, 0
    hit = sum(1 for value in expected.values() if states_value(answer, value))
    return hit, len(expected)


def is_refusal(answer: str) -> bool:
    """
    True if the answer declines rather than inventing something.

    Two conditions, both required: a hedge phrase is present, and the answer
    makes no numeric claim. The second is the one doing the real work - a
    reply that says "I couldn't read it clearly, but the total is 84.50" is
    not a refusal, it is a guess with a disclaimer, and only the numeric
    check catches that.

    Weakness worth knowing: a refusal phrased outside the pattern list
    scores as a non-refusal. The list is easier to extend than a cleverer
    grader is to trust - and the first live run proved the point by marking
    a perfectly good "no visible text" reply as a failure.
    """
    hedged = bool(_HEDGES.search(answer))
    claims_number = any(
        len(_digits(match.group())) >= 2 for match in _VALUE.finditer(answer)
    )
    return hedged and not claims_number


def tools_used(steps) -> list[str]:
    """The tool names a run invoked, in order, excluding parse failures."""
    return [step.tool for step in steps if step.tool]


def tool_choice_ok(
    steps, required: set[str], forbidden: set[str]
) -> tuple[bool, str]:
    """
    Did the agent pick the right readers?

    Returns `(ok, reason)` - the reason so a failing case says what went
    wrong in the report rather than only that it failed.

    This is the criterion that measures the agent's actual reason to exist.
    Getting the answer right by running every tool on every image would
    score well on accuracy and badly here, which is the point: a fixed
    pipeline can do the former and only a decision can do the latter.
    """
    used = set(tools_used(steps))

    missing = required - used
    if missing:
        return False, f"did not use {', '.join(sorted(missing))}"

    used_forbidden = forbidden & used
    if used_forbidden:
        return False, f"should not have used {', '.join(sorted(used_forbidden))}"

    return True, "ok"
