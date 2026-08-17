"""
Check that the numbers an answer states were actually read off the image.

WHY THIS EXISTS

agents/vision_agent.py's central promise - the one thing it offers over a
single call to POST /vision/analyze - is this line from its system prompt:

    "NEVER quote a number, date, reference code or exact spelling that you
     have only seen through inspect_image."

Until this module, that was a request. Prompts are requests; a model that
ignores one produces an answer indistinguishable from a model that
followed it. Meanwhile the OCR grid was sitting in memory the whole time,
which is exactly what the claim can be checked against.

WHY IT REPORTS RATHER THAN ENFORCES

The obvious design - strip or refuse any number that isn't in the OCR
output - is wrong, and the worked example that motivated this agent shows
why:

    "£84.50 across 2 covers is £42.25 a head, inside the £50 cap [E1]."

`84.50` is on the receipt. `42.25` is not - it is arithmetic the agent did
correctly, and `50` came from a retrieved policy passage rather than from
the image. An enforcing validator would flag two correct values and the
right answer would be rejected.

So this returns a *signal*, carried out to the caller as
`unverified_values`. It answers "which numbers in this answer did
character recognition not confirm?", which is a genuinely useful thing to
show a reader, and it makes SC-10 measurable across an evaluation set.
What it cannot do is decide whether an unverified value is a legitimate
derivation or an invention - that needs the reasoning behind it, which is
in the trace.

MATCHING IS FORGIVING ON FORM, STRICT ON DIGITS

Both sides are reduced to digit sequences, so `£1,234.50` matches an OCR
grid containing `1234.50` and formatting differences never raise a false
alarm. It also means `84.50` matches a grid that recognised `8450` - real
Tesseract behaviour on low-resolution input, where the digits were read
correctly and only the decimal point was lost.

Comparison is against the grid's *individual* number tokens, not against
one concatenation of every digit on the page. That distinction is
load-bearing. Concatenated, a policy limit of `£50` quoted from a
retrieved passage matches the `50` sitting inside the receipt's own
`84.50`, and the check silently blesses a figure that never came from the
image at all. Per-token, it is correctly reported.

A prefix counts as a match, so an answer saying "in 2026" is confirmed by
a grid containing `2026-03-11`. A suffix does not, which is what keeps
`50` from matching `8450`. Tokens with fewer than two digits are skipped
entirely - "2 covers" and "step 3" are noise, not claims about the image.
"""
import re

# A number-shaped token: an optional currency mark, then digits, optionally
# joined by the separators that appear inside dates, times, amounts and
# reference codes. Matches "84.50", "£1,234.56", "2026-03-11", "12:30", "42".
_VALUE = re.compile(r"[£$€]?\d+(?:[.,:/\-]\d+)*")

# Citation labels are stripped before scanning. "[E1]" is a pointer to a
# retrieved passage, not a value read off the image, and flagging it would
# make every grounded answer look like it invented something.
_CITATION = re.compile(r"\[E\d+\]")

# Below this many digits a token carries no real claim and matches
# coincidentally too often to be worth reporting.
_MIN_DIGITS = 2


def _digits(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def unverified_values(answer: str, ocr_text: str) -> list[str]:
    """
    Return the number-shaped tokens in `answer` that OCR did not confirm.

    In first-appearance order, deduplicated. An empty list means every
    number in the answer traces back to recognised characters.

    `ocr_text` empty - because recognition was unavailable, found nothing,
    or was never run - is not a special case: every numeric token is then
    unconfirmed, which is precisely the situation worth reporting.
    """
    if not answer:
        return []

    recognised = [
        _digits(match.group()) for match in _VALUE.finditer(ocr_text or "")
    ]

    flagged: list[str] = []
    seen: set[str] = set()

    for match in _VALUE.finditer(_CITATION.sub(" ", answer)):
        token = match.group()
        token_digits = _digits(token)

        if len(token_digits) < _MIN_DIGITS:
            continue
        if any(candidate.startswith(token_digits) for candidate in recognised):
            continue
        if token in seen:
            continue

        seen.add(token)
        flagged.append(token)

    return flagged
