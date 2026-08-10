"""
Sentence segmentation.

Responsibility: cut a passage into sentences, keeping every character, so that
joining the pieces reproduces the input exactly. Pure text in, text out.

Four different parts of the chunking layer need this and all of them need it to
be the same: recursive splitting falls back to sentence boundaries,
semantic chunking measures similarity *between* sentences, sentence-window
retrieval embeds them one at a time, and propositional chunking feeds them to
an LLM. A splitter that disagrees with itself between those would put the same
text in different places depending on the strategy.

Why not just split on ". "
--------------------------
Because a period is not a sentence boundary most of the time:

    "Dr. Smith weighed 3.5 kg of sample no. 4 (see Fig. 2)."

That is one sentence with six periods, and a naive split turns it into seven
fragments - each embedded separately, each meaningless. The rules below are the
short list of exceptions that covers the overwhelming majority of real text:
known abbreviations, single-letter initials, decimals, and ellipses.

This is deliberately not a statistical model. A trained segmenter (spaCy,
NLTK's Punkt) is more accurate on hard cases, but it is a dependency and a
model download to fix a problem that a page of rules gets ~95% right - and the
cost of the remaining 5% is a chunk boundary in a slightly wrong place, not a
wrong answer.
"""
import re

# Abbreviations that end in a period and are almost never sentence-final.
# Case-insensitive, matched as whole words.
_ABBREVIATIONS = {
    # Titles
    "mr", "mrs", "ms", "dr", "prof", "rev", "hon", "st", "sr", "jr",
    # Latin and reference
    "e.g", "i.e", "etc", "vs", "cf", "al", "ca", "approx", "viz",
    "fig", "eq", "ref", "no", "nos", "vol", "ch", "sec", "pp", "p",
    # Business and place
    "inc", "ltd", "co", "corp", "dept", "est", "univ",
    # Months and days, which appear abbreviated in dates
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "tues", "wed", "thu", "thurs", "fri", "sat", "sun",
}

# A candidate boundary: sentence-ending punctuation, optional closing quote or
# bracket, then whitespace. Everything after this is filtered by the rules
# below - this only proposes, it doesn't decide.
_CANDIDATE = re.compile(r'([.!?]+)(["\'’”)\]]*)(\s+)')

# The token immediately before the punctuation.
_PRECEDING_WORD = re.compile(r"([A-Za-z0-9.’']+)$")
# The first character of what follows, once whitespace is skipped.
_FOLLOWING = re.compile(r"^\W*(\w)")


def split_sentences(text: str) -> list[str]:
    """
    Split `text` into sentences.

    Each returned sentence keeps its own trailing punctuation and the
    whitespace that followed it, so `"".join(split_sentences(t)) == t`. That
    matters because callers reassemble these into chunks and rely on the
    reassembly being lossless - a splitter that ate the spaces between
    sentences would quietly reflow the document.
    """
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    for match in _CANDIDATE.finditer(text):
        end = match.end()
        if _is_boundary(text, match):
            sentences.append(text[start:end])
            start = end

    if start < len(text):
        sentences.append(text[start:])
    return [sentence for sentence in sentences if sentence]


def split_sentences_stripped(text: str) -> list[str]:
    """Sentences with surrounding whitespace removed, for callers that only
    want the words (embedding a sentence, showing it, sending it to an LLM)."""
    return [sentence.strip() for sentence in split_sentences(text) if sentence.strip()]


def _is_boundary(text: str, match: re.Match) -> bool:
    """
    Decide whether a candidate really ends a sentence.

    Everything here is a reason to say *no*. The default is that punctuation
    followed by a space ends a sentence, which is right far more often than it
    is wrong; these are the recognisable cases where it isn't.
    """
    punctuation = match.group(1)

    # "!" and "?" are unambiguous - no abbreviation ends in one. Runs of dots
    # ("...") are an ellipsis mid-sentence far more often than a boundary, but
    # a following capital settles it.
    if punctuation != ".":
        if set(punctuation) == {"."}:
            return _starts_a_sentence(text, match.end())
        return True

    preceding = _PRECEDING_WORD.search(text[: match.start()])
    if preceding:
        word = preceding.group(1)
        # "Dr. Smith", "e.g. this", "Fig. 2"
        if word.lower().rstrip(".") in _ABBREVIATIONS:
            return False
        # "J. R. R. Tolkien" - a lone capital letter is an initial.
        if len(word) == 1 and word.isupper():
            return False
        # "3.5" or "1.2.3" - a digit either side of the dot is a number, but
        # "...ends in 2. Next" is a real boundary, so only an *interior* dot
        # counts.
        if word[-1].isdigit() and "." in word[:-1]:
            return False

    return _starts_a_sentence(text, match.end())


def _starts_a_sentence(text: str, position: int) -> bool:
    """
    True unless what follows clearly continues the current sentence.

    A lowercase letter after a period nearly always means the period was doing
    something else - an abbreviation this module doesn't know, or a decimal.
    """
    following = _FOLLOWING.match(text[position:])
    if not following:
        return True  # end of input, or only punctuation left
    character = following.group(1)
    return not character.islower()
