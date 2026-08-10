"""
Document structure.

Responsibility: work out which blocks are headings, how deeply they nest, and
therefore which section every other block belongs to. rag/layout.py answers
"where is this on the page"; this module answers "where is this in the
document". It takes blocks in, returns the same blocks annotated - no PDF
knowledge, no files, no HTTP.

Why a chunk needs to know its section
-------------------------------------
Chunking cuts a document into pieces of a few hundred characters, and each
piece is embedded alone. That works only if the piece is self-contained, and
prose very often isn't:

    "We collected 500 samples over six weeks, discarding any that
     failed the calibration check."

Which study? Whose method? The paragraph never says, because the heading three
inches up the page already did - and the heading is in a different chunk. So
the embedding describes a generic sentence about samples, and a search for
"how was the pesticide data collected" doesn't match it. Carrying the section
path with the chunk puts the missing context back:

    2. Methods > 2.1 Field Sampling

That single line does two jobs. It goes into the embedded text, so the chunk
competes for the queries it actually answers, and it survives into the
retrieved result, so the model reading the chunk knows what it's reading.

How a heading is recognised
---------------------------
Three signals, of decreasing reliability:

  1. **Font size** - the honest one. A heading is set larger than body text,
     and pypdf reports the size of every text fragment, so the body size can
     be measured (whichever size the most *characters* are set in) and
     anything meaningfully bigger flagged. rag/loaders/pdf.py collects these.
  2. **Numbering** - "3.", "3.1", "Appendix B". Unambiguous when present, and
     it survives even when there is no font information at all, which is the
     situation on an OCR'd page.
  3. **Emphasis** - a short, capitalised line with no closing punctuation.
     Guesswork, so it is used *only* for documents with no font data, where
     the alternative is no structure at all.

A block must also be short and standalone. Length is doing a lot of work here:
whatever else a heading is, it is not a paragraph.

Levels come from ranking the heading font sizes largest-first, not from the
numbering depth - a document's title is usually unnumbered but outranks
"1. Introduction", and only the sizes know that. Numbering depth is the
fallback when there are no sizes to rank.
"""
import re
from dataclasses import dataclass, replace

from rag.layout import Block

# A heading is short. This is the single most effective filter - it removes
# every paragraph before any of the cleverer signals get a chance to be wrong.
HEADING_MAX_CHARS = 120
HEADING_MAX_LINES = 2

# How much larger than body text a line must be set to read as a heading.
# Deliberately close to 1: real documents often separate a subheading from
# body by a single point, and the length filter is already carrying the risk.
HEADING_SIZE_RATIO = 1.1

# Font sizes are rounded to this many points before being compared or ranked,
# so 11.0 and 11.04 - the same heading, measured through different transforms -
# don't become two levels.
FONT_SIZE_PRECISION = 0.5

# Deeper than this and "which section is this in" stops being useful. Extra
# levels collapse into the last one rather than being dropped.
MAX_HEADING_LEVEL = 6

# "3", "3.1", "3.1.2" - optionally followed by a dot, then the actual title.
# Depth is the number of components.
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
# "Chapter 4", "Appendix B", "Part II" - named divisions, always top level.
_NAMED_DIVISION = re.compile(
    r"^(?:chapter|section|part|appendix)\s+(?:\d+|[ivxlc]+|[a-z])\b", re.IGNORECASE
)
# A line ending like a sentence is prose, whatever else it looks like.
_TERMINAL_PUNCTUATION = (".", ",", ";", ":", "!", "?")

_WHITESPACE = re.compile(r"\s+")


@dataclass
class Heading:
    """A recognised heading and the level it sits at."""

    text: str
    level: int


def normalize(text: str) -> str:
    """
    Key used to look a line up in the font-size table.

    Whitespace and case are stripped because the two sides come from different
    pypdf passes - layout mode pads with spaces to place text on a grid, the
    visitor pass reports the raw fragment - and neither difference means
    anything about what the line says.
    """
    return _WHITESPACE.sub(" ", text).strip().lower()


def annotate_structure(
    pages_blocks: list[list[Block]],
    font_sizes: list[dict[str, float]] | None = None,
) -> list[list[Block]]:
    """
    Mark heading blocks and give every block its section path.

    Takes the whole document because both halves are document-wide questions:
    body font size is whatever most of the document is set in, and a section
    opened by a heading on page 4 runs until the next heading, wherever that
    falls.

    Blocks that already arrive as `kind="heading"` with a level keep it: a
    loader for a format that states its own structure has better information
    than any heuristic here. Everything else goes through detection.

    `font_sizes` is one dict per page, mapping a normalized line to the size it
    was set in - see rag/loaders/pdf.py. Omit it (or pass empty dicts, as
    happens for OCR'd pages and every non-PDF format) and detection falls back
    to numbering and capitalisation.
    """
    sizes = font_sizes or [{} for _ in pages_blocks]
    body_size = _body_font_size(pages_blocks, sizes)
    has_font_data = body_size is not None

    # Markdown, HTML and Word state their headings outright ("##", "<h2>",
    # style "Heading 2"). Those are facts, not signals, so they are taken as
    # given and only the remaining blocks go through detection - guessing at a
    # heading the document already labelled could only make it worse.
    declared: dict[tuple[int, int], int] = {}
    detected: dict[tuple[int, int], tuple[float | None, str]] = {}
    for page_index, blocks in enumerate(pages_blocks):
        for block_index, block in enumerate(blocks):
            if block.kind == "heading" and block.level:
                declared[(page_index, block_index)] = block.level
                continue
            size = _block_font_size(block, sizes[page_index])
            if _is_heading(block, size, body_size, has_font_data):
                detected[(page_index, block_index)] = (size, block.text)

    levels = _compact_levels(_assign_levels(detected) | declared)

    annotated: list[list[Block]] = []
    stack: list[str] = []
    for page_index, blocks in enumerate(pages_blocks):
        page: list[Block] = []
        for block_index, block in enumerate(blocks):
            level = levels.get((page_index, block_index))
            if level is not None:
                # Opening a level-N heading closes every section at N or
                # deeper: those are done, and anything that follows belongs to
                # this one.
                del stack[level - 1 :]
                stack.append(_single_line(block.text))
                page.append(
                    replace(
                        block,
                        kind="heading",
                        level=level,
                        section_path=tuple(stack),
                    )
                )
            else:
                page.append(replace(block, section_path=tuple(stack)))
        annotated.append(page)
    return annotated


def section_label(section_path: tuple[str, ...]) -> str:
    """Render a section path for a chunk header or search result."""
    return " > ".join(section_path)


# --------------------------------------------------------------------------
# Font sizes
# --------------------------------------------------------------------------


def _round_size(size: float) -> float:
    return round(size / FONT_SIZE_PRECISION) * FONT_SIZE_PRECISION


def _block_font_size(block: Block, page_font_sizes: dict[str, float]) -> float | None:
    """
    The size a block is set in: the largest of its lines' sizes.

    Largest rather than average because a two-line heading whose second line
    wrapped, or a heading with a small trailing marker, should still be judged
    by its heading-sized text.
    """
    sizes = [
        page_font_sizes[key]
        for key in (normalize(line) for line in block.text.split("\n"))
        if key in page_font_sizes
    ]
    return _round_size(max(sizes)) if sizes else None


def _body_font_size(
    pages_blocks: list[list[Block]], font_sizes: list[dict[str, float]]
) -> float | None:
    """
    The size most of the document's *text* is set in.

    Weighted by characters, not by lines: a report has one 24pt title and
    thousands of 10pt words, and only the character count gets that right.
    Counting lines would let a page of one-word headings outvote the prose.
    """
    weight: dict[float, int] = {}
    for blocks, page_font_sizes in zip(pages_blocks, font_sizes):
        for block in blocks:
            for line in block.text.split("\n"):
                size = page_font_sizes.get(normalize(line))
                if size is not None:
                    weight[_round_size(size)] = weight.get(_round_size(size), 0) + len(line)

    if not weight:
        return None
    return max(weight, key=lambda size: weight[size])


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _is_heading(
    block: Block, size: float | None, body_size: float | None, has_font_data: bool
) -> bool:
    if block.kind != "text" or not _is_short(block):
        return False

    # Font size is measured, not guessed, so it is allowed to call a heading
    # even when the text ends in a full stop.
    if size is not None and body_size is not None and size >= body_size * HEADING_SIZE_RATIO:
        return True

    # The weaker signals need the guard. Without it, step 1 of a numbered
    # procedure - "1. Rinse the sample thoroughly." - reads as a section
    # heading and swallows the rest of the document into a section named
    # after it.
    if _ends_like_a_sentence(block.text):
        return False
    if _numbering_depth(block.text) is not None:
        return True
    # Capitalisation alone is too weak to overrule font sizes that say "this is
    # body text", so it only speaks for documents that have no sizes at all.
    return not has_font_data and _looks_emphasised(block.text)


def _is_short(block: Block) -> bool:
    """Whatever else a heading is, it is not a paragraph."""
    lines = block.text.split("\n")
    return 0 < len(lines) <= HEADING_MAX_LINES and len(block.text) <= HEADING_MAX_CHARS


def _ends_like_a_sentence(text: str) -> bool:
    return text.rstrip().endswith(_TERMINAL_PUNCTUATION)


def _numbering_depth(text: str) -> int | None:
    """How deeply a heading's own numbering nests it, or None if unnumbered."""
    line = _single_line(text)
    match = _NUMBERED.match(line)
    if match:
        return min(len(match.group(1).split(".")), MAX_HEADING_LEVEL)
    if _NAMED_DIVISION.match(line):
        return 1
    return None


def _looks_emphasised(text: str) -> bool:
    """
    A short line shouting in capitals, with no sentence punctuation.

    Requires actual letters, so a stray "500" or "$1,200" left alone on a line
    is not promoted to a section heading.
    """
    letters = [character for character in _single_line(text) if character.isalpha()]
    return len(letters) >= 2 and all(character.isupper() for character in letters)


def _single_line(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------


def _compact_levels(levels: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    """
    Renumber levels so the shallowest heading in the document is level 1 and
    there are no gaps.

    This is correctness, not tidiness. The section stack closes a section by
    truncating to `level - 1`, which assumes the outermost heading is at 1. A
    README whose headings are all `##` and `###`, or a Word document with no
    Title style, would otherwise never truncate to an empty stack - so a new
    top-level section would nest *under* the previous one instead of replacing
    it, and every chunk after the first section would be filed in the wrong
    place.

    It also lets each loader map its format's own scale naively - Word's Title
    above Heading 1, HTML starting at <h2> - and be corrected here rather than
    each having to know about the others.
    """
    order = {level: rank for rank, level in enumerate(sorted(set(levels.values())), start=1)}
    return {key: min(order[level], MAX_HEADING_LEVEL) for key, level in levels.items()}


def _assign_levels(
    headings: dict[tuple[int, int], tuple[float | None, str]],
) -> dict[tuple[int, int], int]:
    """
    Turn a set of headings into nesting levels.

    Font size decides when it can: rank the distinct heading sizes largest
    first and the rank *is* the level. This beats reading the numbering,
    because the numbering is blind to everything above it - a title and a
    "1. Introduction" both look like level 1, when the title clearly outranks
    it, and only the sizes know that.

    A heading with no size falls back to its numbering depth, and a heading
    with neither is treated as top level - the safest guess, since it opens a
    section that then holds everything until the next heading.
    """
    ranked = sorted({size for size, _ in headings.values() if size is not None}, reverse=True)
    by_size = {size: min(index + 1, MAX_HEADING_LEVEL) for index, size in enumerate(ranked)}

    levels: dict[tuple[int, int], int] = {}
    for key, (size, text) in headings.items():
        if size is not None and size in by_size:
            levels[key] = by_size[size]
        else:
            levels[key] = _numbering_depth(text) or 1
    return levels
