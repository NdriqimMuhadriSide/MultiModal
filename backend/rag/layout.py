"""
Layout analysis.

Responsibility: take the *visual* picture of a page and recover the order a
human would read it in, plus the shape of anything that isn't prose (tables,
running headers/footers). Pure text in, text out - this module never touches
pypdf, files, or HTTP. rag/loaders/pdf.py feeds it and owns everything PDF.

Why this exists
---------------
`page.extract_text()` walks the PDF's content stream and concatenates text in
the order the *generator* happened to emit it. That order has nothing to do
with reading order, so a two-column page comes back as:

    Neural networks learn by adjusting Transformers replaced recurrence
    weights through backpropagation. with self-attention, which lets

...which is two unrelated sentences interleaved. Embedding that produces a
vector for a sentence nobody wrote, so retrieval either misses the chunk or
returns it for the wrong question. Tables collapse the same way: the row/column
relationship that carried all the meaning ("BERT has 110M parameters") becomes a
flat bag of words.

The input we work from
----------------------
pypdf's `extraction_mode="layout"` re-renders the page as a fixed-width
*character grid* - it already did the coordinate maths and expressed each
glyph's x-position as a column index, padding with spaces:

    Neural networks learn by adjusting          Transformers replaced recurrence
    weights through backpropagation.           with self-attention, which lets

That is ASCII art of the page, and it makes the layout question purely
geometric: **where are the blank rows and blank columns?** Everything below is
built on that one primitive.

The algorithm: recursive XY-cut
-------------------------------
A classic page-segmentation technique. At each step, look at a rectangle of the
grid and try to slice it along whitespace:

    1. Horizontal cut - a run of fully blank rows separates stacked blocks
       (title / body / table / footnote). Recurse top -> bottom.
    2. Vertical cut - a run of fully blank columns (a "gutter") separates
       side-by-side blocks. Recurse left -> right.
    3. Neither - it's a leaf. Decide whether it reads as prose or as a table.

Recursion order *is* reading order, which is the whole point: the two-column
body above gets cut vertically, so the left column is fully emitted before the
right one starts.

The hard part is step 3, because "two text columns" and "a two-column table"
look identical to a whitespace detector - both are content, gutter, content.
They're told apart by shape, not by whitespace: prose columns are wide and
every line runs nearly the full measure; table cells are short and stubby. The
thresholds that encode that live in the constants below, each with the
false-positive it exists to prevent.

Where a heuristic is unsure, this module deliberately falls back to plain text.
A paragraph mistakenly kept as a paragraph costs nothing; a paragraph sliced
down the middle by a phantom gutter is unreadable.
"""
import re
from dataclasses import dataclass

# --- Horizontal cuts -------------------------------------------------------
# One blank row is ordinary paragraph spacing *inside* a block; two or more is
# the visual gap a designer puts *between* blocks. Cutting on a single blank
# row would shatter every paragraph into its own block.
MIN_BLANK_ROWS_BETWEEN_BANDS = 2

# --- Vertical cuts: prose columns ------------------------------------------
# A real inter-column gutter on a printed page is generous - it has to be, or
# the eye jumps between columns mid-line. Accidental "rivers" of whitespace do
# occur inside ordinary paragraphs when word breaks happen to line up, so the
# bar is set high enough that a coincidence is unlikely to clear it.
MIN_PROSE_GUTTER = 5
# ...and a coincidence that clears it is unlikely to survive four consecutive
# lines, or to leave columns wide enough to hold wrapped prose.
MIN_PROSE_COLUMN_ROWS = 4
MIN_PROSE_COLUMN_WIDTH = 15
# The widest column must fill a real share of the block. This is the check that
# separates a two-column page (each column ~40% of the width) from a table
# (each cell ~10%), and it is why a table is not sliced into vertical strips.
MIN_PROSE_COLUMN_WIDTH_RATIO = 0.2
# Prose columns are dense: text flows down them, so nearly every row has
# content in every column. A table with sparse cells fails this.
MIN_PROSE_COLUMN_FILL_RATIO = 0.6

# --- Vertical cuts: tables -------------------------------------------------
# Table gaps are tighter than page gutters, so the threshold is lower - which
# is safe only because MIN_TABLE_ROWS and the span check below do the real
# false-positive filtering.
MIN_TABLE_GUTTER = 3
# Header row + two data rows. Two-row "tables" are indistinguishable from a
# two-line paragraph with an unlucky whitespace river, so they stay prose.
MIN_TABLE_ROWS = 3
# A table row spans its columns; a ragged paragraph line does not. Not *every*
# row (real tables have empty cells), but most.
MIN_TABLE_ROW_SPAN_RATIO = 0.6

# --- Running headers / footers ---------------------------------------------
# A running header is visually detached: a line or two, alone above the body
# with a gap under it. So candidates are the page's first and last *bands*,
# and only when they're short enough to be boilerplate rather than content.
# Measuring a fixed fraction of the page instead would be far more fragile -
# body text on a normally-margined page starts around 11% down, so any band
# generous enough to catch a header also catches the first line of prose.
MAX_BOILERPLATE_BAND_ROWS = 2
# ...and that band still has to sit in a margin, which stops the sole band of
# a sparse page from being mistaken for its own header.
BOILERPLATE_MARGIN_RATIO = 0.25
# Repetition is the signal, and it needs enough pages to be meaningful. Below
# this, only the unambiguous page-number pattern is stripped.
BOILERPLATE_MIN_PAGES = 3
BOILERPLATE_MIN_REPEAT_RATIO = 0.6

# Depth guard. Six alternating cuts is far more nesting than a real page has;
# it exists so a pathological grid can't recurse forever.
MAX_SEGMENTATION_DEPTH = 6

# "12", "Page 12", "- 12 -", "Page 12 of 30" - a page number and nothing else.
_PAGE_NUMBER = re.compile(
    r"^[\-–—\s]*(?:page\s+)?\d+\s*(?:of\s+\d+)?[\-–—\s]*$",
    re.IGNORECASE,
)
_DIGIT_RUN = re.compile(r"\d+")
_SPACE_RUN = re.compile(r"\s+")


@dataclass
class Block:
    """
    One self-contained region of a page, in reading order.

    `kind` is "text" for prose, "table" for a detected grid, and "heading"
    once rag/structure.py has recognised one. Callers that only want a string
    can ignore all of it and read `text`.

    `level` and `section_path` are filled in by rag/structure.py, not here -
    this module can see that a line is short and alone, but not that it is
    *heading 2 inside chapter 3*, which only makes sense across a whole
    document. They stay on Block rather than in a parallel structure so a
    block never gets separated from where it sits in the document.
    """

    kind: str
    text: str
    # Heading depth, 1 being the top level. None for anything not a heading.
    level: int | None = None
    # The headings above this block, outermost first - e.g.
    # ("2. Methods", "2.1 Data Collection"). A heading includes itself, so it
    # groups with the prose it introduces.
    section_path: tuple[str, ...] = ()


def analyze_page(grid_text: str) -> list[Block]:
    """
    Segment one page's character grid into reading-order blocks.

    `grid_text` is the output of pypdf's layout extraction mode - text whose
    horizontal spacing is meaningful. Passing plain-mode text here is harmless
    but pointless: with no geometry to read, it yields a single text block.
    """
    return _segment(_to_grid(grid_text), depth=0)


def analyze_pages(grid_texts: list[str]) -> list[list[Block]]:
    """
    Segment a whole document, one list of blocks per page.

    Takes every page at once (rather than being called per page) because
    running headers and footers can only be recognised by comparing pages:
    a line is boilerplate precisely because it says the same thing on page 3
    that it said on page 2.
    """
    grids = [_to_grid(text) for text in grid_texts]
    return [_segment(grid, depth=0) for grid in _strip_boilerplate(grids)]


def blocks_to_text(blocks: list[Block]) -> str:
    """
    Flatten blocks into the page's plain text.

    Blocks are joined with a blank line, which is the strongest separator
    rag/text_splitter.py knows about - so a chunk boundary lands between two
    blocks in preference to anywhere inside one.
    """
    return "\n\n".join(block.text for block in blocks if block.text)


# --------------------------------------------------------------------------
# Grid handling
# --------------------------------------------------------------------------


def _to_grid(text: str) -> list[str]:
    """
    Turn extracted text into a list of rows whose character index is an
    x-coordinate.

    Tabs are expanded rather than stripped: a tab is a *horizontal* jump, and
    collapsing it to one space would move everything after it left and destroy
    the alignment the rest of this module reads. Trailing spaces are dropped so
    "is this row blank?" is a cheap emptiness test.
    """
    return [row.expandtabs(8).rstrip() for row in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]


def _trim_blank_edges(rows: list[str]) -> list[str]:
    start, end = 0, len(rows)
    while start < end and not rows[start].strip():
        start += 1
    while end > start and not rows[end - 1].strip():
        end -= 1
    return rows[start:end]


def _horizontal_bands(rows: list[str]) -> list[list[str]]:
    """
    Split rows wherever MIN_BLANK_ROWS_BETWEEN_BANDS or more blank rows sit in
    a row. Returns the bands with the separating blank rows removed.
    """
    bands: list[list[str]] = []
    current: list[str] = []
    blank_run = 0

    for row in rows:
        if row.strip():
            if blank_run >= MIN_BLANK_ROWS_BETWEEN_BANDS and current:
                bands.append(_trim_blank_edges(current))
                current = []
            elif blank_run and current:
                # A short gap belongs *inside* the band - it's a paragraph
                # break, and keeping it gives the text splitter a boundary.
                current.extend([""] * blank_run)
            blank_run = 0
            current.append(row)
        else:
            blank_run += 1

    if current:
        bands.append(_trim_blank_edges(current))
    return [band for band in bands if band]


def _column_gutters(rows: list[str], min_width: int) -> list[tuple[int, int]]:
    """
    Find runs of at least `min_width` character columns that are blank on every
    row - the vertical whitespace that separates side-by-side content.

    Only the interior counts: the blank space to the left of the first
    character and to the right of the last is the page margin, not a gutter.
    """
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return []

    occupied = [False] * width
    for row in rows:
        for index, char in enumerate(row):
            if char != " ":
                occupied[index] = True

    if not any(occupied):
        return []

    first = occupied.index(True)
    last = len(occupied) - 1 - occupied[::-1].index(True)

    gutters: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(first, last + 1):
        if not occupied[index]:
            if run_start is None:
                run_start = index
        else:
            if run_start is not None and index - run_start >= min_width:
                gutters.append((run_start, index))
            run_start = None
    return gutters


def _slice_columns(rows: list[str], gutters: list[tuple[int, int]]) -> list[list[str]]:
    """
    Cut every row at the gutters, producing one list of rows per column. All
    columns keep the same row count, so row *n* of each column is the same
    visual line - which is what makes table row assembly a simple zip.
    """
    if not gutters:
        return [rows]

    boundaries = [0] + [edge for gutter in gutters for edge in gutter] + [None]
    columns: list[list[str]] = []
    for index in range(0, len(boundaries) - 1, 2):
        start, stop = boundaries[index], boundaries[index + 1]
        columns.append([row[start:stop] if stop is not None else row[start:] for row in rows])
    return columns


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def _segment(rows: list[str], depth: int) -> list[Block]:
    rows = _trim_blank_edges(rows)
    if not rows:
        return []

    if depth < MAX_SEGMENTATION_DEPTH:
        bands = _horizontal_bands(rows)
        if len(bands) > 1:
            return [block for band in bands for block in _segment(band, depth + 1)]

        # No horizontal cut available, so look for a vertical one. Prose
        # columns are tested first: they demand a wider gutter, so a genuine
        # multi-column page can't be mistaken for a very wide table, while a
        # table's narrower gaps never clear the prose bar.
        prose_columns = _slice_columns(rows, _column_gutters(rows, MIN_PROSE_GUTTER))
        if _looks_like_prose_columns(rows, prose_columns):
            return [block for column in prose_columns for block in _segment(column, depth + 1)]

    cells = _slice_columns(rows, _column_gutters(rows, MIN_TABLE_GUTTER))
    if _looks_like_table(cells):
        return [Block(kind="table", text=_render_table(cells))]

    return [Block(kind="text", text=_render_text(rows))]


def _looks_like_prose_columns(rows: list[str], columns: list[list[str]]) -> bool:
    """
    Decide whether a gutter separates columns of running text (as opposed to
    the cells of a table, or a coincidence inside one paragraph).

    Prose columns are tall, wide, and densely filled; that combination is what
    every check below is measuring.
    """
    if len(columns) < 2 or len(rows) < MIN_PROSE_COLUMN_ROWS:
        return False

    block_width = max(len(row) for row in rows)
    widths = [max((len(row.strip()) for row in column), default=0) for column in columns]

    if min(widths) < MIN_PROSE_COLUMN_WIDTH:
        return False
    if max(widths) / block_width < MIN_PROSE_COLUMN_WIDTH_RATIO:
        return False

    # Text flows top to bottom, so a real column is occupied on most of its
    # rows. A table column with a few filled cells and a lot of blanks is not.
    for column in columns:
        filled = sum(1 for row in column if row.strip())
        if filled / len(column) < MIN_PROSE_COLUMN_FILL_RATIO:
            return False
    return True


def _looks_like_table(columns: list[list[str]]) -> bool:
    """
    Decide whether aligned whitespace gaps are really a table.

    The height requirement plus the span requirement are what keep an ordinary
    paragraph - whose word breaks may line up by chance on two lines - from
    being served back to the model as a table of nonsense.
    """
    if len(columns) < 2:
        return False

    row_count = len(columns[0])
    if row_count < MIN_TABLE_ROWS:
        return False

    # A column that is blank on all but one row is a stray word, not a column.
    for column in columns:
        if sum(1 for row in column if row.strip()) < 2:
            return False

    spanning = sum(
        1
        for index in range(row_count)
        if sum(1 for column in columns if column[index].strip()) >= 2
    )
    return spanning / row_count >= MIN_TABLE_ROW_SPAN_RATIO


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render_text(rows: list[str]) -> str:
    """
    Emit a prose block as plain lines.

    Indentation is dropped here on purpose: it was an x-coordinate, and once
    reading order is settled it carries no meaning - only noise that would be
    embedded along with the words. Blank rows survive as paragraph breaks.
    """
    rendered: list[str] = []
    for row in rows:
        stripped = row.strip()
        if stripped:
            rendered.append(stripped)
        elif rendered and rendered[-1]:
            rendered.append("")
    while rendered and not rendered[-1]:
        rendered.pop()
    return "\n".join(rendered)


def rows_to_markdown(rows: list[list[str]]) -> str:
    """
    Render rows of cells as a Markdown table.

    Markdown - rather than the source's own notation, whatever that was -
    because rag/text_splitter.py collapses runs of spaces before chunking, so
    any structure encoded in alignment is lost by the time a chunk is
    embedded. Pipes survive, which keeps the row/column relationship intact,
    and every LLM already reads the notation fluently.

    The first row becomes the header (with the `---` separator Markdown
    requires). For a detected PDF table that's a guess - a good one, since a
    table's top row is a header far more often than not, and the cost of being
    wrong is one mislabelled row, not lost data. For the other formats it is
    simply what the source said.

    Ragged rows are padded to the widest, since Markdown needs every row to
    have the same number of cells, and HTML in particular is full of tables
    whose rows disagree.

    Public because every loader needs it: a `<table>`, a CSV, and a Word table
    should all reach the model in the same notation as a PDF one.
    """
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    padded = [
        [cell.strip().replace("|", "\\|") for cell in row] + [""] * (width - len(row))
        for row in rows
    ]
    lines = [_render_table_row(padded[0]), _render_table_row(["---"] * width)]
    lines.extend(_render_table_row(row) for row in padded[1:])
    return "\n".join(lines)


def _render_table(columns: list[list[str]]) -> str:
    """Render detected grid cells, which arrive column-first, as a table."""
    return rows_to_markdown([list(row) for row in zip(*columns)])


def _render_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


# --------------------------------------------------------------------------
# Running headers / footers
# --------------------------------------------------------------------------


def _strip_boilerplate(grids: list[list[str]]) -> list[list[str]]:
    """
    Blank out running headers and footers across a document.

    Two rules, both confined to the top and bottom slivers of each page:

      * a line that repeats in the same margin on most pages is boilerplate -
        that's what "running header" means, and it's the only reliable signal,
        since a header is otherwise just a short line of text;
      * a line that is only a page number is boilerplate on sight, even in a
        one-page document, because it cannot be anything else.

    Boilerplate matters more than its size suggests. Repeated on every page, it
    is the single most common phrase in the document, so it turns up inside
    many chunks and drags them all toward the same region of embedding space -
    "ACME Research Quarterly" starts matching questions it has no answer to.

    Offending rows are replaced with blank rows rather than deleted, so row
    indices - and therefore the vertical gaps the segmenter reads - stay true
    to the page.
    """
    page_count = len(grids)
    bands = [_margin_rows(grid) for grid in grids]

    pages_seen: dict[str, set[int]] = {}
    for page_index, band_rows in enumerate(bands):
        for _, row in band_rows:
            key = _boilerplate_key(row)
            if key:
                pages_seen.setdefault(key, set()).add(page_index)

    repeat_threshold = page_count * BOILERPLATE_MIN_REPEAT_RATIO
    stripped = [list(grid) for grid in grids]
    for page_index, band_rows in enumerate(bands):
        for row_index, row in band_rows:
            key = _boilerplate_key(row)
            repeats = (
                page_count >= BOILERPLATE_MIN_PAGES
                and key
                and len(pages_seen[key]) >= repeat_threshold
            )
            if repeats or _PAGE_NUMBER.match(row.strip()):
                stripped[page_index][row_index] = ""
    return stripped


def _margin_rows(grid: list[str]) -> list[tuple[int, str]]:
    """
    Rows that could be a running header or footer, with their indices.

    A page needs at least two bands for the question to make sense: with only
    one, there is no way to tell a header from the body it would sit above.
    """
    bands = _band_row_indices(grid)
    if len(bands) < 2:
        return []

    candidates: list[int] = []
    top_limit = len(grid) * BOILERPLATE_MARGIN_RATIO
    bottom_limit = len(grid) * (1 - BOILERPLATE_MARGIN_RATIO)

    if len(bands[0]) <= MAX_BOILERPLATE_BAND_ROWS and bands[0][0] <= top_limit:
        candidates.extend(bands[0])
    if len(bands[-1]) <= MAX_BOILERPLATE_BAND_ROWS and bands[-1][-1] >= bottom_limit:
        candidates.extend(bands[-1])

    return [(index, grid[index]) for index in sorted(set(candidates))]


def _band_row_indices(grid: list[str]) -> list[list[int]]:
    """
    Group the indices of non-blank rows into bands, using the same blank-run
    rule as _horizontal_bands. Indices rather than text, because callers need
    to reach back into the original grid to blank a row out.
    """
    bands: list[list[int]] = []
    current: list[int] = []
    blank_run = 0

    for index, row in enumerate(grid):
        if row.strip():
            if blank_run >= MIN_BLANK_ROWS_BETWEEN_BANDS and current:
                bands.append(current)
                current = []
            blank_run = 0
            current.append(index)
        else:
            blank_run += 1

    if current:
        bands.append(current)
    return bands


def _boilerplate_key(row: str) -> str:
    """
    Normalise a margin line so the *same* header matches across pages.

    Digit runs collapse to "#" so "Page 3" and "Page 4" - which are the same
    footer - compare equal, and spacing/case differences are ignored.
    """
    return _DIGIT_RUN.sub("#", _SPACE_RUN.sub(" ", row.strip().lower()))
