"""
OCR fallback for pages with no text layer.

Responsibility: recover text from PDF pages that have none - scanned
documents, photographed contracts, exported slide images. Everything else in
the ingestion pipeline assumes a PDF *contains* its text; this module is what
happens when it doesn't. rag/loaders/pdf.py decides which pages need it.

Why a PDF can have no text
--------------------------
A PDF is a container, not a format for text. A born-digital PDF holds glyph
codes and positions, so pypdf can read the words back out. A scan holds one
JPEG per page and nothing else - the words are pixels. `extract_text()`
correctly returns "" for those pages, and until now the whole document was
recorded as FAILED, which is honest but useless: the document was perfectly
readable, just not by a text extractor.

The pipeline here
-----------------
    PDF page -> raster image (pypdfium2) -> word boxes (Tesseract)
             -> character grid -> rag/layout.py

The last two steps are the interesting part. Tesseract can hand back plain
text, but asking for plain text throws away everything it knows about *where*
the words are - and a scanned two-column page or a scanned table has exactly
the same reading-order problem as a digital one. So this module asks for word
bounding boxes instead and paints them back onto a fixed-width character grid,
which is the same representation pypdf's layout mode produces. Scanned pages
then flow through the identical segmentation as digital ones and get column
ordering, table detection, and header stripping for free.

Optional by design
------------------
Tesseract is a system binary, not a Python package, so it may simply not be
there. `is_available()` reports that honestly and the loader carries on
without OCR rather than failing an upload - see
app/services/document_service.py, which turns the absence into a FAILED status
that says *why*.
"""
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from statistics import median

from app.core.config import settings

# Tesseract reports a 0-100 confidence per word. Anything under this is
# usually noise it found in JPEG artefacts or page edges - keeping it would
# put invented words into the embedding. -1 means "not a word box at all"
# (Tesseract uses it for structural rows) and is filtered separately.
MIN_WORD_CONFIDENCE = 40

# Two words belong to the same grid row when their vertical centres are
# within this fraction of a typical line's height. Generous enough to survive
# a slightly skewed scan, tight enough not to merge adjacent lines.
ROW_MERGE_TOLERANCE = 0.6

# Fallback used only if a page's word boxes give no usable measurement (e.g.
# a single one-character word); any positive number keeps the grid maths from
# dividing by zero, and a page that degenerate has no layout worth recovering.
FALLBACK_CHAR_WIDTH = 8.0


@dataclass
class Word:
    """One recognised word and where Tesseract found it, in image pixels."""

    text: str
    left: int
    top: int
    width: int
    height: int


def is_available() -> bool:
    """
    True when OCR can actually run: both Python packages import *and* the
    Tesseract binary answers.

    Checked rather than assumed because `pip install pytesseract` only
    installs a wrapper - the engine it wraps is a separate system install, so
    the import succeeding proves nothing.
    """
    return _probe() is None


def unavailable_reason() -> str | None:
    """
    Why OCR can't run, phrased for a user, or None when it can.

    Returned as a string rather than logged so the reason can travel all the
    way to the document list - "this scan failed" is unhelpful; "this scan
    failed because OCR isn't installed" is actionable.
    """
    if not settings.ocr_enabled:
        return "OCR is disabled (OCR_ENABLED=false)."
    return _probe()


@lru_cache(maxsize=1)
def _probe() -> str | None:
    """
    Cached capability check. Tests reset it with `_probe.cache_clear()`.

    Imports are deferred to here rather than done at module scope so a
    deployment without the OCR extras still starts, and merely reports OCR as
    unavailable instead of crashing on import.
    """
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return "OCR is not available: the 'pypdfium2' package is not installed."

    try:
        import pytesseract
    except ImportError:
        return "OCR is not available: the 'pytesseract' package is not installed."

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return (
            "OCR is not available: the Tesseract binary was not found. "
            "Install it with 'brew install tesseract' (macOS) or "
            "'apt-get install tesseract-ocr' (Debian/Ubuntu)."
        )
    return None


def ocr_pdf_pages(file_bytes: bytes, page_indexes: list[int]) -> dict[int, str]:
    """
    OCR the given 0-indexed pages and return each one's character grid.

    Only the requested pages are rendered. OCR costs roughly a second per
    page, so running it across a whole document to salvage two scanned inserts
    would be a poor trade - rag/loaders/pdf.py asks only for the pages that came
    back empty.

    Pages that fail to render or recognise are omitted from the result rather
    than raising: one unreadable page should cost that page's text, not the
    upload.
    """
    if not page_indexes or not is_available():
        return {}

    import pypdfium2
    import pytesseract

    # pypdfium2 renders at a scale factor relative to 72dpi, the unit PDF
    # coordinates are expressed in.
    scale = settings.ocr_dpi / 72

    grids: dict[int, str] = {}
    document = pypdfium2.PdfDocument(BytesIO(file_bytes))
    try:
        for page_index in page_indexes:
            try:
                image = document[page_index].render(scale=scale).to_pil()
                data = pytesseract.image_to_data(
                    image,
                    lang=settings.ocr_language,
                    output_type=pytesseract.Output.DICT,
                )
            except Exception:
                continue

            grid = words_to_grid(_to_words(data))
            if grid:
                grids[page_index] = grid
    finally:
        document.close()

    return grids


def ocr_image(image_bytes: bytes) -> str:
    """
    OCR a standalone image and return its character grid.

    The image counterpart to `ocr_pdf_pages`, added for agents/vision_agent.py.
    Everything after rasterization is shared - word boxes, the confidence
    floor, the grid painting - because a photographed receipt has the same
    reading-order problem as a scanned page and deserves the same treatment.

    What it skips is the pypdfium2 step, which exists only because a PDF
    page is not an image until something renders it. An upload already is
    one, so it goes straight to Tesseract, and there is no `ocr_dpi` here:
    the file has whatever resolution it has. A photo taken too far away
    reads badly and no setting on this side can fix it.

    Returns "" rather than raising when Tesseract is absent or the bytes are
    not a decodable image, matching `ocr_pdf_pages`' contract. Callers that
    need to explain the absence should ask `unavailable_reason()` - an empty
    grid on its own cannot distinguish "no OCR engine" from "a photo of a
    blank wall", and those need different things said about them.
    """
    if not image_bytes or not is_available():
        return ""

    import pytesseract
    from PIL import Image

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            data = pytesseract.image_to_data(
                image,
                lang=settings.ocr_language,
                output_type=pytesseract.Output.DICT,
            )
    except Exception:
        # An unreadable image costs its text, not the request - the caller
        # gets an empty grid and decides what to say about it.
        return ""

    return words_to_grid(_to_words(data))


def _to_words(data: dict) -> list[Word]:
    """Turn Tesseract's parallel-array TSV output into Word records."""
    words: list[Word] = []
    for index, text in enumerate(data["text"]):
        if not text or not text.strip():
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < MIN_WORD_CONFIDENCE:
            continue

        words.append(
            Word(
                text=text.strip(),
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
            )
        )
    return words


def words_to_grid(words: list[Word]) -> str:
    """
    Paint word boxes onto a fixed-width character grid.

    This is the bridge between "OCR found words at pixel coordinates" and the
    input rag/layout.py expects, and it works by inventing a monospace font:
    measure how many pixels one character takes on this page, then a word's
    column is simply its x-offset divided by that width. Vertical gaps become
    blank rows the same way, using the typical line pitch as the unit - which
    is what lets the segmenter see the space above a footer or between two
    stacked blocks.

    Public (not underscore-prefixed) because it's the piece worth testing on
    its own: it takes plain data, so its behaviour can be pinned down without
    Tesseract installed at all.
    """
    if not words:
        return ""

    char_width = _character_width(words)
    rows = _group_into_rows(words)
    pitch = _row_pitch(rows)

    lines: list[str] = []
    previous_centre: float | None = None
    for row in rows:
        centre = median([word.top + word.height / 2 for word in row])
        if previous_centre is not None:
            # A gap bigger than one line means real vertical whitespace on the
            # page. Reproducing it - rather than closing it up - is what keeps
            # blocks separable downstream.
            blank_rows = max(0, round((centre - previous_centre) / pitch) - 1)
            lines.extend([""] * blank_rows)
        lines.append(_render_row(row, char_width))
        previous_centre = centre

    return "\n".join(lines)


def _character_width(words: list[Word]) -> float:
    """
    Estimate the width of one character in pixels.

    Measured from multi-character words only: a one-character box is mostly
    the glyph's own bearing and would skew the estimate. The median resists
    the odd wildly-misrecognised box.
    """
    widths = [word.width / len(word.text) for word in words if len(word.text) > 1]
    return median(widths) if widths else FALLBACK_CHAR_WIDTH


def _group_into_rows(words: list[Word]) -> list[list[Word]]:
    """
    Group words into visual rows by vertical position.

    Deliberately *not* Tesseract's own line numbering: it groups per detected
    block, so on a two-column page the left and right column's first lines are
    different "lines" even though they sit side by side. The grid needs them on
    the same row - a row is a horizontal slice of the page, and the whole
    column-detection step depends on that being true.
    """
    typical_height = median([word.height for word in words])
    tolerance = max(1.0, typical_height * ROW_MERGE_TOLERANCE)

    rows: list[list[Word]] = []
    row_centres: list[float] = []
    for word in sorted(words, key=lambda item: item.top + item.height / 2):
        centre = word.top + word.height / 2
        if rows and centre - row_centres[-1] <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
            row_centres.append(centre)
    return rows


def _row_pitch(rows: list[list[Word]]) -> float:
    """
    The typical vertical distance between consecutive text rows, used as the
    unit for converting a gap in pixels into a number of blank rows.

    Taken from the smaller half of the observed gaps: the large ones are the
    very whitespace being measured, so including them would inflate the unit
    and quietly erase the gaps.
    """
    centres = [median([word.top + word.height / 2 for word in row]) for row in rows]
    gaps = sorted(
        later - earlier for earlier, later in zip(centres, centres[1:]) if later > earlier
    )
    if not gaps:
        return max(1.0, median([word.height for row in rows for word in row]))
    return median(gaps[: max(1, len(gaps) // 2)])


def _render_row(row: list[Word], char_width: float) -> str:
    """
    Lay one row's words out as text, each starting at its own column.

    Neighbouring words always end up separated by at least one space. The
    column each word wants is derived from an *estimated* character width, and
    on a proportional font that estimate drifts - so a word's computed column
    can land on or before the end of the word before it. Letting that stand
    would run two words together ("learnby adjusting"), inventing a token that
    no embedding or search will ever match. Sliding right costs a column of
    alignment; colliding costs the words themselves.
    """
    line = ""
    for word in sorted(row, key=lambda item: item.left):
        column = round(word.left / char_width)
        if line and column <= len(line):
            column = len(line) + 1
        line = line.ljust(column) + word.text
    return line
