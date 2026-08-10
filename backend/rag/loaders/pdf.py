"""
PDF loader.

Responsibility: turn raw PDF bytes into per-page text in *reading order*.
Nothing else - no chunking, no embeddings, no FastAPI/HTTP knowledge. This is
the first stage of the ingestion pipeline; rag/text_splitter.py picks up from
here.

Extraction runs in pypdf's layout mode, which re-renders each page as a
fixed-width character grid (x-positions expressed as spaces) instead of
concatenating text in content-stream order. rag/layout.py then reads that grid
geometrically to recover reading order, tables, and running headers/footers -
see that module for why the default mode is not good enough. Set
PDF_LAYOUT_MODE=false to fall back to plain extraction.

Pages that come back with no text at all are scans: the words are pixels, not
glyphs. Those - and only those - are handed to rag/ocr.py, which returns a
character grid of the same shape, so an OCR'd page is segmented exactly like a
digital one. OCR is optional; when it isn't installed the pages simply stay
empty and app/services/document_service.py reports why.
"""
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import settings
from rag import ocr
from rag.layout import analyze_pages, blocks_to_text
from rag.loaders.base import PageContent
from rag.structure import annotate_structure, normalize


def load_pdf(
    file_bytes: bytes,
    layout_aware: bool | None = None,
    use_ocr: bool | None = None,
) -> list[PageContent]:
    """
    Extract text from every page of a PDF, falling back to OCR for pages that
    have no text layer.

    `layout_aware` and `use_ocr` default to settings.pdf_layout_mode and
    settings.ocr_enabled. Passing them explicitly is mainly for tests and for
    comparing modes on a real document.

    Raises:
        ValueError: if the bytes are empty, not a valid PDF, or the PDF has
            no pages. Callers (the service layer) turn this into an HTTP 400.
    """
    if not file_bytes:
        raise ValueError("PDF file is empty.")

    try:
        reader = PdfReader(BytesIO(file_bytes))
    except PdfReadError as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc

    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages.")

    if layout_aware is None:
        layout_aware = settings.pdf_layout_mode
    if use_ocr is None:
        use_ocr = settings.ocr_enabled

    # extract_text() returns "" (not None) for pages with no extractable text
    # (e.g. a scanned image page with no OCR layer) - that's a valid result,
    # not an error, so we keep the page rather than skip it.
    raw_pages = [_extract(page, layout_aware) for page in reader.pages]

    ocr_pages: dict[int, str] = {}
    if use_ocr:
        ocr_pages = ocr.ocr_pdf_pages(file_bytes, _pages_needing_ocr(raw_pages))
        for index, grid in ocr_pages.items():
            raw_pages[index] = grid

    if not layout_aware:
        return [
            PageContent(page_number=index + 1, text=text, from_ocr=index in ocr_pages)
            for index, text in enumerate(raw_pages)
        ]

    # Analysed as a document, not page by page: a running header is only
    # recognisable by the fact that it repeats across pages, and a section
    # opened on page 4 runs until the next heading, wherever that lands.
    pages_blocks = analyze_pages(raw_pages)
    # OCR'd pages have no font metrics - their words came from pixels - so
    # their hint dict is empty and rag/structure.py falls back accordingly.
    font_sizes = [
        {} if index in ocr_pages else _font_sizes(page)
        for index, page in enumerate(reader.pages)
    ]
    pages_blocks = annotate_structure(pages_blocks, font_sizes)

    return [
        PageContent(
            page_number=index + 1,
            text=blocks_to_text(blocks),
            blocks=blocks,
            from_ocr=index in ocr_pages,
        )
        for index, blocks in enumerate(pages_blocks)
    ]


def _font_sizes(page) -> dict[str, float]:
    """
    Map each line of a page to the font size it was set in.

    Collected with a visitor callback during a second, plain-mode extraction
    pass. Layout mode can't supply this - it renders the page as characters on
    a grid, and a grid has no font sizes - but the size is the one measured
    signal that separates a heading from a short paragraph, so it's worth the
    extra pass.

    Only `font_size` is read from the callback, deliberately. The text matrix
    it also reports lags by one operation in pypdf, so positions taken from it
    are wrong; the font size is reported correctly alongside each fragment.

    Keyed on normalized text rather than position for the same reason: it's
    what can be matched reliably back to a line of the grid.
    """
    sizes: dict[str, float] = {}

    def visitor(text, _cm, _tm, _font_dict, font_size) -> None:
        key = normalize(text)
        # Largest wins: the same words can be drawn twice (an outline plus a
        # fill), and a heading repeated in a running header would otherwise be
        # recorded at the header's smaller size.
        if key and font_size and font_size > sizes.get(key, 0):
            sizes[key] = float(font_size)

    try:
        page.extract_text(visitor_text=visitor)
    except Exception:
        # Same reasoning as _extract: a font pypdf can't measure costs this
        # page its heading detection, not the document its text.
        return {}
    return sizes


def _pages_needing_ocr(raw_pages: list[str]) -> list[int]:
    """
    Indexes of the pages worth running OCR over, capped at settings.ocr_max_pages.

    "Almost empty" rather than "empty": a scanned page frequently carries a
    digitally-stamped page number or Bates number over the image, so it has a
    text layer holding half a dozen characters and none of the content.

    The cap is applied to the earliest pages rather than a sample spread
    through the document - a partial read of a long scan is far more useful
    when it's the beginning, and it keeps the behaviour predictable.
    """
    empty = [
        index
        for index, text in enumerate(raw_pages)
        if len("".join(text.split())) < settings.ocr_min_text_chars
    ]
    return empty[: max(0, settings.ocr_max_pages)]


def _extract(page, layout_aware: bool) -> str:
    """
    Pull text off one page, preferring layout mode.

    Layout mode leans on font metrics to place characters on the grid, and a
    PDF with a broken or exotic font descriptor can make it fail on a page that
    plain extraction handles fine. One bad page shouldn't cost the whole
    document its text, so that page silently drops to plain mode - it loses its
    geometry, and rag/layout.py degrades to treating it as one block of prose.
    """
    if not layout_aware:
        return page.extract_text() or ""

    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:
        return page.extract_text() or ""
