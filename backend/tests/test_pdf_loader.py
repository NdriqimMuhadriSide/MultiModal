import io

import pytest
from pypdf import PdfWriter

from rag import ocr
from rag.loaders.pdf import _pages_needing_ocr, load_pdf
from tests.pdf_builder import build_pdf, build_scanned_pdf, table_rows, two_column_page

requires_tesseract = pytest.mark.skipif(
    not ocr.is_available(), reason="Tesseract is not installed"
)

LEFT_COLUMN = [
    "Neural networks learn by adjusting",
    "weights through backpropagation.",
    "Each layer transforms its input",
    "into a richer representation.",
]
RIGHT_COLUMN = [
    "Transformers replaced recurrence",
    "with self-attention, which lets",
    "every token look at every other",
    "token in a single step.",
]
TABLE = [
    ["Model", "Params", "Accuracy"],
    ["BERT", "110M", "88.5"],
    ["GPT-2", "1.5B", "91.2"],
]


def _blank_pdf_bytes(num_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _report_pdf_bytes(page_count: int = 3) -> bytes:
    """A page with a running header, two text columns, a table, and a footer."""
    pages = []
    for page_number in range(1, page_count + 1):
        items = [(72, 740, 9, "ACME RESEARCH QUARTERLY")]
        items += two_column_page(LEFT_COLUMN, RIGHT_COLUMN, top=700)
        items += table_rows(TABLE, column_x=[72, 220, 330], top=600)
        items.append((300, 40, 9, f"Page {page_number}"))
        pages.append(items)
    return build_pdf(pages)


def test_load_pdf_returns_one_entry_per_page():
    pages = load_pdf(_blank_pdf_bytes(num_pages=3))

    assert len(pages) == 3
    assert [page.page_number for page in pages] == [1, 2, 3]
    # Blank pages have no extractable text - should be "" not an error.
    assert all(page.text == "" for page in pages)


def test_load_pdf_rejects_empty_bytes():
    with pytest.raises(ValueError):
        load_pdf(b"")


def test_load_pdf_rejects_invalid_pdf():
    with pytest.raises(ValueError):
        load_pdf(b"this is not a pdf")


def test_columns_are_interleaved_without_layout_analysis():
    """
    Pins the failure mode the layout work exists to fix.

    Without geometry, text comes back in content-stream order, so a page laid
    out row by row across two columns yields lines that splice one column's
    sentence onto the other's.
    """
    text = load_pdf(_report_pdf_bytes(), layout_aware=False)[0].text

    assert "Neural networks learn by adjusting Transformers replaced recurrence" in text


def test_columns_are_read_in_order_with_layout_analysis():
    text = load_pdf(_report_pdf_bytes(), layout_aware=True)[0].text

    assert "Neural networks learn by adjusting Transformers replaced recurrence" not in text
    for line in LEFT_COLUMN + RIGHT_COLUMN:
        assert line in text
    # Every line of the left column comes before any line of the right one.
    assert max(text.index(line) for line in LEFT_COLUMN) < min(
        text.index(line) for line in RIGHT_COLUMN
    )


def test_tables_survive_as_markdown():
    page = load_pdf(_report_pdf_bytes(), layout_aware=True)[0]

    assert "| BERT | 110M | 88.5 |" in page.text
    assert "| GPT-2 | 1.5B | 91.2 |" in page.text
    assert [block.kind for block in page.blocks] == ["text", "text", "table"]


def test_running_header_and_footer_are_dropped():
    pages = load_pdf(_report_pdf_bytes(page_count=3), layout_aware=True)

    for page in pages:
        assert "ACME RESEARCH QUARTERLY" not in page.text
        assert "Page" not in page.text


def test_header_and_footer_are_kept_without_layout_analysis():
    text = load_pdf(_report_pdf_bytes(), layout_aware=False)[0].text

    assert "ACME RESEARCH QUARTERLY" in text


def test_blocks_are_empty_without_layout_analysis():
    pages = load_pdf(_report_pdf_bytes(), layout_aware=False)

    assert all(page.blocks == [] for page in pages)


def test_layout_mode_follows_settings_by_default(monkeypatch):
    from rag.loaders import pdf as pdf_loader

    monkeypatch.setattr(pdf_loader.settings, "pdf_layout_mode", False)
    assert load_pdf(_report_pdf_bytes())[0].blocks == []

    monkeypatch.setattr(pdf_loader.settings, "pdf_layout_mode", True)
    assert load_pdf(_report_pdf_bytes())[0].blocks != []


# --- Heading structure ------------------------------------------------------


def _structured_report_pdf_bytes() -> bytes:
    """A title, two numbered sections, a subsection, and prose under each."""
    return build_pdf(
        [
            [
                (72, 740, 20, "Annual Research Report"),
                (72, 690, 14, "1. Introduction"),
                (72, 660, 10, "This report covers the year in review and the"),
                (72, 646, 10, "work carried out by the research team."),
                (72, 600, 12, "1.1 Scope"),
                (72, 570, 10, "Everything shipped between January and December."),
                (72, 510, 14, "2. Methods"),
                (72, 480, 10, "We collected 500 samples over six weeks."),
            ]
        ]
    )


def test_headings_are_detected_and_ranked_by_font_size():
    blocks = load_pdf(_structured_report_pdf_bytes())[0].blocks

    headings = [(block.text, block.level) for block in blocks if block.kind == "heading"]
    assert headings == [
        ("Annual Research Report", 1),
        ("1. Introduction", 2),
        ("1.1 Scope", 3),
        ("2. Methods", 2),
    ]


def test_prose_carries_the_section_it_sits_under():
    blocks = load_pdf(_structured_report_pdf_bytes())[0].blocks

    def section_of(prefix: str) -> tuple[str, ...]:
        return next(
            block.section_path
            for block in blocks
            if block.kind == "text" and block.text.startswith(prefix)
        )

    assert section_of("This report covers") == ("Annual Research Report", "1. Introduction")
    assert section_of("Everything shipped") == (
        "Annual Research Report",
        "1. Introduction",
        "1.1 Scope",
    )
    # "2. Methods" is a sibling of "1. Introduction", so it closes 1.1 as well.
    assert section_of("We collected 500") == ("Annual Research Report", "2. Methods")


def test_structure_is_skipped_without_layout_analysis():
    pages = load_pdf(_structured_report_pdf_bytes(), layout_aware=False)

    assert pages[0].blocks == []
    assert "1. Introduction" in pages[0].text


# --- OCR fallback for scanned pages ----------------------------------------


def _scanned_report_pdf_bytes(page_count: int = 1) -> bytes:
    """The same report as above, but as page images - i.e. a scan."""
    pages = []
    for page_number in range(1, page_count + 1):
        items = [(72, 760, 9, "ACME RESEARCH QUARTERLY")]
        items += two_column_page(LEFT_COLUMN, RIGHT_COLUMN, top=690, leading=18)
        items += table_rows(TABLE, column_x=[72, 260, 400], top=560, leading=18)
        items.append((300, 60, 9, f"Page {page_number}"))
        pages.append(items)
    return build_scanned_pdf(pages)


def test_scanned_pdf_has_no_text_without_ocr():
    """The failure this fallback exists for: the words are pixels."""
    pages = load_pdf(_scanned_report_pdf_bytes(), use_ocr=False)

    assert pages[0].text == ""
    assert pages[0].from_ocr is False


@requires_tesseract
def test_scanned_pdf_text_is_recovered_by_ocr():
    page = load_pdf(_scanned_report_pdf_bytes(), use_ocr=True)[0]

    assert page.from_ocr is True
    assert "backpropagation" in page.text
    assert "self-attention" in page.text


@requires_tesseract
def test_scanned_pdf_keeps_its_columns_and_table():
    """
    OCR'd pages go through the same layout analysis as digital ones, because
    rag/ocr.py hands back a character grid rather than a flat string.
    """
    page = load_pdf(_scanned_report_pdf_bytes(), use_ocr=True)[0]

    # The banner is flagged as a heading by the capitalisation fallback: an
    # OCR'd page carries no font sizes, so that is the only signal left.
    assert [block.kind for block in page.blocks] == ["heading", "text", "text", "table"]
    assert page.text.index("backpropagation") < page.text.index("self-attention")
    assert "| BERT | 110M | 88.5 |" in page.text


@requires_tesseract
def test_scanned_running_header_and_footer_are_dropped():
    """
    Boilerplate detection compares pages, and it can only do that because OCR
    output re-enters the pipeline at the same point digital text does.
    """
    pages = load_pdf(_scanned_report_pdf_bytes(page_count=3), use_ocr=True)

    assert all(page.from_ocr for page in pages)
    for page in pages:
        assert "QUARTERLY" not in page.text
        assert "backpropagation" in page.text


def test_pages_with_text_are_not_sent_to_ocr(monkeypatch):
    requested: list[list[int]] = []
    monkeypatch.setattr(
        ocr, "ocr_pdf_pages", lambda file_bytes, indexes: requested.append(indexes) or {}
    )

    load_pdf(_report_pdf_bytes(page_count=2), use_ocr=True)

    assert requested == [[]]


def test_ocr_result_replaces_the_empty_page(monkeypatch):
    monkeypatch.setattr(
        ocr, "ocr_pdf_pages", lambda file_bytes, indexes: {index: "Recovered text." for index in indexes}
    )

    pages = load_pdf(_scanned_report_pdf_bytes(), use_ocr=True)

    assert pages[0].text == "Recovered text."
    assert pages[0].from_ocr is True


def test_unavailable_ocr_leaves_the_page_empty_instead_of_failing(monkeypatch):
    monkeypatch.setattr(ocr, "is_available", lambda: False)

    pages = load_pdf(_scanned_report_pdf_bytes(), use_ocr=True)

    assert pages[0].text == ""
    assert pages[0].from_ocr is False


def test_a_page_with_only_a_stamped_page_number_still_needs_ocr(monkeypatch):
    # A scanned page often carries a digitally-stamped number over the image,
    # so "has a text layer" is not the same as "has its text".
    monkeypatch.setattr("rag.loaders.pdf.settings.ocr_min_text_chars", 32)

    assert _pages_needing_ocr(["Page 12", "A" * 200]) == [0]


def test_ocr_is_capped_at_a_maximum_number_of_pages(monkeypatch):
    # OCR runs inside a synchronous upload request, so an unbounded scan
    # would hold the connection open until it timed out.
    monkeypatch.setattr("rag.loaders.pdf.settings.ocr_min_text_chars", 32)
    monkeypatch.setattr("rag.loaders.pdf.settings.ocr_max_pages", 3)

    assert _pages_needing_ocr([""] * 10) == [0, 1, 2]


def test_page_that_breaks_layout_extraction_falls_back_to_plain_text(monkeypatch):
    from pypdf._page import PageObject

    original = PageObject.extract_text

    def failing_layout(self, *args, **kwargs):
        if kwargs.get("extraction_mode") == "layout":
            raise ValueError("broken font descriptor")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", failing_layout)

    text = load_pdf(_report_pdf_bytes(), layout_aware=True)[0].text

    # No geometry left to read, so the page degrades to plain extraction
    # rather than losing its text entirely.
    assert "Neural networks learn by adjusting" in text
