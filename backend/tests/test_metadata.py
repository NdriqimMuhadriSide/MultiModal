"""
Document metadata tests.

Each format keeps its metadata somewhere different, so most of these are about
finding it. The rest are about not trusting it: an embedded title is frequently
a filename, and an embedded date is frequently unparseable.
"""
import io

import docx
import pytest
from pypdf import PdfWriter

from rag.loaders import load_document
from rag.metadata import DocumentMetadata, extract_metadata


def _pdf(info: dict) -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata(info)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _docx(**properties) -> bytes:
    document = docx.Document()
    for key, value in properties.items():
        setattr(document.core_properties, key, value)
    document.add_paragraph("Body text.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _extract(filename: str, data: bytes) -> DocumentMetadata:
    """Extract the way ingestion does - with the loaded pages available."""
    return extract_metadata(filename, data, load_document(filename, data))


# --- Per-format extraction --------------------------------------------------


def test_pdf_info_dictionary_is_read():
    metadata = extract_metadata(
        "report.pdf",
        _pdf(
            {
                "/Title": "Annual Safety Review",
                "/Author": "R. Team",
                "/Subject": "Site inspections",
                "/CreationDate": "D:20260302120000+01'00'",
            }
        ),
    )

    assert metadata.title == "Annual Safety Review"
    assert metadata.author == "R. Team"
    assert metadata.subject == "Site inspections"
    assert metadata.document_date == "2026-03-02"


def test_docx_core_properties_are_read():
    metadata = _extract("report.docx", _docx(title="Field Report", author="A. Scientist"))

    assert metadata.title == "Field Report"
    assert metadata.author == "A. Scientist"
    # python-docx stamps a created date on every new document.
    assert metadata.document_date is not None


def test_html_title_and_meta_tags_are_read():
    html = (
        b"<html><head><title>Quarterly Update</title>"
        b'<meta name="author" content="Comms Team">'
        b'<meta name="description" content="What shipped this quarter.">'
        b'<meta property="article:published_time" content="2026-01-15T09:00:00Z">'
        b"</head><body><p>Body.</p></body></html>"
    )

    metadata = _extract("update.html", html)

    assert metadata.title == "Quarterly Update"
    assert metadata.author == "Comms Team"
    assert metadata.subject == "What shipped this quarter."
    assert metadata.document_date == "2026-01-15"


def test_meta_tags_after_body_are_ignored():
    """A stray <meta> buried in the page shouldn't be able to rename the document."""
    html = (
        b"<html><head><title>Real Title</title></head>"
        b'<body><meta name="title" content="Injected"><p>Body.</p></body></html>'
    )

    assert _extract("page.html", html).title == "Real Title"


def test_markdown_front_matter_is_read():
    source = b'---\ntitle: "Design Notes"\nauthor: N. Writer\ndate: 2026-02-11\n---\n\n# Heading\n\nBody.\n'

    metadata = _extract("notes.md", source)

    assert metadata.title == "Design Notes"
    assert metadata.author == "N. Writer"
    assert metadata.document_date == "2026-02-11"


def test_front_matter_is_not_ingested_as_content():
    """Left in, "layout: post" would be a searchable paragraph."""
    source = b"---\ntitle: Design Notes\nlayout: post\n---\n\n# Heading\n\nBody.\n"

    blocks = load_document("notes.md", source)[0].blocks

    assert [block.text for block in blocks] == ["Heading", "Body."]


def test_an_unclosed_front_matter_fence_is_left_alone():
    # It's a horizontal rule at the top of the document, not front matter.
    source = b"---\n\nJust a document that starts with a rule.\n"

    assert load_document("notes.md", source)[0].blocks


def test_csv_has_no_metadata_to_find():
    metadata = _extract("orders.csv", b"city,orders\nBergen,4412\n")

    assert metadata.author is None
    assert metadata.document_date is None
    # ...but a title is always resolved, so the list has something to show.
    assert metadata.title == "orders"


# --- Title fallbacks --------------------------------------------------------


def test_the_first_heading_is_used_when_no_title_is_embedded():
    metadata = _extract("8f2a-final-v3.md", b"# Annual Safety Review\n\nBody.\n")

    assert metadata.title == "Annual Safety Review"


def test_the_filename_is_the_last_resort():
    metadata = _extract("meeting-notes.txt", b"Just some text with no heading.\n")

    assert metadata.title == "meeting-notes"


@pytest.mark.parametrize(
    "embedded",
    ["Microsoft Word - quarterly.doc", "report.pdf", "Untitled", "thesis.tex"],
)
def test_titles_that_are_really_filenames_are_rejected(embedded):
    """
    Word, LaTeX and export tools all stamp the source filename into /Title.
    Every one of them hides the real title sitting in the first heading.
    """
    metadata = extract_metadata(
        "whatever.pdf",
        _pdf({"/Title": embedded}),
        load_document("notes.md", b"# The Real Title\n\nBody."),
    )

    assert metadata.title == "The Real Title"


def test_a_genuine_title_is_not_mistaken_for_a_filename():
    metadata = extract_metadata("whatever.pdf", _pdf({"/Title": "Reporting Standards"}))

    assert metadata.title == "Reporting Standards"


# --- Dates ------------------------------------------------------------------


@pytest.mark.parametrize(
    "written,expected",
    [
        ("2026-03-02", "2026-03-02"),
        ("2026-03-02T12:00:00Z", "2026-03-02"),
        ("02/03/2026", "2026-03-02"),
        ("2 March 2026", "2026-03-02"),
        ("March 2, 2026", "2026-03-02"),
    ],
)
def test_dates_are_normalised_to_iso(written, expected):
    """One shape out, however the author typed it - a date only earns its
    place in a filter if it can be compared."""
    source = f"---\ndate: {written}\n---\n\nBody.\n".encode()

    assert _extract("notes.md", source).document_date == expected


def test_an_unparseable_date_is_dropped_rather_than_stored():
    source = b"---\ndate: sometime last spring\n---\n\nBody.\n"

    assert _extract("notes.md", source).document_date is None


# --- Robustness -------------------------------------------------------------


def test_malformed_metadata_never_fails_an_upload():
    """A file whose /Info dictionary is broken is still a good document."""
    metadata = extract_metadata("report.pdf", b"not a pdf at all")

    assert metadata.title == "report"
    assert metadata.author is None


def test_a_file_with_no_metadata_yields_empty_fields():
    metadata = extract_metadata("blank.pdf", _pdf({}))

    assert metadata.author is None
    assert metadata.subject is None
    assert metadata.document_date is None
