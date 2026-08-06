import io

import pytest
from pypdf import PdfWriter

from rag.pdf_loader import load_pdf


def _blank_pdf_bytes(num_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


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
