"""
PDF loader.

Responsibility: turn raw PDF bytes into per-page plain text. Nothing else -
no chunking, no embeddings, no FastAPI/HTTP knowledge. This is the first
stage of the ingestion pipeline; rag/text_splitter.py picks up from here.
"""
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass
class PageContent:
    """Plain text extracted from a single PDF page."""

    page_number: int  # 1-indexed, matches how a human would refer to a page
    text: str


def load_pdf(file_bytes: bytes) -> list[PageContent]:
    """
    Extract text from every page of a PDF.

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

    pages: list[PageContent] = []
    for index, page in enumerate(reader.pages):
        # extract_text() returns "" (not None) for pages with no extractable
        # text (e.g. a scanned image page with no OCR layer) - that's a valid
        # result, not an error, so we keep the page rather than skip it.
        text = page.extract_text() or ""
        pages.append(PageContent(page_number=index + 1, text=text))

    return pages
