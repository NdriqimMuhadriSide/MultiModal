"""
Document loaders.

Responsibility: pick the right loader for an uploaded file and hand back the
one shape the rest of the pipeline understands - `list[PageContent]`, each page
carrying blocks in reading order.

That shared shape is the point of the package. rag/structure.py,
rag/text_splitter.py, the embedder and the vector store were all written
against PDF pages, and none of them changed to gain four more formats: a
loader's whole job is to reach the same representation. Supporting the next
format is one new module and one entry in the table below.

What each loader is actually doing differs completely, though:

    PDF          reverse-engineers structure from where glyphs sit on the page
                 (rag/layout.py), because a PDF records appearance, not meaning
    DOCX/HTML/MD read structure straight out of the file, because these formats
                 record meaning - a heading says it is a heading
    CSV          has no structure to find beyond "it is a table", and spends its
                 effort on grouping rows so chunks keep their column headers

Dispatch is by file extension rather than the upload's Content-Type. Browsers
are unreliable here - a .md or .csv usually arrives as text/plain or
application/octet-stream, and the same .docx can arrive under three different
types - whereas the extension is what the user actually named the file.
"""
from pathlib import Path

from rag.loaders.base import PageContent, decode_text, single_page
from rag.loaders.csv import load_csv
from rag.loaders.docx import load_docx
from rag.loaders.html import load_html
from rag.loaders.markdown import load_markdown
from rag.loaders.pdf import load_pdf

__all__ = [
    "PageContent",
    "SUPPORTED_EXTENSIONS",
    "decode_text",
    "load_document",
    "load_pdf",
    "single_page",
    "supported_extensions_label",
]

# Extension -> loader. `.txt` goes to the Markdown loader on purpose: every
# rule in that parser is opt-in, so a file with no Markdown in it falls through
# to "blank lines separate paragraphs", which is the right reading of a .txt.
_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_markdown,
    ".csv": load_csv,
    ".tsv": load_csv,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)


def load_document(filename: str, file_bytes: bytes) -> list[PageContent]:
    """
    Load any supported document into pages of blocks.

    Raises:
        ValueError: if the file is empty, its extension isn't supported, or
            the chosen loader can't read it. Callers (the service layer) turn
            this into an HTTP 400 - the three cases are all "this file can't
            be ingested, and it isn't the server's fault".
    """
    if not file_bytes:
        raise ValueError("File is empty.")

    extension = Path(filename or "").suffix.lower()
    loader = _LOADERS.get(extension)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{extension or filename}'. "
            f"Supported types: {supported_extensions_label()}."
        )

    return loader(file_bytes)


def supported_extensions_label() -> str:
    """The supported extensions as a stable, human-readable list."""
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))
