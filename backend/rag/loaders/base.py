"""
What every loader returns, and the few things they all need.

Kept in its own module so the loaders and the dispatcher in __init__.py can
share it without importing each other.

The shared contract is `list[PageContent]`, each page carrying `Block`s in
reading order. That is the seam the whole ingestion pipeline hangs off: once a
file - PDF, Word, HTML, Markdown, CSV - has been turned into blocks,
rag/structure.py, rag/text_splitter.py, the embedder and the vector store
neither know nor care where it came from. Adding a format means writing one
loader, not touching the pipeline.
"""
from dataclasses import dataclass, field

from rag.layout import Block

# Tried in order. utf-8-sig first because a BOM is common in files exported
# from Excel and Windows editors, and reading one as plain utf-8 leaves a
# zero-width character glued to the first heading. latin-1 last because it
# cannot fail - every byte maps to some character - so it guarantees a result
# rather than a crash on an unknown legacy encoding.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


@dataclass
class PageContent:
    """
    Text extracted from a single page, in reading order.

    "Page" is a PDF idea. The other formats have no pages at all, so their
    loaders return exactly one PageContent covering the whole file: inventing
    page breaks by character count would put a number in every citation that
    corresponds to nothing a reader could look up. Those documents are cited by
    section instead - see rag/structure.py.
    """

    page_number: int  # 1-indexed, matches how a human would refer to a page
    text: str
    # Reading-order regions of the page ("text", "heading" or "table"). Empty
    # when layout analysis is off for a PDF. Callers that just want the words
    # can ignore this and read `text`, which is these blocks flattened.
    blocks: list[Block] = field(default_factory=list)
    # True when this page had no text layer and its words were recovered by
    # OCR. Recorded because OCR'd text is a transcription, not the source
    # text - useful for reporting, and for deciding how much to trust it.
    from_ocr: bool = False


def decode_text(file_bytes: bytes) -> str:
    """
    Decode a text-based upload without knowing its encoding.

    Nothing in an HTTP upload reliably says what encoding a .md or .csv is in,
    so this tries the plausible ones in order of likelihood. The last one
    always succeeds, which matters more than being right about an unusual
    encoding: a document that ingests with a few mangled accents is still
    searchable, while one that raises never gets ingested at all.
    """
    for encoding in _ENCODINGS:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Unreachable - latin-1 accepts any byte sequence - but a loader should
    # never depend on that being true of whatever _ENCODINGS ends with.
    return file_bytes.decode("utf-8", errors="replace")


def single_page(blocks: list[Block]) -> list[PageContent]:
    """
    Wrap a whole-document block list as the one page a pageless format has,
    and give every block its section path.

    Structure annotation happens here rather than in each loader because every
    format that lands in this helper is in the same position: it has already
    declared its own headings, and it has no font metrics to offer. The PDF
    loader does its own annotation instead - it is the one format that has to
    *infer* headings, and it needs to pass the font sizes it collected.
    """
    from rag.layout import blocks_to_text
    from rag.structure import annotate_structure

    annotated = annotate_structure([blocks])[0]
    return [PageContent(page_number=1, text=blocks_to_text(annotated), blocks=annotated)]
