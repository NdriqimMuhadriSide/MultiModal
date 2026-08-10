"""
Document metadata.

Responsibility: answer "what *is* this document" - its title, who wrote it,
and when - as opposed to rag/loaders/, which answers "what does it say".

Why it's worth having
---------------------
Retrieval so far can only rank chunks by what they contain. Metadata is the
axis it can't see: "the 2025 safety report", "anything by the compliance team",
"only documents from last quarter". Those are filters, not queries - no amount
of semantic similarity finds a date - and every one of them needs a field that
was never read off the file.

It also fixes a smaller, more visible thing: a document list showing
`8f2a-final-v3-FINAL.pdf` when the file itself says "Annual Safety Review
2025".

Where it hides
--------------
Every format keeps it somewhere different, and none of them where the text is:

    PDF    the trailer's /Info dictionary
    DOCX   docProps/core.xml inside the zip
    HTML   <title> and <meta> tags in the head
    MD     the "---" fenced front matter block at the top
    CSV    nowhere at all - it has no place to put any

So this is a second, separate read of the bytes rather than something the
loaders return alongside their text. That costs one extra parse per upload,
which is small next to embedding every chunk, and it buys a module that has
exactly one job and can be asked for metadata without ingesting anything.

What's embedded is often wrong, too - see `_looks_like_a_filename`.
"""
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

from rag.loaders.base import PageContent, decode_text
from rag.loaders.markdown import split_front_matter

# Front-matter / <meta> keys worth reading, in the order they're preferred.
_TITLE_KEYS = ("title", "og:title", "dc.title")
_AUTHOR_KEYS = ("author", "authors", "creator", "dc.creator", "article:author")
_SUBJECT_KEYS = ("subject", "description", "og:description", "summary")
_DATE_KEYS = ("date", "created", "published", "article:published_time", "dc.date")

# Producers that stamp the source filename into /Title. Word is the notorious
# one ("Microsoft Word - report.doc"), but any title that is really a filename
# tells the reader nothing they didn't already have.
_FILENAME_TITLE_PREFIXES = ("microsoft word - ", "microsoft powerpoint - ", "untitled")
_FILENAME_TITLE_SUFFIXES = (".doc", ".docx", ".pdf", ".rtf", ".odt", ".tex", ".pages")

# Date formats seen in HTML <meta> tags and Markdown front matter, tried in
# order. PDF and DOCX hand back real datetimes and skip this entirely.
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y", "%Y/%m/%d")


@dataclass
class DocumentMetadata:
    """
    What a document says about itself. Every field is optional, because most
    documents fill in some of them and plenty fill in none.
    """

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    # ISO 8601 date the *document* carries, which is not when it was uploaded.
    # Stored as a string rather than a date so it survives a round trip through
    # SQLite and Chroma metadata unchanged, and sorts correctly as text.
    document_date: str | None = None


def extract_metadata(
    filename: str, file_bytes: bytes, pages: list[PageContent] | None = None
) -> DocumentMetadata:
    """
    Read a document's own metadata, filling gaps from the content.

    `pages` is the already-loaded document, used only for the title fallback -
    a document with no embedded title usually opens with one as a heading, and
    that is a far better answer than the filename.

    Never raises. A file whose metadata is missing or malformed is still a
    perfectly good document, and failing an upload over its /Info dictionary
    would be absurd.
    """
    extension = Path(filename or "").suffix.lower()
    try:
        metadata = _EXTRACTORS.get(extension, _no_metadata)(file_bytes)
    except Exception:  # noqa: BLE001 - metadata is never worth failing over
        metadata = DocumentMetadata()

    metadata.title = _resolve_title(metadata.title, filename, pages)
    return metadata


# --------------------------------------------------------------------------
# Per-format extraction
# --------------------------------------------------------------------------


def _no_metadata(_file_bytes: bytes) -> DocumentMetadata:
    """For CSV and plain text, which have nowhere to keep any."""
    return DocumentMetadata()


def _from_pdf(file_bytes: bytes) -> DocumentMetadata:
    from pypdf import PdfReader

    info = PdfReader(BytesIO(file_bytes)).metadata
    if info is None:
        return DocumentMetadata()

    # pypdf parses PDF's own "D:YYYYMMDDHHmmSS+HH'mm'" date format for us, but
    # raises on the malformed ones plenty of producers emit.
    try:
        stamped = info.creation_date or info.modification_date
    except Exception:  # noqa: BLE001
        stamped = None

    return DocumentMetadata(
        title=_clean(info.title),
        author=_clean(info.author),
        subject=_clean(info.subject),
        document_date=_iso(stamped),
    )


def _from_docx(file_bytes: bytes) -> DocumentMetadata:
    import docx

    properties = docx.Document(BytesIO(file_bytes)).core_properties
    return DocumentMetadata(
        title=_clean(properties.title),
        author=_clean(properties.author),
        subject=_clean(properties.subject or properties.comments),
        document_date=_iso(properties.created or properties.modified),
    )


def _from_html(file_bytes: bytes) -> DocumentMetadata:
    head = _HeadReader()
    head.feed(decode_text(file_bytes))
    head.close()
    return _from_fields(head.fields)


def _from_markdown(file_bytes: bytes) -> DocumentMetadata:
    fields, _ = split_front_matter(decode_text(file_bytes))
    return _from_fields(fields)


def _from_fields(fields: dict[str, str]) -> DocumentMetadata:
    """Build metadata from a flat key/value map (HTML meta tags, front matter)."""
    return DocumentMetadata(
        title=_first(fields, _TITLE_KEYS),
        author=_first(fields, _AUTHOR_KEYS),
        subject=_first(fields, _SUBJECT_KEYS),
        document_date=_iso(_first(fields, _DATE_KEYS)),
    )


_EXTRACTORS = {
    ".pdf": _from_pdf,
    ".docx": _from_docx,
    ".html": _from_html,
    ".htm": _from_html,
    ".md": _from_markdown,
    ".markdown": _from_markdown,
}


class _HeadReader(HTMLParser):
    """
    Collects <title> and every <meta> into a flat key/value map.

    Stops at <body>: everything after it is content, already handled by
    rag/loaders/html.py, and a stray <meta> buried in the page shouldn't be
    able to rename the document.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fields: dict[str, str] = {}
        self._in_title = False
        self._done = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._done:
            return
        if tag == "body":
            self._done = True
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            attributes = {key.lower(): (value or "") for key, value in attrs}
            key = attributes.get("name") or attributes.get("property")
            content = attributes.get("content", "").strip()
            if key and content:
                self.fields.setdefault(key.strip().lower(), content)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and data.strip():
            self.fields.setdefault("title", " ".join(data.split()))


# --------------------------------------------------------------------------
# Cleaning and fallbacks
# --------------------------------------------------------------------------


def _resolve_title(embedded: str | None, filename: str, pages: list[PageContent] | None) -> str:
    """
    Settle on a title: what the file claims, else its first heading, else the
    filename.

    The embedded title is preferred but not trusted blindly - see
    `_looks_like_a_filename`. The filename is the last resort rather than the
    first because it is the one thing the reader can already see.
    """
    if embedded and not _looks_like_a_filename(embedded):
        return embedded

    heading = _first_heading(pages)
    if heading:
        return heading

    return Path(filename or "").stem or (embedded or "Untitled")


def _looks_like_a_filename(title: str) -> bool:
    """
    Recognise the titles that are really filenames.

    Word stamps "Microsoft Word - quarterly.doc" into /Title, LaTeX leaves the
    source name, and export tools leave "Untitled". Every one of them tells the
    reader exactly what the filename already told them, while hiding the real
    title sitting in the document's first heading.
    """
    lowered = title.strip().lower()
    return lowered.startswith(_FILENAME_TITLE_PREFIXES) or lowered.endswith(
        _FILENAME_TITLE_SUFFIXES
    )


def _first_heading(pages: list[PageContent] | None) -> str | None:
    for page in pages or []:
        for block in page.blocks:
            if block.kind == "heading" and block.text.strip():
                return " ".join(block.text.split())
    return None


def _first(fields: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _clean(fields.get(key))
        if value:
            return value
    return None


def _clean(value: str | None) -> str | None:
    """Collapse whitespace and treat blank as absent."""
    if not value:
        return None
    collapsed = " ".join(str(value).split())
    return collapsed or None


def _iso(value) -> str | None:
    """
    Normalise whatever a format hands back into an ISO 8601 date.

    One shape out, however it arrived: PDF and DOCX give real datetimes, HTML
    and Markdown give strings in whatever the author typed. A date only earns
    its place if it can be compared and sorted, which means one format.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = _clean(value)
    if not text:
        return None

    try:
        # Handles "2026-03-02", "2026-03-02T12:00:00+01:00" and similar.
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass

    for pattern in _DATE_FORMATS:
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    # An unparseable date is dropped rather than stored as free text: the whole
    # point of the field is that it can be compared.
    return None
