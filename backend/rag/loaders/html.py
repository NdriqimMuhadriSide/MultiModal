"""
HTML loader.

Responsibility: turn an HTML document into blocks. Like Markdown, HTML states
its own structure - <h2> is a level-2 heading - so there is nothing to infer.
The work is in ignoring almost all of it: a saved web page is mostly
navigation, scripts, styling and layout scaffolding, and none of that is
content anybody will ever search for.

Built on the standard library's HTMLParser rather than BeautifulSoup or lxml.
That's not only about avoiding a dependency: HTMLParser is a *stream* of
tokens, and a stream is the right shape here. Extracting blocks means walking
the document once in order, flushing the current block whenever a tag says one
ended - a tree would be built only to be walked linearly anyway. It is also
tolerant of the unclosed tags and stray markup that real pages are full of,
where a strict XML parser would refuse the file outright.
"""
from html.parser import HTMLParser

from rag.layout import Block, rows_to_markdown
from rag.loaders.base import PageContent, decode_text, single_page

_HEADINGS = {f"h{level}": level for level in range(1, 7)}

# Elements whose *contents* are not content: code that runs, rules that style,
# and the document head. Everything between the open and close tag is dropped.
_SKIPPED = {"script", "style", "head", "noscript", "template", "svg"}

# Elements that end whatever text was being collected. Not an exhaustive list
# of block-level tags - just the ones that reliably mean "new paragraph" -
# because wrongly splitting a sentence costs more than merging two.
_BOUNDARIES = {
    "p", "div", "section", "article", "header", "footer", "main", "aside",
    "li", "dd", "dt", "blockquote", "pre", "figcaption", "hr", "br",
}


def load_html(file_bytes: bytes) -> list[PageContent]:
    """Parse an HTML document into a single page of blocks."""
    return single_page(parse_html(decode_text(file_bytes)))


def parse_html(source: str) -> list[Block]:
    """Public so it can be tested directly on a string."""
    extractor = _BlockExtractor()
    extractor.feed(source)
    extractor.close()
    return extractor.blocks


class _BlockExtractor(HTMLParser):
    """
    Walks the token stream once, emitting a block per structural boundary.

    Three pieces of state, each earning its place:

      * `_skip_depth` counts how deep we are inside a <script>/<style>/<head>.
        A counter rather than a flag because these nest (an <svg> inside an
        <svg>), and a flag would be cleared by the first closing tag.
      * `_heading_level` is set while inside <h1>..<h6>, so the text collected
        is emitted as a heading rather than a paragraph.
      * `_table` accumulates rows while inside a <table>, since a table's
        meaning lives in its shape and can only be rendered once complete.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._text: list[str] = []
        self._skip_depth = 0
        self._heading_level: int | None = None
        self._table: list[list[str]] | None = None
        self._cell: list[str] | None = None

    # --- tags -------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._skip_depth or tag in _SKIPPED:
            self._skip_depth += tag in _SKIPPED
            return

        if tag == "table":
            self._flush()
            # Nested tables are flattened into the outer one rather than
            # tracked separately: they're rare, almost always layout rather
            # than data, and the alternative is a stack for no real gain.
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._table.append([])
        elif tag in ("td", "th") and self._table is not None:
            self._flush_cell()
            self._cell = []
        elif tag in _HEADINGS:
            self._flush()
            self._heading_level = _HEADINGS[tag]
        elif tag in _BOUNDARIES:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return

        if tag == "table" and self._table is not None:
            self._flush_cell()
            rows = [row for row in self._table if any(cell.strip() for cell in row)]
            self._table = None
            if rows:
                self.blocks.append(Block(kind="table", text=rows_to_markdown(rows)))
        elif tag in ("td", "th"):
            self._flush_cell()
        elif tag in _HEADINGS or tag in _BOUNDARIES:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._cell is not None:
            self._cell.append(data)
        else:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        # A page that never closes its last tag - common enough - still owes
        # us whatever it had collected.
        self._flush_cell()
        if self._table:
            self.blocks.append(Block(kind="table", text=rows_to_markdown(self._table)))
            self._table = None
        self._flush()

    # --- emitting ---------------------------------------------------------

    def _flush(self) -> None:
        text = _collapse("".join(self._text))
        self._text.clear()
        level, self._heading_level = self._heading_level, None
        if not text:
            return
        if level is not None:
            self.blocks.append(Block(kind="heading", text=text, level=level))
        else:
            self.blocks.append(Block(kind="text", text=text))

    def _flush_cell(self) -> None:
        if self._cell is None:
            return
        cell, self._cell = _collapse("".join(self._cell)), None
        if self._table is not None:
            if not self._table:
                # A <td> outside any <tr> - malformed, but real. Give it a row
                # rather than dropping the text.
                self._table.append([])
            self._table[-1].append(cell)


def _collapse(text: str) -> str:
    """
    Squeeze HTML whitespace down to single spaces.

    In HTML, newlines and indentation are formatting of the *source*, not of
    the content - a paragraph split across six indented lines in the file is
    one line on screen. Keeping them would put the file's indentation into the
    embedded text.
    """
    return " ".join(text.split())
