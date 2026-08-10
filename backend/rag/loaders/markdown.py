"""
Markdown loader (also used for plain text).

Responsibility: turn Markdown source into blocks. After the PDF work this one
is almost restful: a PDF hides its structure in glyph positions and has to be
reverse-engineered, whereas Markdown *is* structure. "## Methods" is a level-2
heading because it says so, and rag/structure.py takes that as given rather
than guessing from font sizes it doesn't have.

Only block-level structure is parsed - headings, tables, code fences,
paragraphs. Inline syntax (emphasis, links, inline code) is left exactly as
written: it is already readable text, an LLM reads Markdown natively, and
stripping it would only risk mangling content for no gain in retrieval.

Plain .txt files come here too. They have no Markdown in them, which is fine -
every rule below is opt-in, so a .txt file falls through to "blank lines
separate paragraphs", which is the correct reading of a .txt file anyway.
"""
import re

from rag.layout import Block
from rag.loaders.base import PageContent, decode_text, single_page

# "### Heading" - one to six hashes, then the text. The closing hashes some
# authors add ("### Heading ###") are decoration and get stripped.
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# The underlined form: text on one line, then a rule of = or - beneath it.
_SETEXT_UNDERLINE = re.compile(r"^(=+|-{2,})\s*$")
# ``` or ~~~, optionally with a language after it.
_FENCE = re.compile(r"^(```+|~~~+)")
# A table row - starts and ends with a pipe once trimmed.
_TABLE_ROW = re.compile(r"^\|.*\|$")
# The |---|---| line under a Markdown table's header.
_TABLE_DIVIDER = re.compile(r"^\|[\s:|-]+\|$")


def load_markdown(file_bytes: bytes) -> list[PageContent]:
    """Parse Markdown (or plain text) into a single page of blocks."""
    _, body = split_front_matter(decode_text(file_bytes))
    return single_page(parse_markdown(body))


def split_front_matter(source: str) -> tuple[dict[str, str], str]:
    """
    Separate a YAML front-matter block from the document body.

    Front matter is the `---` fenced key/value block many Markdown documents
    open with, and it is where a Markdown file keeps its title, author and
    date - so rag/metadata.py reads it. It has to be *removed* here either
    way: left in, it would be ingested as an ordinary paragraph, and "layout:
    post" is not content anybody wants back from a search.

    Parsed as flat `key: value` lines rather than with a YAML library. Real
    YAML supports nesting, anchors and multi-line scalars, none of which
    appear in the handful of fields worth reading, and none of which justify a
    dependency. Anything more elaborate is skipped rather than guessed at.
    """
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, source

    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            fields = {}
            for line in lines[1:index]:
                key, separator, value = line.partition(":")
                if separator and key.strip() and not key.startswith((" ", "\t", "-")):
                    fields[key.strip().lower()] = value.strip().strip("\"'")
            return fields, "\n".join(lines[index + 1 :])

    # An opening fence with no closing one isn't front matter, it's a
    # horizontal rule at the top of the document. Leave the text alone.
    return {}, source


def parse_markdown(source: str) -> list[Block]:
    """
    Walk the source line by line, emitting a block at every boundary.

    Public so it can be tested directly on a string, and reused by any future
    format that lowers to Markdown.
    """
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Block(kind="text", text="\n".join(paragraph).strip()))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        fence = _FENCE.match(stripped)
        if fence:
            # A fenced block is passed through untouched. This is not just
            # tidiness: a Python comment inside a code sample starts with "#",
            # and parsing it as Markdown would turn it into a section heading
            # and hang the rest of the document under it.
            flush()
            index = _consume_fence(lines, index, fence.group(1)[0], blocks)
            continue

        if not stripped:
            flush()
            index += 1
            continue

        heading = _ATX_HEADING.match(stripped)
        if heading:
            flush()
            blocks.append(
                Block(kind="heading", text=heading.group(2).strip(), level=len(heading.group(1)))
            )
            index += 1
            continue

        if _is_setext_heading(lines, index):
            flush()
            blocks.append(
                Block(
                    kind="heading",
                    text=stripped,
                    level=1 if lines[index + 1].strip().startswith("=") else 2,
                )
            )
            index += 2
            continue

        if _TABLE_ROW.match(stripped):
            flush()
            index = _consume_table(lines, index, blocks)
            continue

        paragraph.append(line.rstrip())
        index += 1

    flush()
    return blocks


def _consume_fence(lines: list[str], start: int, marker: str, blocks: list[Block]) -> int:
    """Collect a fenced code block verbatim, including its fences."""
    body = [lines[start]]
    index = start + 1
    while index < len(lines):
        body.append(lines[index])
        if lines[index].strip().startswith(marker * 3):
            index += 1
            break
        index += 1
    blocks.append(Block(kind="text", text="\n".join(body)))
    return index


def _consume_table(lines: list[str], start: int, blocks: list[Block]) -> int:
    """
    Collect consecutive pipe rows as one table block.

    Kept in the source's own notation rather than re-rendered: it is already
    the Markdown table every other loader is converting *into*. The alignment
    divider is dropped and re-added by nothing - it stays, because a reader
    (and a model) expects it between header and body.
    """
    body: list[str] = []
    index = start
    while index < len(lines) and _TABLE_ROW.match(lines[index].strip()):
        body.append(lines[index].strip())
        index += 1

    # A single pipe line with no divider under it is more likely a stray line
    # of prose than a table; treat it as text rather than inventing a header.
    kind = "table" if len(body) > 1 and _TABLE_DIVIDER.match(body[1]) else "text"
    blocks.append(Block(kind=kind, text="\n".join(body)))
    return index


def _is_setext_heading(lines: list[str], index: int) -> bool:
    """
    An underlined heading: "Overview" followed by "========".

    Requires the underlined line to be ordinary text - a list item or another
    heading followed by dashes is a list item followed by a horizontal rule,
    not a heading.
    """
    if index + 1 >= len(lines) or not _SETEXT_UNDERLINE.match(lines[index + 1].strip()):
        return False
    text = lines[index].strip()
    return bool(text) and not text.startswith(("#", "-", "*", "+", ">", "|"))
