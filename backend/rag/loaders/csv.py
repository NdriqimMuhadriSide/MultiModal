"""
CSV loader.

Responsibility: turn a delimited data file into blocks. A CSV is nothing but a
table, so this is the one format where the *whole* document is the case the
PDF work had to detect - and where the interesting problem isn't parsing at
all, it's chunking.

Why a CSV can't be one big table
--------------------------------
A ten-thousand-row export rendered as a single Markdown table becomes one
enormous block, and rag/text_splitter.py then cuts it into chunks at whatever
row happens to fall on a boundary. Every chunk after the first loses the header
row, so a chunk reads:

    | Bergen | 4412 | 2026-03-02 |

...and nothing in it says which column is the city, the order id, or the date.
The numbers are all still there and all meaningless.

So rows are emitted in groups, each group a self-contained table that repeats
the header. Every chunk then carries its own column names, whatever row it
starts at. The group size is chosen to fit comfortably inside one chunk rather
than to be tidy: a group that gets split still loses headers, so the point is
for it not to be split.
"""
import csv
import io

from rag.layout import Block, rows_to_markdown
from rag.loaders.base import PageContent, decode_text, single_page

# Rows per emitted table. Small enough that a group plus its header fits well
# inside the default 800-character chunk for typical data, which is what keeps
# the header attached to the rows it describes.
ROWS_PER_GROUP = 20

# Delimiters worth guessing between. Restricted rather than left to Sniffer's
# defaults, which will happily decide a column of prose is space-delimited.
_CANDIDATE_DELIMITERS = ",;\t|"

# How much of the file to hand the delimiter sniffer. A couple of rows is
# enough to spot the pattern, and reading the whole of a large export to make
# a one-character decision is waste.
_SNIFF_BYTES = 4096


def load_csv(file_bytes: bytes) -> list[PageContent]:
    """Parse a delimited file into a single page of grouped table blocks."""
    return single_page(parse_csv(decode_text(file_bytes)))


def parse_csv(source: str) -> list[Block]:
    """Public so it can be tested directly on a string."""
    rows = _read_rows(source)
    if not rows:
        return []

    header, body = rows[0], rows[1:]
    if not body:
        # Header only: still worth ingesting - "what columns does this file
        # have" is a reasonable question to ask of it.
        return [Block(kind="table", text=rows_to_markdown([header]))]

    return [
        Block(kind="table", text=rows_to_markdown([header, *body[start : start + ROWS_PER_GROUP]]))
        for start in range(0, len(body), ROWS_PER_GROUP)
    ]


def _read_rows(source: str) -> list[list[str]]:
    """
    Parse the text into rows, guessing the delimiter.

    Guessed rather than assumed because "CSV" routinely means semicolons (the
    default in locales where the comma is a decimal separator) or tabs. Falling
    back to a comma when the guess fails is safe: a wrong delimiter yields
    one-column rows, which still ingest as text rather than failing.
    """
    try:
        dialect = csv.Sniffer().sniff(source[:_SNIFF_BYTES], delimiters=_CANDIDATE_DELIMITERS)
    except csv.Error:
        dialect = csv.excel

    rows = [row for row in csv.reader(io.StringIO(source), dialect) if any(field.strip() for field in row)]
    # Newlines inside a quoted field are legal CSV and would break the
    # one-row-per-line shape a Markdown table needs.
    return [[" ".join(field.split()) for field in row] for row in rows]
