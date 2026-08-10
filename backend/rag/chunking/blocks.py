"""
Block-aware segmentation: the shared step every strategy runs first.

Responsibility: turn a section's blocks into segments, marking the ones that
must not be cut by a general-purpose text splitter, and splitting the ones that
are too big in the way their own format demands.

Why a table can't be split like prose
-------------------------------------
The loaders hand tables over as Markdown, which means the column names live on
the first line and nowhere else. Split that at an arbitrary character offset
and every chunk after the first reads:

    | Bergen | 4412 | 2026-03-02 |
    | Oslo   | 9931 | 2026-03-03 |

Nothing in it says which column is the city, the order count or the date. The
numbers are all still there and all meaningless - the same failure the CSV
loader already avoids at load time, reappearing at chunk time for tables that
came from a PDF or a Word file.

So a table is split at *row* boundaries, and every part repeats the header. A
fenced code block is treated the same way for the same reason: a splitter that
cuts it mid-line produces something that is neither prose nor code, and the
fence markers stop matching.
"""
from dataclasses import dataclass

from rag.layout import Block
from rag.chunking.base import ChunkBudget

# Enough rows to be worth returning. A "table" of one data row that has been
# split off from its neighbours is usually less useful than slightly
# overshooting the budget to keep two together.
MIN_TABLE_ROWS_PER_PART = 1


@dataclass
class Segment:
    """
    A piece of a section, with a note on whether it can be cut further.

    `atomic` segments are emitted as their own chunk even if that leaves the
    chunk under-full: the alternative is a table or a code block sliced in a
    way that destroys what it meant.
    """

    text: str
    atomic: bool


def segment_blocks(blocks: list[Block], budget: ChunkBudget) -> list[Segment]:
    """
    Flatten blocks into segments, splitting oversized tables by row.

    Prose blocks come back as one non-atomic segment each; the caller is free
    to join consecutive ones and split them however it likes.
    """
    segments: list[Segment] = []
    for block in blocks:
        if block.kind == "table":
            segments.extend(_table_segments(block.text, budget))
        elif _is_code_fence(block.text) and budget.measure(block.text) <= budget.size:
            segments.append(Segment(text=block.text, atomic=True))
        else:
            segments.append(Segment(text=block.text, atomic=False))
    return segments


def prose_runs(segments: list[Segment]) -> list[list[Segment] | Segment]:
    """
    Group consecutive non-atomic segments together, leaving atomic ones alone.

    Prose either side of a table is still one argument, so it should be
    chunked together with overlap; the table between them should not be dragged
    into that. This is what preserves both.
    """
    grouped: list[list[Segment] | Segment] = []
    run: list[Segment] = []
    for segment in segments:
        if segment.atomic:
            if run:
                grouped.append(run)
                run = []
            grouped.append(segment)
        else:
            run.append(segment)
    if run:
        grouped.append(run)
    return grouped


def _is_code_fence(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _table_segments(table: str, budget: ChunkBudget) -> list[Segment]:
    """
    Split a Markdown table into parts that each repeat the header.

    A table that already fits comes back untouched. One that doesn't is cut
    between rows, never inside one, and each part is a valid standalone table -
    so a chunk retrieved from the middle of a 500-row export still knows what
    its columns are.
    """
    if budget.measure(table) <= budget.size:
        return [Segment(text=table, atomic=True)]

    lines = table.split("\n")
    header = lines[:2] if len(lines) > 1 and _is_divider(lines[1]) else lines[:1]
    rows = lines[len(header) :]
    if not rows:
        return [Segment(text=table, atomic=True)]

    preamble = "\n".join(header)
    parts: list[Segment] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join([preamble, *current, row])
        if current and budget.measure(candidate) > budget.size:
            parts.append(Segment(text="\n".join([preamble, *current]), atomic=True))
            current = []
        current.append(row)

    if current:
        parts.append(Segment(text="\n".join([preamble, *current]), atomic=True))
    return parts


def _is_divider(line: str) -> bool:
    """The |---|---| line Markdown puts under a table's header row."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("|-: ")
