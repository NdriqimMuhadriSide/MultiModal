"""
Text splitter.

Responsibility: split a block of text into overlapping chunks suitable for
embedding and retrieval. Pure text in, text out - no PDF knowledge, no
metadata beyond a chunk's position within the text it was given
(chunk_index). Filename/page number are attached by the caller
(app/services/document_service.py), since this module doesn't know where
the text came from.

Implemented by hand (rather than pulling in LangChain's
RecursiveCharacterTextSplitter) so the chunking logic is fully visible and
easy to reason about for learning purposes. The algorithm is the same idea:

    1. Normalise whitespace *without* flattening the document's structure.
    2. Break the text on the most structural separator that produces pieces
       small enough to work with - paragraph, then line, then sentence, then
       word, and only as a last resort a blind character cut.
    3. Pack consecutive pieces into chunks, repeating the tail of the previous
       chunk at the start of the next one so an idea that lands on a boundary
       still appears intact somewhere.

Why the structure matters: each chunk is embedded as a single vector, and an
embedding is only a useful summary if the chunk is a coherent unit. A chunk
that ends mid-sentence and a chunk that starts mid-sentence produce two
vectors that represent neither idea, so the retriever can't match either.

How a chunk is measured
-----------------------
`chunk_size` counts whatever `measure` counts. The default is `len`, so by
default it counts characters and this module has no idea an embedding model
exists. Ingestion passes the model's own token counter instead - see
app/services/document_service.py - because characters are the wrong unit for
the thing that actually breaks:

    an embedding model has a hard token limit, and crossing it is silent.

MiniLM-L6 reads 256 tokens and quietly drops the rest, returning a vector that
describes only the beginning of the chunk. Characters do not predict tokens
well enough to stay under that: the same 800 characters are ~180 tokens of
English prose but ~340 tokens of Markdown table rows, because every pipe,
digit and date fragment becomes a token of its own. Sizing in characters means
prose chunks are needlessly small *and* dense ones are silently truncated.

Keeping the unit behind a function rather than importing a tokenizer is what
lets this module stay pure - text in, text out, no model to load in a test -
while still being exact about the limit that matters.
"""
import re
from collections.abc import Callable
from dataclasses import dataclass

from rag.sentences import split_sentences

# Defaults are in characters, matching the default `measure` of `len`.
# Ingestion overrides both with token counts (see app/core/config.py).
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def _by_literal(separator: str) -> Callable[[str], list[str]]:
    """
    A splitter that cuts on a literal string, keeping the separator attached to
    the piece it followed - so joining the pieces reproduces the input exactly.
    """

    def split(text: str) -> list[str]:
        parts = text.split(separator)
        return [part + separator for part in parts[:-1]] + [parts[-1]]

    return split


# Splitters in descending order of how much structure they preserve. The
# splitter walks this list and stops at the first one that gets a run of text
# down to size. `None` is the fallback: no structure left to exploit, cut blind.
#
# The sentence level is rag/sentences.py rather than a split on ". " - a period
# is not a sentence boundary most of the time ("Dr. Smith weighed 3.5 kg of
# sample no. 4"), and cutting on every one of them produces fragments that mean
# nothing on their own and embed to nothing useful.
SEPARATORS: list[Callable[[str], list[str]] | None] = [
    _by_literal("\n\n"),
    _by_literal("\n"),
    split_sentences,
    _by_literal(" "),
    None,
]

_LINE_ENDINGS = re.compile(r"\r\n?")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")  # whitespace runs, but not newlines
_SPACE_HUGGING_NEWLINE = re.compile(r" *\n *")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")
_ANY_WHITESPACE = re.compile(r"\s")


@dataclass
class TextChunk:
    chunk_index: int  # 0-indexed position of this chunk within the input text
    text: str


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    measure: Callable[[str], int] = len,
) -> list[TextChunk]:
    """
    Split `text` into overlapping chunks of at most `chunk_size` units,
    preferring boundaries that mean something to a reader.

    A "unit" is whatever `measure` counts - characters by default, tokens when
    ingestion passes the embedding model's counter. No chunk exceeds
    `chunk_size` under that measure; that is the guarantee the caller is
    relying on to stay inside the model's context window.

    Up to `chunk_overlap` units from the end of one chunk are repeated at the
    start of the next, so a sentence or idea that happens to fall on a chunk
    boundary still appears intact in at least one chunk.

    Raises:
        ValueError: if chunk_size/chunk_overlap are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    normalized = _normalize(text)
    if not normalized:
        return []

    # Each chunk is "tail of the previous chunk" + "fresh text", so the fresh
    # part gets the remaining budget. Keeping pieces within `step` is what
    # guarantees no chunk ever exceeds chunk_size.
    step = chunk_size - chunk_overlap

    pieces = _split_on_separators(normalized, step, SEPARATORS, measure)

    chunks: list[TextChunk] = []
    carry = ""  # trailing text of everything emitted so far, capped at chunk_overlap
    for window in _pack(pieces, step, measure):
        prefix = _overlap_prefix(carry, chunk_overlap, measure)
        chunk_text = _fit(prefix, window, chunk_size, measure)
        if chunk_overlap:
            carry = _longest_suffix_within(carry + window, chunk_overlap, measure)
        if chunk_text:
            chunks.append(TextChunk(chunk_index=len(chunks), text=chunk_text))

    return chunks


def _fit(prefix: str, window: str, chunk_size: int, measure) -> str:
    """
    Join the repeated tail and the fresh text, shrinking the tail if the pair
    overflows.

    Packing budgets each part separately and adds the numbers up, which is
    exact for characters but only approximate for tokens: a tokenizer can merge
    across the seam, so two parts that each fit can very occasionally overflow
    together. Rather than let that through, the *overlap* is trimmed - it is by
    definition text that already appears in full in the previous chunk, so
    dropping some of it costs nothing, whereas trimming the window would lose
    the only copy of that text.
    """
    if measure(prefix + window) <= chunk_size:
        return (prefix + window).strip()

    room = chunk_size - measure(window)
    trimmed = _longest_suffix_within(prefix, room, measure) if room > 0 else ""
    return (trimmed + window).strip()


def _normalize(text: str) -> str:
    """
    Tidy up extracted text while keeping its line and paragraph boundaries.

    PDF extraction is messy - stray tabs, runs of spaces used for layout,
    trailing spaces before every line break. All of that is noise. Newlines
    are not: a blank line is a paragraph break and a single newline is a list
    item, table row, or heading. Those are exactly the boundaries we want to
    cut on later, so they survive normalisation.
    """
    text = _LINE_ENDINGS.sub("\n", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = _SPACE_HUGGING_NEWLINE.sub("\n", text)
    text = _BLANK_LINE_RUN.sub("\n\n", text)
    return text.strip()


def _split_on_separators(text: str, limit: int, separators: list, measure) -> list[str]:
    """
    Break `text` into pieces of at most `limit` units, using the first
    splitter in `separators` that does the job and falling back to the next
    one for any piece that is still too long.

    Every splitter is lossless - joining its output reproduces its input - so
    joining every returned piece reproduces `text` exactly. That keeps chunk
    assembly a plain string concatenation and keeps the overlap arithmetic
    honest.
    """
    if measure(text) <= limit:
        return [text] if text else []

    splitter, remaining = separators[0], separators[1:]

    if splitter is None:
        # Nothing structural left (e.g. one enormous unbroken run) - cut blind,
        # a measured mouthful at a time.
        pieces = []
        while text:
            head = _longest_prefix_within(text, limit, measure)
            if not head:
                # A single character that the measure already calls oversized.
                # Emitting it keeps the loop finite; the caller's final fit
                # check is what still bounds the assembled chunk.
                head = text[0]
            pieces.append(head)
            text = text[len(head) :]
        return pieces

    pieces = splitter(text)
    if len(pieces) <= 1:
        # This splitter found nothing to cut on; go straight to the next.
        return _split_on_separators(text, limit, remaining, measure)

    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if measure(piece) <= limit:
            out.append(piece)
        else:
            out.extend(_split_on_separators(piece, limit, remaining, measure))
    return out


def _pack(pieces: list[str], budget: int, measure) -> list[str]:
    """
    Greedily glue consecutive pieces together while they fit in `budget`.

    Splitting alone would leave us with lots of tiny chunks (one per sentence);
    packing puts as much related text as possible into each chunk while never
    crossing a boundary chosen by _split_on_separators.
    """
    windows: list[str] = []
    current = ""

    for piece in pieces:
        # Measures the joined text rather than summing the pieces' measures.
        # Adding them up is exact for characters but wrong for tokens: a
        # tokenizer counts two special tokens per string it is handed, so
        # summing ten pieces over-counts by eighteen and leaves that much of
        # the model's window unused on every chunk.
        if current and measure(current + piece) > budget:
            windows.append(current)
            current = ""
        current += piece

    if current:
        windows.append(current)
    return windows


def _overlap_prefix(carry: str, chunk_overlap: int, measure) -> str:
    """
    Pick the text to repeat at the start of the next chunk.

    Snapped forward to a whitespace boundary so the repeat starts on a whole
    word. If the tail contains no whitespace at all (one very long run) the
    raw tail is used - a mid-word repeat beats no overlap.
    """
    if chunk_overlap <= 0 or not carry:
        return ""

    tail = _longest_suffix_within(carry, chunk_overlap, measure)
    boundary = _ANY_WHITESPACE.search(tail)
    if boundary:
        snapped = tail[boundary.end() :]
        if snapped:
            return snapped
    return tail


def _longest_prefix_within(text: str, budget: int, measure) -> str:
    """
    The longest opening slice of `text` that fits in `budget`.

    Found by binary search on the character index rather than by slicing
    directly, because only the caller's `measure` knows what a unit is - a
    token boundary is nowhere the string index can point to. Costs O(log n)
    measurements instead of one, which is cheap next to embedding the result.

    Assumes lengthening a string never shortens its measure. True for
    characters, and true for tokenizers in every case that matters; the
    assembly-time check in `_fit` is what makes the guarantee absolute.
    """
    if budget <= 0 or not text:
        return ""
    if measure(text) <= budget:
        return text

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if measure(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _longest_suffix_within(text: str, budget: int, measure) -> str:
    """The longest closing slice of `text` that fits in `budget`."""
    if budget <= 0 or not text:
        return ""
    if measure(text) <= budget:
        return text

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if measure(text[-middle:]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[len(text) - low :]
