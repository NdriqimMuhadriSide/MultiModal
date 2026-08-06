"""
Text splitter.

Responsibility: split a block of text into overlapping, fixed-size chunks
suitable for embedding and retrieval. Pure text in, text out - no PDF
knowledge, no metadata beyond a chunk's position within the text it was
given (chunk_index). Filename/page number are attached by the caller
(app/services/document_service.py), since this module doesn't know where
the text came from.

Implemented as a plain sliding-window splitter (rather than pulling in
LangChain's RecursiveCharacterTextSplitter) so the chunking logic is fully
visible and easy to reason about for learning purposes. See the size/
overlap discussion in the docstrings below.
"""
from dataclasses import dataclass

# Character-based, not token-based, for simplicity. As a rough rule of
# thumb 1 token ~= 4 characters for English text, so 1000 chars ~= 250
# tokens - comfortably inside any embedding model's context window.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass
class TextChunk:
    chunk_index: int  # 0-indexed position of this chunk within the input text
    text: str


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split `text` into overlapping chunks of at most `chunk_size` characters.

    `chunk_overlap` characters from the end of one chunk are repeated at the
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

    # Collapse all whitespace (newlines, tabs, repeated spaces) down to single
    # spaces. PDF text extraction is often messy - this keeps chunk boundaries
    # meaningful instead of splitting mid-word around stray line breaks.
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    length = len(normalized)
    start = 0
    chunk_index = 0

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(TextChunk(chunk_index=chunk_index, text=normalized[start:end]))

        if end == length:
            break

        start += step
        chunk_index += 1

    return chunks
