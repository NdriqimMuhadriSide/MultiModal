import pytest

from rag.text_splitter import split_text


def test_split_text_returns_empty_list_for_blank_text():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_split_text_single_chunk_when_shorter_than_chunk_size():
    chunks = split_text("hello world", chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "hello world"


def test_split_text_produces_overlapping_chunks():
    # Distinct characters (not all "A") so overlap can be verified precisely.
    text = "".join(chr(ord("a") + (i % 26)) for i in range(25))
    chunks = split_text(text, chunk_size=10, chunk_overlap=3)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Every chunk after the first should start with the overlap from the
    # tail of the previous chunk.
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text[-3:] == current.text[:3]
    # The last chunk should reach the end of the original text.
    assert chunks[-1].text.endswith(text[-1])


def test_split_text_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=10, chunk_overlap=10)
