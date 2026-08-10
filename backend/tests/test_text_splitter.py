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


def test_split_text_prefers_paragraph_boundaries():
    first = "Alpha " * 10  # 60 chars
    second = "Beta " * 10  # 50 chars
    chunks = split_text(f"{first.strip()}\n\n{second.strip()}", chunk_size=80, chunk_overlap=10)

    # Both paragraphs fit individually but not together, so the split should
    # land on the blank line rather than at character 80.
    assert len(chunks) == 2
    assert chunks[0].text.startswith("Alpha")
    assert chunks[0].text.endswith("Alpha")
    assert chunks[1].text.endswith("Beta")


def test_split_text_falls_back_to_sentence_then_word_boundaries():
    sentences = " ".join(f"Sentence number {n} here." for n in range(1, 13))
    chunks = split_text(sentences, chunk_size=90, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        # No chunk should begin or end halfway through a word.
        assert chunk.text.split(" ")[0] in sentences.split(" ")
        assert chunk.text.endswith(".")


def test_split_text_never_exceeds_chunk_size():
    text = "\n\n".join("word " * 40 for _ in range(6))
    chunks = split_text(text, chunk_size=120, chunk_overlap=30)

    assert chunks
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_split_text_keeps_paragraph_breaks_but_collapses_layout_whitespace():
    chunks = split_text("First\t\tparagraph   line\n\n\n\nSecond paragraph", chunk_size=1000)

    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph line\n\nSecond paragraph"


def test_split_text_preserves_every_word_of_the_source():
    text = "\n\n".join(f"Paragraph {n} with several distinct words {n}." for n in range(1, 15))
    chunks = split_text(text, chunk_size=100, chunk_overlap=25)

    joined = " ".join(chunk.text for chunk in chunks)
    for word in text.split():
        assert word in joined


# --- Sizing by an arbitrary measure -----------------------------------------
#
# The splitter takes a `measure` so ingestion can size chunks in the embedding
# model's tokens rather than characters. These use cheap stand-in measures
# instead of a real tokenizer, so they stay fast and state the property
# exactly.


def _words(text: str) -> int:
    return len(text.split())


def _with_overhead(text: str) -> int:
    """
    A measure that charges a fixed cost per string, the way a tokenizer charges
    two special tokens.

    This is what makes measures non-additive: measure(a) + measure(b) is larger
    than measure(a + b), so anything that adds up piece sizes over-counts.
    """
    return len(text) + 2


def test_chunks_are_sized_by_the_given_measure():
    text = " ".join(f"word{index}" for index in range(60))

    chunks = split_text(text, chunk_size=10, chunk_overlap=2, measure=_words)

    assert len(chunks) > 1
    assert all(_words(chunk.text) <= 10 for chunk in chunks)


def test_the_default_measure_is_characters():
    text = "a" * 500

    by_default = split_text(text, chunk_size=100, chunk_overlap=20)
    explicitly = split_text(text, chunk_size=100, chunk_overlap=20, measure=len)

    assert [chunk.text for chunk in by_default] == [chunk.text for chunk in explicitly]


def test_no_chunk_exceeds_the_budget_under_a_non_additive_measure():
    """
    The guarantee the caller relies on to stay inside the model's context
    window - and the case where naive arithmetic breaks it.
    """
    text = ". ".join(f"Sentence number {index} with some filler words" for index in range(40))

    chunks = split_text(text, chunk_size=60, chunk_overlap=12, measure=_with_overhead)

    assert len(chunks) > 1
    assert all(_with_overhead(chunk.text) <= 60 for chunk in chunks)


def test_an_unbreakable_run_is_still_cut_to_fit():
    # No separator anywhere, so the blind-cut fallback has to respect the
    # measure rather than slicing by character index.
    chunks = split_text("x" * 300, chunk_size=40, chunk_overlap=8, measure=_with_overhead)

    assert all(_with_overhead(chunk.text) <= 40 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks).count("x") >= 300


def test_no_text_is_lost_when_a_chunk_has_to_be_trimmed():
    """
    When the overlap and the fresh text overflow together, the *overlap* is
    what gets shortened - it already appears in full in the previous chunk,
    whereas the fresh text exists nowhere else.
    """
    sentences = [f"Fact {index} about the subject" for index in range(30)]
    text = ". ".join(sentences)

    chunks = split_text(text, chunk_size=50, chunk_overlap=15, measure=_with_overhead)
    rejoined = " ".join(chunk.text for chunk in chunks)

    for index in range(30):
        assert f"Fact {index} about the subject" in rejoined


def test_overlap_is_measured_too():
    text = " ".join(f"word{index}" for index in range(60))

    chunks = split_text(text, chunk_size=12, chunk_overlap=4, measure=_words)

    # Consecutive chunks share words, and never more than the overlap allows.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    shared = [word for word in second_words if word in first_words]
    assert 0 < len(shared) <= 4
