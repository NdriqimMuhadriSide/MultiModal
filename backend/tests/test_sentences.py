"""
Sentence segmentation tests.

Almost all of these are about periods that are *not* sentence boundaries. That
is the whole difficulty: splitting on ". " is right often enough to look like
it works and wrong often enough to shred real documents.
"""
from rag.sentences import split_sentences, split_sentences_stripped


def test_plain_sentences_are_split():
    assert split_sentences_stripped("First one. Second one. Third one.") == [
        "First one.",
        "Second one.",
        "Third one.",
    ]


def test_question_and_exclamation_marks_end_sentences():
    assert split_sentences_stripped("Really? Yes! Fine.") == ["Really?", "Yes!", "Fine."]


def test_titles_do_not_end_a_sentence():
    assert split_sentences_stripped("Dr. Smith arrived. He waited.") == [
        "Dr. Smith arrived.",
        "He waited.",
    ]


def test_decimals_do_not_end_a_sentence():
    assert split_sentences_stripped("We weighed 3.5 kg today. Then we stopped.") == [
        "We weighed 3.5 kg today.",
        "Then we stopped.",
    ]


def test_latin_abbreviations_do_not_end_a_sentence():
    assert split_sentences_stripped("Use a solvent, e.g. acetone. Then rinse.") == [
        "Use a solvent, e.g. acetone.",
        "Then rinse.",
    ]


def test_reference_abbreviations_do_not_end_a_sentence():
    assert split_sentences_stripped("See Fig. 2 and sample no. 4. Results follow.") == [
        "See Fig. 2 and sample no. 4.",
        "Results follow.",
    ]


def test_initials_do_not_end_a_sentence():
    assert split_sentences_stripped("We saw J. R. R. Tolkien. He waved.") == [
        "We saw J. R. R. Tolkien.",
        "He waved.",
    ]


def test_an_ellipsis_mid_sentence_is_not_a_boundary():
    assert split_sentences_stripped("Absolutely... maybe not. The end.") == [
        "Absolutely... maybe not.",
        "The end.",
    ]


def test_a_number_at_the_end_of_a_sentence_still_ends_it():
    """The digit rule must not swallow "...costs 20. Next" into one sentence."""
    assert split_sentences_stripped("The sample weighed 20. Then it was discarded.") == [
        "The sample weighed 20.",
        "Then it was discarded.",
    ]


def test_the_hard_case_all_at_once():
    text = "Dr. Smith weighed 3.5 kg of sample no. 4 (see Fig. 2). Then he left."

    assert len(split_sentences_stripped(text)) == 2


def test_text_with_no_final_punctuation_is_one_sentence():
    assert split_sentences_stripped("No full stop here") == ["No full stop here"]


def test_empty_input_gives_no_sentences():
    assert split_sentences("") == []


def test_splitting_is_lossless():
    """
    Callers reassemble sentences into chunks, so a splitter that ate the
    whitespace between them would quietly reflow the document.
    """
    text = "Dr. Smith weighed 3.5 kg.  Then he left!  Really?\n\nYes."

    assert "".join(split_sentences(text)) == text


def test_the_recursive_splitter_uses_real_sentence_boundaries():
    """
    Pins the fix in place: the splitter's sentence level is this module, not a
    split on ". ", so a chunk boundary never lands inside "Dr. Smith".
    """
    from rag.text_splitter import split_text

    text = "Dr. Smith weighed 3.5 kg of sample no. 4. " * 6

    chunks = split_text(text, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert not chunk.text.endswith("Dr.")
        assert not chunk.text.endswith("no.")
