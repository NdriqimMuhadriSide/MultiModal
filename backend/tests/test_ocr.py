"""
OCR tests.

Most of these never touch Tesseract. The engine's job - turning pixels into
words - is not ours to test; what *is* ours is the step after it: painting
word boxes back onto a character grid so rag/layout.py can read them. That
takes plain data, so it can be pinned down exactly, with synthetic boxes
standing in for whatever a real scan happens to produce.

The handful of tests that do run the engine are skipped when it isn't
installed, since OCR is an optional capability.
"""
import pytest

from rag import ocr
from rag.layout import analyze_page
from rag.ocr import Word, words_to_grid

requires_tesseract = pytest.mark.skipif(
    not ocr.is_available(), reason="Tesseract is not installed"
)

# One character is 10px wide and rows sit 40px apart in these fixtures, which
# makes every expected column and row below arithmetic rather than guesswork.
CHAR = 10
ROW = 40


def _word(text: str, column: int, row: int) -> Word:
    return Word(text=text, left=column * CHAR, top=row * ROW, width=len(text) * CHAR, height=20)


# --- Building the grid ------------------------------------------------------


def test_words_on_one_line_become_one_row():
    grid = words_to_grid([_word("hello", 0, 0), _word("world", 6, 0)])

    assert grid == "hello world"


def test_horizontal_position_becomes_column_position():
    grid = words_to_grid([_word("left", 0, 0), _word("right", 20, 0)])

    assert grid == "left" + " " * 16 + "right"


def test_stacked_words_become_stacked_rows():
    grid = words_to_grid([_word("first", 0, 0), _word("second", 0, 1)])

    assert grid == "first\nsecond"


def test_vertical_gaps_become_blank_rows():
    # Rows 0 and 1 are adjacent; row 4 is three line-heights further down, so
    # the gap has to survive as blank rows - it's what lets the segmenter tell
    # a footer apart from the body above it.
    grid = words_to_grid([_word("body", 0, 0), _word("more", 0, 1), _word("footer", 0, 4)])

    assert grid == "body\nmore\n\n\nfooter"


def test_words_at_the_same_height_share_a_row_across_columns():
    """
    The property the whole column-detection step rests on.

    Tesseract numbers lines per detected block, so a two-column page gives the
    left and right column's first lines different line numbers. Grouping by
    those numbers would stack them; grouping by height puts them side by side,
    which is where they actually are.
    """
    grid = words_to_grid(
        [
            _word("left", 0, 0),
            _word("right", 30, 0),
            _word("left", 0, 1),
            _word("right", 30, 1),
        ]
    )

    rows = grid.split("\n")
    assert len(rows) == 2
    assert all(row.startswith("left") and row.endswith("right") for row in rows)


def test_slightly_skewed_rows_are_still_one_row():
    # A scan is never perfectly straight; boxes a few pixels apart vertically
    # are the same line of text.
    skewed = Word(text="right", left=300, top=5, width=50, height=20)

    grid = words_to_grid([_word("left", 0, 0), skewed])

    assert grid.count("\n") == 0


def test_words_are_never_run_together():
    # Boxes that round to the same column must not produce "onetwo": the
    # merged token would match nothing at retrieval time.
    crowded = [
        Word(text="one", left=0, top=0, width=100, height=20),
        Word(text="two", left=5, top=0, width=100, height=20),
    ]

    grid = words_to_grid(crowded)

    assert "onetwo" not in grid
    assert grid.split() == ["one", "two"]


def test_no_words_produces_no_grid():
    assert words_to_grid([]) == ""


def test_single_character_words_do_not_break_the_width_estimate():
    grid = words_to_grid([_word("a", 0, 0), _word("b", 2, 0)])

    assert grid.split() == ["a", "b"]


# --- Feeding rag/layout.py --------------------------------------------------


def test_grid_from_word_boxes_segments_into_columns():
    """The point of the grid: an OCR'd page gets the same layout analysis."""
    left = ["Neural networks learn by", "adjusting their weights"]
    right = ["Transformers replaced", "recurrence with attention"]
    words = [
        _word(text, column, row)
        for row, (left_line, right_line) in enumerate(zip(left, right))
        for column, text in ((0, left_line), (40, right_line))
    ]
    # Four rows of prose is what column detection needs, so repeat the pair.
    words += [
        Word(text=word.text, left=word.left, top=word.top + 2 * ROW, width=word.width, height=word.height)
        for word in words
    ]

    blocks = analyze_page(words_to_grid(words))

    assert [block.kind for block in blocks] == ["text", "text"]
    assert blocks[0].text.startswith("Neural networks")
    assert blocks[1].text.startswith("Transformers replaced")


# --- Reading Tesseract's output --------------------------------------------


def test_low_confidence_words_are_discarded():
    data = {
        "text": ["real", "n0is3", ""],
        "conf": ["96", "12", "-1"],
        "left": [0, 100, 200],
        "top": [0, 0, 0],
        "width": [40, 50, 0],
        "height": [20, 20, 0],
    }

    words = ocr._to_words(data)

    assert [word.text for word in words] == ["real"]


def test_non_numeric_confidence_is_skipped_rather_than_crashing():
    data = {
        "text": ["ok", "weird"],
        "conf": ["90", None],
        "left": [0, 100],
        "top": [0, 0],
        "width": [20, 50],
        "height": [20, 20],
    }

    assert [word.text for word in ocr._to_words(data)] == ["ok"]


# --- Availability reporting -------------------------------------------------


def test_disabled_ocr_reports_a_reason(monkeypatch):
    monkeypatch.setattr(ocr.settings, "ocr_enabled", False)

    reason = ocr.unavailable_reason()

    assert reason is not None
    assert "OCR_ENABLED" in reason


def test_missing_binary_reports_how_to_install_it(monkeypatch):
    monkeypatch.setattr(ocr.settings, "ocr_enabled", True)
    monkeypatch.setattr(ocr, "_probe", lambda: "OCR is not available: the Tesseract binary was not found.")

    assert "Tesseract" in ocr.unavailable_reason()


@requires_tesseract
def test_available_ocr_reports_no_reason(monkeypatch):
    monkeypatch.setattr(ocr.settings, "ocr_enabled", True)

    assert ocr.unavailable_reason() is None
