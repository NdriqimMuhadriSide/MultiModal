"""
Layout analysis tests.

These work on character grids written inline as strings rather than on real
PDFs: the grid *is* rag/layout.py's input, so a test can state the exact
geometry it cares about. End-to-end coverage from actual PDF bytes lives in
test_pdf_loader.py.
"""
import textwrap

from rag.layout import analyze_page, analyze_pages, blocks_to_text


def _grid(text: str) -> str:
    return textwrap.dedent(text).strip("\n")


# --- Blocks and reading order ---------------------------------------------


def test_plain_paragraph_is_one_text_block():
    blocks = analyze_page(
        _grid(
            """
            The quick brown fox jumps over
            the lazy dog and keeps running
            """
        )
    )

    assert [block.kind for block in blocks] == ["text"]
    assert blocks[0].text == "The quick brown fox jumps over\nthe lazy dog and keeps running"


def test_blank_rows_separate_stacked_blocks():
    blocks = analyze_page(
        _grid(
            """
            A Short Heading


            Body text under the heading.
            """
        )
    )

    assert [block.text for block in blocks] == [
        "A Short Heading",
        "Body text under the heading.",
    ]


def test_two_columns_are_read_left_then_right():
    blocks = analyze_page(
        _grid(
            """
            Neural networks learn by adjusting        Transformers replaced recurrence
            weights through backpropagation.          with self-attention, which lets
            Each layer transforms its input           every token look at every other
            into a richer representation.             token in a single step.
            """
        )
    )

    assert [block.kind for block in blocks] == ["text", "text"]
    # The left column is emitted whole before the right one starts - not
    # interleaved line by line, which is what content-stream order would give.
    assert blocks[0].text.startswith("Neural networks")
    assert blocks[0].text.endswith("into a richer representation.")
    assert blocks[1].text.startswith("Transformers replaced")
    assert blocks[1].text.endswith("token in a single step.")


def test_indentation_is_dropped_from_text_blocks():
    blocks = analyze_page("        indented line\n        another line")

    assert blocks[0].text == "indented line\nanother line"


# --- Tables ----------------------------------------------------------------


def test_aligned_columns_become_a_markdown_table():
    blocks = analyze_page(
        _grid(
            """
            Model          Params      Accuracy
            BERT           110M        88.5
            GPT-2          1.5B        91.2
            """
        )
    )

    assert [block.kind for block in blocks] == ["table"]
    assert blocks[0].text == (
        "| Model | Params | Accuracy |\n"
        "| --- | --- | --- |\n"
        "| BERT | 110M | 88.5 |\n"
        "| GPT-2 | 1.5B | 91.2 |"
    )


def test_tall_table_is_not_mistaken_for_prose_columns():
    # Four rows clears the row count that prose columns need, so this is only
    # kept as a table because the cells are too narrow to be wrapped text.
    blocks = analyze_page(
        _grid(
            """
            Model          Params      Accuracy
            BERT           110M        88.5
            GPT-2          1.5B        91.2
            T5             11B         92.7
            """
        )
    )

    assert [block.kind for block in blocks] == ["table"]
    assert blocks[0].text.endswith("| T5 | 11B | 92.7 |")


def test_empty_cells_are_preserved_as_empty_columns():
    blocks = analyze_page(
        _grid(
            """
            Model          Params      Accuracy
            BERT                       88.5
            GPT-2          1.5B        91.2
            """
        )
    )

    assert "| BERT |  | 88.5 |" in blocks[0].text


def test_pipes_in_cells_are_escaped():
    blocks = analyze_page(
        _grid(
            """
            Field          Value
            ratio          a|b
            scale          c|d
            """
        )
    )

    assert r"| ratio | a\|b |" in blocks[0].text


def test_two_rows_are_too_few_to_be_a_table():
    # A two-line paragraph whose word breaks happen to line up would otherwise
    # be served back as a table of nonsense.
    blocks = analyze_page(
        _grid(
            """
            Alpha          beta gamma
            Gamma          delta epsilon
            """
        )
    )

    assert [block.kind for block in blocks] == ["text"]


def test_narrow_gaps_do_not_make_a_table():
    blocks = analyze_page(
        _grid(
            """
            Alpha  beta
            Gamma  delta
            Kappa  zeta
            """
        )
    )

    assert [block.kind for block in blocks] == ["text"]


def test_a_wider_gap_does_make_a_table():
    # Same content as above with one more space in the gap - the threshold
    # this pair of tests pins down.
    blocks = analyze_page(
        _grid(
            """
            Alpha   beta
            Gamma   delta
            Kappa   zeta
            """
        )
    )

    assert [block.kind for block in blocks] == ["table"]


def test_a_stray_word_does_not_create_a_column():
    blocks = analyze_page(
        _grid(
            """
            The quick brown fox jumps over the lazy dog
            and then keeps on running through the woods
            until it reaches the river         Footnote
            """
        )
    )

    assert [block.kind for block in blocks] == ["text"]


# --- Running headers and footers -------------------------------------------


def _document_page(page_number: int) -> str:
    return _grid(
        f"""
        ACME RESEARCH QUARTERLY


        Body text on every page.
        A second line of body text.


        Page {page_number}
        """
    )


def test_repeated_header_and_footer_are_stripped():
    pages = analyze_pages([_document_page(number) for number in (1, 2, 3)])

    for blocks in pages:
        text = blocks_to_text(blocks)
        assert text == "Body text on every page.\nA second line of body text."


def test_page_numbers_are_stripped_even_without_enough_pages_to_compare():
    # Two pages is below the repetition threshold, so the header survives -
    # but a footer that is only a page number can't be anything else.
    pages = analyze_pages([_document_page(number) for number in (1, 2)])

    for blocks in pages:
        text = blocks_to_text(blocks)
        assert "ACME RESEARCH QUARTERLY" in text
        assert "Page" not in text


def test_body_text_is_never_treated_as_boilerplate():
    # The body repeats verbatim across all three pages here; only the margins
    # are eligible, so it survives.
    pages = analyze_pages([_document_page(number) for number in (1, 2, 3)])

    assert all("Body text on every page." in blocks_to_text(blocks) for blocks in pages)


def test_single_band_page_keeps_its_only_content():
    pages = analyze_pages(["Just one line of text.", "Just one line of text."])

    assert [blocks_to_text(blocks) for blocks in pages] == [
        "Just one line of text.",
        "Just one line of text.",
    ]


# --- Flattening ------------------------------------------------------------


def test_blocks_to_text_separates_blocks_with_a_blank_line():
    blocks = analyze_page(
        _grid(
            """
            A Short Heading


            Body text under the heading.
            """
        )
    )

    assert blocks_to_text(blocks) == "A Short Heading\n\nBody text under the heading."


def test_empty_page_produces_no_blocks():
    assert analyze_page("") == []
    assert analyze_page("   \n\n  ") == []
    assert blocks_to_text([]) == ""
