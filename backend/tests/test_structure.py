"""
Structure tests.

Blocks and font sizes are built by hand here rather than extracted from a PDF:
they are exactly rag/structure.py's input, so a test can state the precise
document shape it's about. End-to-end coverage from real PDF bytes lives in
test_pdf_loader.py.
"""
from rag.layout import Block
from rag.structure import annotate_structure, normalize, section_label

BODY = 10.0


def _text(text: str) -> Block:
    return Block(kind="text", text=text)


def _sizes(*pairs: tuple[str, float]) -> dict[str, float]:
    return {normalize(text): size for text, size in pairs}


def _annotate(blocks: list[Block], sizes: dict[str, float]) -> list[Block]:
    return annotate_structure([blocks], [sizes])[0]


# --- Detection --------------------------------------------------------------


def test_a_larger_short_line_is_a_heading():
    blocks = _annotate(
        [_text("Introduction"), _text("This report covers the year in review.")],
        _sizes(("Introduction", 16.0), ("This report covers the year in review.", BODY)),
    )

    assert [block.kind for block in blocks] == ["heading", "text"]
    assert blocks[0].level == 1


def test_body_sized_prose_is_not_a_heading_however_short():
    blocks = _annotate(
        [_text("Introduction"), _text("This report covers the year.")],
        _sizes(("Introduction", BODY), ("This report covers the year.", BODY)),
    )

    assert [block.kind for block in blocks] == ["text", "text"]


def test_a_long_block_is_never_a_heading_even_when_set_large():
    # A pull quote is set large and is still prose.
    quote = "The single most important finding of the year was that " * 3
    blocks = _annotate([_text(quote)], _sizes((quote, 18.0)))

    assert blocks[0].kind == "text"


def test_numbered_headings_are_detected_without_font_data():
    blocks = _annotate([_text("3.1 Field Sampling"), _text("We collected samples.")], {})

    assert [block.kind for block in blocks] == ["heading", "text"]


def test_a_numbered_procedure_step_is_not_a_heading():
    """
    The guard that stops one list item swallowing the document.

    "1. Rinse the sample thoroughly." matches the numbering pattern exactly as
    a heading would; the closing full stop is what separates them.
    """
    blocks = _annotate([_text("1. Rinse the sample thoroughly.")], {})

    assert blocks[0].kind == "text"


def test_named_divisions_are_headings():
    blocks = _annotate([_text("Appendix B"), _text("Supporting tables follow.")], {})

    assert blocks[0].kind == "heading"
    assert blocks[0].level == 1


def test_capitalised_lines_are_headings_only_without_font_data():
    """
    The OCR case. With sizes available, capitals must not overrule a measured
    body size - plenty of body text is set in capitals.
    """
    without_fonts = _annotate([_text("METHODS"), _text("We collected samples.")], {})
    with_fonts = _annotate(
        [_text("METHODS"), _text("We collected samples.")],
        _sizes(("METHODS", BODY), ("We collected samples.", BODY)),
    )

    assert without_fonts[0].kind == "heading"
    assert with_fonts[0].kind == "text"


def test_a_lone_number_is_not_promoted_to_a_heading():
    blocks = _annotate([_text("500"), _text("units shipped")], {})

    assert [block.kind for block in blocks] == ["text", "text"]


def test_tables_are_never_headings():
    table = Block(kind="table", text="| a | b |")

    blocks = _annotate([table], _sizes(("| a | b |", 20.0)))

    assert blocks[0].kind == "table"


# --- Levels -----------------------------------------------------------------


def test_levels_are_ranked_by_font_size():
    blocks = _annotate(
        [_text("Annual Report"), _text("1. Introduction"), _text("1.1 Scope")],
        _sizes(("Annual Report", 20.0), ("1. Introduction", 14.0), ("1.1 Scope", 12.0)),
    )

    assert [block.level for block in blocks] == [1, 2, 3]


def test_an_unnumbered_title_outranks_a_numbered_chapter():
    """
    Why levels come from size rather than numbering depth.

    Read off the numbering alone, "1. Introduction" is level 1 and the title -
    having no number at all - would default to level 1 too, making them
    siblings. The sizes know better.
    """
    blocks = _annotate(
        [_text("Annual Report"), _text("1. Introduction")],
        _sizes(("Annual Report", 20.0), ("1. Introduction", 14.0)),
    )

    assert blocks[0].level == 1
    assert blocks[1].level == 2


def test_numbering_depth_sets_the_level_without_font_data():
    blocks = _annotate([_text("3. Methods"), _text("3.1 Sampling"), _text("3.1.2 Sites")], {})

    assert [block.level for block in blocks] == [1, 2, 3]


def test_near_identical_sizes_are_one_level():
    # The same heading measured through slightly different transforms must
    # not become two levels.
    blocks = _annotate(
        [_text("First Section"), _text("Second Section")],
        _sizes(("First Section", 14.0), ("Second Section", 14.04)),
    )

    assert blocks[0].level == blocks[1].level


# --- Section paths ----------------------------------------------------------


def test_blocks_inherit_the_heading_above_them():
    blocks = _annotate(
        [_text("2. Methods"), _text("We collected 500 samples.")],
        _sizes(("2. Methods", 14.0), ("We collected 500 samples.", BODY)),
    )

    assert blocks[1].section_path == ("2. Methods",)


def test_a_heading_includes_itself_in_its_path():
    """So a heading groups with the prose it introduces rather than alone."""
    blocks = _annotate([_text("2. Methods")], _sizes(("2. Methods", 14.0)))

    assert blocks[0].section_path == ("2. Methods",)


def test_a_subsection_nests_under_its_parent():
    blocks = _annotate(
        [_text("2. Methods"), _text("2.1 Sampling"), _text("We collected samples.")],
        _sizes(("2. Methods", 14.0), ("2.1 Sampling", 12.0), ("We collected samples.", BODY)),
    )

    assert blocks[2].section_path == ("2. Methods", "2.1 Sampling")


def test_a_sibling_heading_closes_the_previous_subsection():
    blocks = _annotate(
        [
            _text("2. Methods"),
            _text("2.1 Sampling"),
            _text("3. Results"),
            _text("Yields rose sharply."),
        ],
        _sizes(
            ("2. Methods", 14.0),
            ("2.1 Sampling", 12.0),
            ("3. Results", 14.0),
            ("Yields rose sharply.", BODY),
        ),
    )

    assert blocks[3].section_path == ("3. Results",)


def test_text_before_the_first_heading_has_no_section():
    blocks = _annotate(
        [_text("Published March 2026"), _text("1. Introduction")],
        _sizes(("Published March 2026", BODY), ("1. Introduction", 14.0)),
    )

    assert blocks[0].section_path == ()


def test_sections_continue_across_pages():
    """A section opened on one page runs until the next heading, wherever it is."""
    pages = annotate_structure(
        [[_text("2. Methods"), _text("We collected samples.")], [_text("Analysis continued.")]],
        [_sizes(("2. Methods", 14.0), ("We collected samples.", BODY)), {}],
    )

    assert pages[1][0].section_path == ("2. Methods",)


def test_documents_with_no_headings_leave_every_path_empty():
    blocks = _annotate(
        [_text("Just prose."), _text("More prose.")],
        _sizes(("Just prose.", BODY), ("More prose.", BODY)),
    )

    assert all(block.section_path == () for block in blocks)
    assert all(block.kind == "text" for block in blocks)


# --- Labels -----------------------------------------------------------------


def test_section_label_joins_the_path():
    assert section_label(("2. Methods", "2.1 Sampling")) == "2. Methods > 2.1 Sampling"
    assert section_label(()) == ""
