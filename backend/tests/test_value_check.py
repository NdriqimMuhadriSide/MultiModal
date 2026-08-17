"""
Tests for the numeric traceability check (agents/value_check.py).

This is the function behind SC-10 - "every number, date and code in the
answer appears in the OCR output" - so the cases that matter most are the
two it must not get wrong: a fabricated figure has to be caught, and a
correctly formatted one must not be flagged for punctuation.
"""
from agents.value_check import unverified_values

RECEIPT = "CAFE MILANO\nDATE  2026-03-11\n2 COVERS\nTOTAL      84.50\nVAT  14.08"


# ---- The case it exists for ------------------------------------------------


def test_a_fabricated_total_is_flagged():
    answer = "The total on the receipt is £92.00."
    assert unverified_values(answer, RECEIPT) == ["£92.00"]


def test_an_answer_grounded_in_the_grid_is_clean():
    answer = "The total is 84.50, dated 2026-03-11, VAT 14.08."
    assert unverified_values(answer, RECEIPT) == []


def test_no_ocr_at_all_flags_every_figure():
    """read_text never ran, or found nothing - so nothing is confirmed."""
    answer = "The total is 84.50 on 2026-03-11."
    assert unverified_values(answer, "") == ["84.50", "2026-03-11"]


# ---- Formatting must not cause false alarms --------------------------------


def test_currency_symbols_and_separators_are_ignored():
    answer = "The total is £84.50."
    assert unverified_values(answer, RECEIPT) == []


def test_thousands_separators_are_ignored():
    assert unverified_values("It comes to £1,234.50.", "TOTAL 1234.50") == []


def test_a_lost_decimal_point_in_the_grid_still_matches():
    """
    Real Tesseract behaviour on low-resolution input: 84.50 recognised as
    8450. The digits were read correctly and only the point was dropped, so
    quoting 84.50 is not a fabrication.
    """
    assert unverified_values("The total is 84.50.", "TOTAL  8450") == []


# ---- Things that are not claims about the image ----------------------------


def test_citation_labels_are_not_treated_as_values():
    answer = "The cap is £50 per person [E1] and approval is needed [E12]."
    assert unverified_values(answer, RECEIPT) == ["£50"]


def test_single_digit_tokens_are_skipped_as_noise():
    """"2 covers" and "step 3" are not figures worth reporting."""
    assert unverified_values("There were 2 covers across 3 items.", RECEIPT) == []


def test_an_answer_with_no_numbers_is_clean():
    assert unverified_values("It appears to be a restaurant receipt.", RECEIPT) == []


def test_an_empty_answer_is_clean():
    assert unverified_values("", RECEIPT) == []


# ---- Reporting shape -------------------------------------------------------


def test_results_are_deduplicated_in_first_appearance_order():
    answer = "It says 99.99, then 77.77, then 99.99 again."
    assert unverified_values(answer, RECEIPT) == ["99.99", "77.77"]


def test_a_derived_value_is_reported_not_suppressed():
    """
    84.50 across 2 covers is 42.25 a head - correct arithmetic, and not on
    the receipt. It is reported because the reader should know it was
    computed, not read. This is why the check informs rather than enforces.
    """
    answer = "£84.50 across 2 covers is £42.25 a head, inside the £50 cap [E1]."
    assert unverified_values(answer, RECEIPT) == ["£42.25", "£50"]


def test_a_figure_is_not_confirmed_by_digits_inside_another_number():
    """
    The trap this check has to avoid. Concatenating every digit on the page
    makes "50" match the 50 inside "84.50", so a policy limit that never
    appeared on the image would be silently blessed.
    """
    assert unverified_values("The cap is 50.", "TOTAL 84.50") == ["50"]


def test_a_prefix_of_a_recognised_token_counts_as_confirmed():
    """"in 2026" is grounded by a grid containing 2026-03-11."""
    assert unverified_values("It was issued in 2026.", RECEIPT) == []
