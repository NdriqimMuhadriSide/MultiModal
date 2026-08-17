"""
Test helper: build document images whose contents are known by construction.

Evaluating agents/vision_agent.py needs images with ground truth attached -
"the total on this one really is 84.50" - and the honest ways to get that
are to hand-label real receipts or to draw them. Drawing wins here for
three reasons: the labels cannot be wrong, nothing sensitive is committed
to the repo, and the *degradations* that matter (low resolution, rotation,
no text at all) can be produced on demand rather than hunted for.

The same reasoning as tests/pdf_builder.py, which draws PDFs rather than
checking in fixtures, and the same naming rule: not test_*.py, so pytest
treats this as a helper rather than a test file.

WHAT THIS DELIBERATELY IS NOT

A stand-in for real photographs. A drawn receipt has perfect contrast, no
JPEG artefacts, no shadow, no crumple and no motion blur. Recognition will
do better on these than on anything a phone camera produces, so the numbers
an evaluation gets from them are an upper bound rather than a prediction.
`low_res` and `rotated` push in the right direction; they do not close the
gap. Real photographs are the next thing this set needs.
"""
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

# macOS ships Courier New; the Linux CI fallback is PIL's bitmap font, which
# is far worse for recognition. Tried in order, first hit wins.
_MONO_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)


@dataclass
class BuiltImage:
    """An image plus what is provably in it."""

    png: bytes
    # What a correct answer must contain. Keys are free-form; the graders
    # read `total`, `date` and `vendor`.
    truth: dict[str, str] = field(default_factory=dict)
    # False for images with no machine-readable printed text - a
    # photograph, a chart, a blank page. Recognition returning nothing on
    # these is the correct outcome, not a failure.
    has_text: bool = True


def _font(size: int):
    for path in _MONO_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


_RECEIPT_LINES = [
    (60, 40, 34, "CAFE MILANO"),
    (60, 110, 30, "DATE   2026-03-11"),
    (60, 160, 30, "TABLE  7      COVERS  2"),
    (60, 240, 30, "Pasta al forno        28.00"),
    (60, 285, 30, "Sea bass              34.50"),
    (60, 330, 30, "Mineral water          6.00"),
    # The line items have to sum to the total. Without it the first live
    # run produced a genuinely good answer - the agent added the items,
    # got 68.50, noticed it contradicted the printed 84.50, and reported
    # the discrepancy - against an inconsistency that was a bug in this
    # fixture rather than anything to find in the image.
    (60, 360, 30, "Service 12.5%         16.00"),
    (60, 425, 34, "TOTAL                 84.50"),
]

_RECEIPT_TRUTH = {
    "vendor": "CAFE MILANO",
    "date": "2026-03-11",
    "total": "84.50",
    "covers": "2",
}


def _draw_receipt(scale: float = 1.0) -> Image.Image:
    image = Image.new("RGB", (int(900 * scale), int(540 * scale)), "white")
    draw = ImageDraw.Draw(image)
    for x, y, size, text in _RECEIPT_LINES:
        draw.text(
            (x * scale, y * scale),
            text,
            font=_font(max(8, int(size * scale))),
            fill="black",
        )
    return image


def clean_receipt() -> BuiltImage:
    """A printed receipt at a resolution recognition should handle easily."""
    return BuiltImage(png=_to_png(_draw_receipt()), truth=dict(_RECEIPT_TRUTH))


def low_res_receipt() -> BuiltImage:
    """
    The same receipt at a third of the size.

    This is the case that produced the real finding behind constraint C-8:
    at small type Tesseract drops the decimal point, reading 84.50 as 8450.
    An evaluation that only ever sees clean input would never learn that the
    agent has to cope with it.
    """
    return BuiltImage(png=_to_png(_draw_receipt(scale=0.34)), truth=dict(_RECEIPT_TRUTH))


def rotated_receipt(degrees: float = 8.0) -> BuiltImage:
    """A receipt photographed at an angle - the commonest real degradation."""
    rotated = _draw_receipt().rotate(degrees, expand=True, fillcolor="white")
    return BuiltImage(png=_to_png(rotated), truth=dict(_RECEIPT_TRUTH))


def photograph() -> BuiltImage:
    """
    A textless image: a soft gradient with a few shapes.

    Recognition must return nothing here, and the agent must say so rather
    than reporting a total. Asking "what is the total?" of this image is the
    single most useful case in the set - it is where a confident wrong
    answer costs the most.
    """
    image = Image.new("RGB", (700, 480))
    draw = ImageDraw.Draw(image)
    for y in range(480):
        shade = 90 + int(110 * y / 480)
        draw.line([(0, y), (700, y)], fill=(shade, shade - 20, 60))
    draw.ellipse([120, 120, 330, 330], fill=(210, 180, 90))
    draw.polygon([(430, 350), (540, 140), (650, 350)], fill=(70, 110, 90))
    return BuiltImage(png=_to_png(image), truth={}, has_text=False)


def bar_chart() -> BuiltImage:
    """
    A chart with axis labels but no prose.

    Tests the other half of the reading decision: the shape of the data is
    something only the vision model can describe, while the labels are
    printed text. Neither tool alone answers "which quarter was highest".
    """
    image = Image.new("RGB", (720, 480), "white")
    draw = ImageDraw.Draw(image)
    bars = [("Q1", 90), ("Q2", 150), ("Q3", 240), ("Q4", 120)]
    draw.line([(80, 400), (660, 400)], fill="black", width=2)
    draw.line([(80, 60), (80, 400)], fill="black", width=2)
    for index, (label, height) in enumerate(bars):
        left = 130 + index * 130
        draw.rectangle([left, 400 - height, left + 80, 400], fill=(60, 90, 150))
        draw.text((left + 22, 412), label, font=_font(26), fill="black")
    draw.text((250, 20), "REVENUE BY QUARTER", font=_font(28), fill="black")
    return BuiltImage(
        png=_to_png(image),
        truth={"highest": "Q3", "title": "REVENUE BY QUARTER"},
    )


def blank_page() -> BuiltImage:
    """Nothing at all. The degenerate case both readers must survive."""
    return BuiltImage(png=_to_png(Image.new("RGB", (600, 400), "white")), truth={}, has_text=False)
