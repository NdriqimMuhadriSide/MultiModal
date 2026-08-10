"""
Test helper: build a PDF with text at chosen (x, y) positions.

Layout tests need PDFs whose *geometry* is known - two columns here, a table
there, a footer down at the bottom - and pypdf can only copy or blank pages,
not draw text. Rather than adding a rendering library (reportlab) as a test
dependency, this writes the PDF by hand; a page with positioned text needs
only a catalog, a page tree, one content stream, and a standard font.

Not named test_*.py so pytest treats it as a helper module, not a test file.
"""
from io import BytesIO

# (x, y, font_size, text). y is measured from the *bottom* of the page, which
# is how PDF's coordinate system works.
TextItem = tuple[float, float, float, str]

US_LETTER = (612, 792)


def build_pdf(
    pages: list[list[TextItem]],
    width: float = US_LETTER[0],
    height: float = US_LETTER[1],
    font: str = "Helvetica",
) -> bytes:
    """Render one content stream per page, each placing its items absolutely."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # PDF object numbers are 1-indexed

    content_ids = [add(_content_stream(items)) for items in pages]
    font_id = add(f"<< /Type /Font /Subtype /Type1 /BaseFont /{font} >>".encode())

    # Each page points back at the page tree, which is only created after the
    # pages themselves - so its object number is worked out in advance.
    pages_id = len(objects) + len(pages) + 1
    page_ids = [
        add(
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        for content_id in content_ids
    ]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    assert add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()) == pages_id
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    return _serialize(objects, catalog_id)


def _content_stream(items: list[TextItem]) -> bytes:
    """A text object per item: set the font, set the text matrix, show the string."""
    operators = [b"BT"]
    for x, y, size, text in items:
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        operators.append(f"/F1 {size} Tf 1 0 0 1 {x} {y} Tm ({escaped}) Tj".encode("latin-1"))
    operators.append(b"ET")
    stream = b"\n".join(operators)
    return b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)


def _serialize(objects: list[bytes], catalog_id: int) -> bytes:
    """Write the objects out with the cross-reference table pypdf expects."""
    out = BytesIO()
    out.write(b"%PDF-1.4\n")

    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))

    xref_offset = out.tell()
    out.write(b"xref\n0 %d\n" % (len(objects) + 1))
    out.write(b"0000000000 65535 f \n")
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, catalog_id, xref_offset)
    )
    return out.getvalue()


def build_scanned_pdf(
    pages: list[list[TextItem]],
    width: int = 1700,
    height: int = 2200,
    font_size: int = 34,
) -> bytes:
    """
    Build an image-only PDF - the shape a scanner produces.

    Each page is drawn onto a bitmap and embedded as a single image, so the
    PDF carries no glyphs at all and `extract_text()` correctly returns "".
    That is exactly the input the OCR fallback exists for.

    Coordinates match build_pdf's convention (y from the bottom, in points at
    72dpi) and are scaled up to the raster's pixel grid here, so a page layout
    can be handed to either builder unchanged.
    """
    from PIL import Image, ImageDraw, ImageFont

    scale = height / US_LETTER[1]
    font = ImageFont.load_default(size=font_size)

    images = []
    for items in pages:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        for x, y, _, text in items:
            draw.text((x * scale, (US_LETTER[1] - y) * scale), text, fill="black", font=font)
        images.append(image)

    buffer = BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def two_column_page(
    left: list[str],
    right: list[str],
    top: float = 700,
    leading: float = 14,
    size: float = 10,
    left_x: float = 72,
    right_x: float = 330,
) -> list[TextItem]:
    """
    Lay two blocks of text side by side, one line per entry.

    The content stream is written row by row (left cell, right cell, next row)
    rather than column by column. That's the case worth testing: it's what many
    generators emit, and it's exactly where content-stream order stops matching
    reading order.
    """
    items: list[TextItem] = []
    for index in range(max(len(left), len(right))):
        y = top - index * leading
        if index < len(left):
            items.append((left_x, y, size, left[index]))
        if index < len(right):
            items.append((right_x, y, size, right[index]))
    return items


def table_rows(
    rows: list[list[str]],
    column_x: list[float],
    top: float = 500,
    leading: float = 14,
    size: float = 10,
) -> list[TextItem]:
    """Lay out rows of cells at fixed column x-positions."""
    return [
        (column_x[column], top - index * leading, size, cell)
        for index, row in enumerate(rows)
        for column, cell in enumerate(row)
        if cell
    ]
