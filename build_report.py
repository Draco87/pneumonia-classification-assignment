"""Build report.pdf from REPORT.md with figures embedded in document order.

This utility uses ReportLab only for report generation. It is not required for
model training, evaluation, inference, or the AI assistant.

Install the optional documentation dependency with:
    pip install reportlab
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "REPORT.md"
OUTPUT = ROOT / "report.pdf"

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 18 * mm
TOP = 17 * mm
BOTTOM = 16 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT


def inline_markup(text: str) -> str:
    """Convert the small inline Markdown subset used by REPORT.md."""
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(
        r"(https?://[^\s<]+)",
        r'<link href="\1" color="#1f618d">\1</link>',
        text,
    )
    return text


styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=21, leading=25, alignment=TA_CENTER, textColor=colors.HexColor("#154360"),
    spaceAfter=8 * mm,
)
H2 = ParagraphStyle(
    "H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=15, leading=18, textColor=colors.HexColor("#1b4f72"),
    spaceBefore=5 * mm, spaceAfter=2.5 * mm, keepWithNext=True,
)
H3 = ParagraphStyle(
    "H3Custom", parent=styles["Heading3"], fontName="Helvetica-Bold",
    fontSize=11.5, leading=14, textColor=colors.HexColor("#21618c"),
    spaceBefore=3.5 * mm, spaceAfter=1.5 * mm, keepWithNext=True,
)
BODY = ParagraphStyle(
    "BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.6, leading=13.1, alignment=TA_JUSTIFY,
    textColor=colors.HexColor("#17202a"), spaceAfter=2.2 * mm,
)
BULLET = ParagraphStyle(
    "BulletCustom", parent=BODY, alignment=TA_LEFT, leftIndent=0,
    firstLineIndent=0, spaceAfter=0.8 * mm,
)
CAPTION = ParagraphStyle(
    "CaptionCustom", parent=BODY, fontName="Helvetica-Oblique",
    fontSize=8.2, leading=10, alignment=TA_CENTER,
    textColor=colors.HexColor("#566573"), spaceBefore=1.2 * mm,
    spaceAfter=3 * mm,
)
REFERENCE = ParagraphStyle(
    "ReferenceCustom", parent=BODY, alignment=TA_LEFT,
    leftIndent=7 * mm, firstLineIndent=-7 * mm, spaceAfter=1.8 * mm,
)


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(PAGE_W / 2, 8 * mm, str(doc.page))
    canvas.restoreState()


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="CPU-Friendly Pneumonia Classification from Pediatric Chest X-Rays",
            author="AI/ML Engineer Hiring Assignment",
        )
        frame = Frame(
            LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=page_number))


def scaled_image(path: Path, max_width=CONTENT_W, max_height=105 * mm) -> Image:
    """Create an aspect-ratio-preserving image that fits comfortably on A4."""
    img = Image(str(path))
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"
    return img


def table_flowable(rows: list[list[str]]) -> Table:
    data = [[Paragraph(inline_markup(cell), BODY) for cell in row] for row in rows]
    cols = max(len(row) for row in rows)

    # Distribute width by content length while keeping tables inside the page.
    lengths = []
    for c in range(cols):
        lengths.append(max(len(row[c]) if c < len(row) else 0 for row in rows))
    total = max(sum(lengths), 1)
    widths = [max(20 * mm, CONTENT_W * n / total) for n in lengths]
    factor = CONTENT_W / sum(widths)
    widths = [w * factor for w in widths]

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d6eaf8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17202a")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#85929e")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return table


def markdown_to_story(markdown: str):
    lines = markdown.splitlines()
    story = []
    paragraph = []
    bullets = []
    in_references = False
    i = 0

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), BODY))
            paragraph = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            items = [ListItem(Paragraph(inline_markup(x), BULLET)) for x in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=6 * mm))
            story.append(Spacer(1, 1.5 * mm))
            bullets = []

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Pipe table
        if line.startswith("|") and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if sep.startswith("|") and re.fullmatch(r"[\|\-:\s]+", sep):
                flush_paragraph()
                flush_bullets()
                rows = [[c.strip() for c in line.strip("|").split("|")]]
                i += 2
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                story.append(table_flowable(rows))
                story.append(Spacer(1, 2 * mm))
                continue

        # Image + caption kept together at its exact Markdown position.
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", line)
        if m:
            flush_paragraph()
            flush_bullets()
            caption, rel = m.group(1), m.group(2)
            path = ROOT / rel
            if path.exists():
                fig = scaled_image(path)
                story.append(KeepTogether([
                    fig,
                    Paragraph(inline_markup(caption), CAPTION),
                ]))
            else:
                story.append(Paragraph(
                    f"<b>Missing figure:</b> {escape(rel)}", BODY
                ))
            i += 1
            continue

        h = re.match(r"^(#{1,6})\s+(.+)$", line)
        if h:
            flush_paragraph()
            flush_bullets()
            level = len(h.group(1))
            heading_text = h.group(2)
            in_references = heading_text.lower() == "references"
            if level == 1:
                story.append(Paragraph(inline_markup(heading_text), TITLE))
            elif level == 2:
                story.append(Paragraph(inline_markup(heading_text), H2))
            else:
                story.append(Paragraph(inline_markup(heading_text), H3))
            i += 1
            continue

        # Numbered references: one paragraph per source with hanging indent.
        ref = re.match(r"^(\d+)\.\s+(.+)$", line)
        if in_references and ref:
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(
                f"<b>{ref.group(1)}.</b> {inline_markup(ref.group(2))}",
                REFERENCE,
            ))
            i += 1
            continue

        if line.startswith("- "):
            flush_paragraph()
            bullets.append(line[2:])
        elif not line:
            flush_paragraph()
            flush_bullets()
        else:
            flush_bullets()
            paragraph.append(line)

        i += 1

    flush_paragraph()
    flush_bullets()
    return story


def build_pdf():
    markdown = SOURCE.read_text(encoding="utf-8")
    story = markdown_to_story(markdown)
    doc = ReportDocTemplate(str(OUTPUT))
    doc.build(story)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_pdf()