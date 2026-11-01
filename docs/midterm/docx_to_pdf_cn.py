from io import BytesIO
from pathlib import Path

from docx import Document
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph
from lxml import etree
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph as RLParagraph,
    Spacer, Table as RLTable, TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output/midterm/博弈驱动的去中心化语义共识机制_中期报告.docx"
OUT = ROOT / "output/midterm/博弈驱动的去中心化语义共识机制_中期报告.pdf"
FONT = Path("/Library/Fonts/Arial Unicode.ttf")
if not FONT.exists():
    FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")

pdfmetrics.registerFont(TTFont("CN", str(FONT)))


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("CN", 9)
    canvas.drawCentredString(A4[0] / 2, 1.15 * cm, str(doc.page))
    canvas.restoreState()


styles = getSampleStyleSheet()
BODY = ParagraphStyle("CNBody", parent=styles["BodyText"], fontName="CN", fontSize=10.5,
                      leading=17, firstLineIndent=2.0 * 10.5, alignment=TA_JUSTIFY,
                      spaceAfter=4, wordWrap="CJK", textColor=colors.black)
NOINDENT = ParagraphStyle("CNNoIndent", parent=BODY, firstLineIndent=0)
H1 = ParagraphStyle("CNH1", parent=BODY, fontSize=15, leading=21, spaceBefore=10,
                    spaceAfter=7, firstLineIndent=0, keepWithNext=True)
H2 = ParagraphStyle("CNH2", parent=BODY, fontSize=12, leading=18, spaceBefore=7,
                    spaceAfter=5, firstLineIndent=0, keepWithNext=True)
TITLE = ParagraphStyle("CNTitle", parent=BODY, fontSize=24, leading=34, alignment=TA_CENTER,
                       firstLineIndent=0, spaceAfter=25)
SUBTITLE = ParagraphStyle("CNSubtitle", parent=BODY, fontSize=18, leading=27, alignment=TA_CENTER,
                          firstLineIndent=0, spaceAfter=30)
CAPTION = ParagraphStyle("CNCaption", parent=BODY, fontSize=9.5, leading=14, alignment=TA_CENTER,
                         firstLineIndent=0, spaceBefore=3, spaceAfter=7)
CELL = ParagraphStyle("CNCell", parent=BODY, fontSize=8.2, leading=11.5, alignment=TA_LEFT,
                      firstLineIndent=0, spaceAfter=0)
CELL_CENTER = ParagraphStyle("CNCellCenter", parent=CELL, alignment=TA_CENTER)
REF = ParagraphStyle("CNRef", parent=BODY, fontSize=9, leading=13, leftIndent=17,
                     firstLineIndent=-17, spaceAfter=3)


def iter_blocks(parent):
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield DocxTable(child, parent)


def has_page_break(p):
    return bool(p._p.xpath('.//w:br[@w:type="page"]'))


def paragraph_images(p):
    images = []
    for blip in p._p.xpath(".//a:blip"):
        rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
        if rid:
            images.append(p.part.related_parts[rid].blob)
    return images


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def para_style(p, index):
    name = p.style.name if p.style else ""
    text = p.text.strip()
    if index == 4:
        return TITLE
    if index == 5:
        return SUBTITLE
    if name == "Heading 1":
        return H1
    if name == "Heading 2":
        return H2
    if text.startswith("图 ") or text.startswith("表 "):
        return CAPTION
    if text.startswith("[") and text[:3].strip("[").isdigit():
        return REF
    if p.paragraph_format.first_line_indent is None or p.paragraph_format.first_line_indent == 0:
        return NOINDENT
    return BODY


def build():
    source = Document(SRC)
    frame = Frame(2.25 * cm, 1.65 * cm, A4[0] - 4.5 * cm, A4[1] - 3.75 * cm,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    template = PageTemplate(id="normal", frames=[frame], onPage=page_number)
    doc = BaseDocTemplate(str(OUT), pagesize=A4, pageTemplates=[template],
                          leftMargin=2.25 * cm, rightMargin=2.25 * cm,
                          topMargin=2.1 * cm, bottomMargin=1.65 * cm,
                          title="博弈驱动的去中心化语义共识机制 中期报告",
                          author="马天成")
    story = []
    p_index = 0
    for block in iter_blocks(source):
        if isinstance(block, Paragraph):
            p_index += 1
            if has_page_break(block):
                story.append(PageBreak())
                continue
            imgs = paragraph_images(block)
            for blob in imgs:
                bio = BytesIO(blob)
                im = Image(bio)
                max_w, max_h = 15.2 * cm, 8.7 * cm
                scale = min(max_w / im.imageWidth, max_h / im.imageHeight, 1)
                im.drawWidth = im.imageWidth * scale
                im.drawHeight = im.imageHeight * scale
                im.hAlign = "CENTER"
                story.append(im)
            text = block.text.strip()
            if not text:
                if p_index < 5:
                    story.append(Spacer(1, 0.45 * cm))
                continue
            story.append(RLParagraph(esc(text), para_style(block, p_index)))
        else:
            data = []
            for i, row in enumerate(block.rows):
                vals = []
                for j, cell in enumerate(row.cells):
                    txt = "<br/>".join(esc(p.text.strip()) for p in cell.paragraphs if p.text.strip())
                    vals.append(RLParagraph(txt or " ", CELL_CENTER if i == 0 or j == 0 else CELL))
                data.append(vals)
            if not data:
                continue
            n = len(data[0])
            avail = A4[0] - 4.5 * cm
            if n == 2:
                widths = [avail * 0.34, avail * 0.66]
            elif n == 3:
                widths = [avail * 0.22, avail * 0.55, avail * 0.23]
            elif n == 4:
                widths = [avail * 0.17, avail * 0.33, avail * 0.17, avail * 0.33]
            elif n == 5:
                widths = [avail * 0.19, avail * 0.13, avail * 0.15, avail * 0.15, avail * 0.38]
            else:
                widths = [avail / n] * n
            table = RLTable(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
            commands = [
                ("FONTNAME", (0, 0), (-1, -1), "CN"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            for i in range(2, len(data), 2):
                commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#EAF2F8")))
            table.setStyle(TableStyle(commands))
            story.extend([Spacer(1, 4), table, Spacer(1, 7)])
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
