from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "毕业论文整合稿.md"
OUTPUT = ROOT / "博弈驱动的去中心化语义共识机制及在影视推荐中的应用_毕业论文整合稿.docx"


def set_fonts(run, cn="Songti SC", latin="Times New Roman", size=12, bold=None):
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cn)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.append(fld)


doc = Document()
sec = doc.sections[0]
sec.top_margin = Cm(2.6)
sec.bottom_margin = Cm(2.4)
sec.left_margin = Cm(3.0)
sec.right_margin = Cm(2.6)
add_page_number(sec.footer.paragraphs[0])

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Times New Roman"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Songti SC")
normal.font.size = Pt(12)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.first_line_indent = Cm(0.74)
normal.paragraph_format.space_after = Pt(0)

for name, size, cn in (("Title", 22, "STHeiti"), ("Heading 1", 16, "STHeiti"),
                       ("Heading 2", 14, "STHeiti"), ("Heading 3", 12, "STHeiti")):
    style = styles[name]
    style.font.name = "Times New Roman"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), cn)
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.space_before = Pt(12)
    style.paragraph_format.space_after = Pt(6)

lines = SOURCE.read_text(encoding="utf-8").splitlines()
first_title = True
for raw in lines:
    line = raw.strip()
    if not line:
        continue
    if line.startswith("# "):
        p = doc.add_paragraph(style="Title")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_fonts(p.add_run(line[2:]), cn="STHeiti", size=22, bold=True)
        first_title = False
    elif line.startswith("## "):
        if line.startswith("## 第") and len(doc.paragraphs) > 4:
            doc.add_page_break()
        p = doc.add_paragraph(line[3:], style="Heading 1")
    elif line.startswith("### "):
        p = doc.add_paragraph(line[4:], style="Heading 2")
    elif re.match(r"^\d+\.\s", line):
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.first_line_indent = Cm(0)
        p.add_run(re.sub(r"^\d+\.\s*", "", line))
    elif line.startswith("关键词："):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        r = p.add_run(line)
        set_fonts(r, size=12, bold=True)
    elif line.startswith("作者：") or line.startswith("导师："):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        set_fonts(p.add_run(line), size=12)
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_fonts(p.add_run(line), size=12)

settings = doc.settings.element
update = OxmlElement("w:updateFields")
update.set(qn("w:val"), "true")
settings.append(update)
doc.save(OUTPUT)
print(OUTPUT)
