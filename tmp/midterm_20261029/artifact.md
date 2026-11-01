# Template contract
Reference layout authority: /Users/spike/Downloads/LwMidReport.docx.
Narrative/figure organization authority: /Users/spike/Downloads/中期报告.pdf (42 pages, supplied example; its personal details and results are not project facts).
Existing project-specific cover source: /Users/spike/code/MAS/output/midterm/中期报告_学校Word模板重写稿.docx.
Reference DOCX SHA256: c8fcea4dce4384fafbbe83d0411f213cb11b5f0574162c20ecd87d8c398aa738.
A4 portrait, original w:sectPr retained: 11906 x 16838 twips, margins top/bottom1440 left1418 right851, header/footer567; first-page behavior retained.
Preserve source header/footer, school marks, section and styles, cover geometry, outer frame tables, information rows, mentor/assessment signature areas. Use existing repaired project DOCX package as working copy because original has malformed relationship/content-type handling of embedded aFChunks; existing copy preserves original styles and school marks.
Editable: cover project identity/date, thesis metadata values, repeated body content frames, report figures and source references. Unknown student ID/admission data remain blank; no copied example person/date/achievements.
Body: source-derived 宋体10.5pt, 312twip exact line spacing, 420twip first-line indent, 200twip left/right inner paragraph inset; bold black subsection headings. Captions centered10pt. Paragraph-rich technical exposition modeled after PDF; numbered six school sections and two research contents (one innovation plus one system).
Figures: new HTML-authored SVG drawings, white paper background with muted blue/teal/orange. They replace prior placeholders; high-resolution raster copies embedded in Word, independent SVG retained for editing.
First body frame retains six metadata rows; continuing frames clone source table border/grid without metadata, with correct current section heading. Content capacities checked by rendered pages; no font shrinking to force overflow.
Package preservation: all source parts except document.xml, document.xml.rels, Content_Types and image additions remain byte-for-byte identical; no changes to source or old project DOCX.
Reference rendering: reference_render (font warning diagnosed); corrected reference_fonts using task-local fontconfig mapping 宋体 to Songti SC. Original has imported-content pagination defects; do not reproduce blank pages or missing text. PDF reference is the visual authority for readable prose density.
Verification: render final using bundled soffice through render_docx.py and task-local fontconfig; visually inspect every final page, verify no clipping/blank spills and preserve section/header/package signatures.
