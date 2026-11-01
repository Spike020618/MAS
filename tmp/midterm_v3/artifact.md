# Word template execution contract

Reference: /Users/spike/Downloads/LwMidReport.docx
SHA256: c8fcea4dce4384fafbbe83d0411f213cb11b5f0574162c20ecd87d8c398aa738
Reference render: /Users/spike/code/MAS/tmp/midterm_v3/reference
Reference runtime pagination: 52 pages, including visible altChunk pagination defects.

## Page system and recurring elements

One original A4 portrait section, w:pgSz 11906 by 16838 twips. Margins top/bottom 1440, left 1418, right 851; header/footer 567 twips. One column. First page header suppression via titlePg. Header1 contains school logo, report name and horizontal rule; footer1 contains the page field. Preserve both parts, their relationships and media byte-for-byte.

Normal style 1 uses Calibri Latin, 宋体 East Asian, 10.5 pt, justified, widow control off. Report body patterns in the embedded documents use 宋体, paragraph spacing after 50 twips, line 276 auto, first line approximately two Chinese characters, 200 twip left/right indents. Section heading patterns use source cell paragraphs with 宋体 and direct formatting. Cover title 黑体 24 pt. Cover information 宋体14 pt, aligned labels with underline shapes. Retain source cover title, school logo, underlines, label positions and margins. Replace personal values and dates; remove the other student's barcode image from its cover cell, replacing it with a pending barcode label.

## Outer form and slot map

word/document.xml body/table[1]: cover logo and barcode; preserve logo and table layout, replace barcode content.
body paragraphs 11-18: personal fields, replace only value text nodes; unknown values become 待填写. paragraph23 date replaced. Cover spacing may reduce empty spacer paragraphs only to keep university/date on cover rather than spilling onto another page.
body/table[2] rows1-6: metadata grid. Preserve widths, merges, cell margins, borders, labels. Replace value cell text. Row7: section one bordered body slot.
body/table[3] row1: section two progress slot. Row2: section three research and system成果 slot.
body/table[4] rows1-2: sections four/five plan and整改.
body/table[5] row1: references, replace unrelated bibliography.
body/table[6]: evaluation form. Preserve label, rows, border pattern, signature/check controls; remove someone else's advisor assessment and leave blank.

Outer metadata table: style5, centered, outer borders12 eighth points, internal6, grid 1848/3250/1825/2701 twips. Section-body cells are single merged cells with source border patterns and top/left/right padding. Preserve outer frame style. Remove large source row minimums when filling to prevent blank framed pages. Permit native rows/cells to split across pages. Content growth may clone native paragraph/table/figure patterns. No generic restyling or blank-document reconstruction.

## Native content replacement and compatibility

The reference stores five unrelated narrative sections as aFChunk embedded DOCX packages. Replace those chunks with native w:p/w:tbl content inside the corresponding source cells. Remove their document relationships and unused embedded packages as intended source-content removal; all other opaque parts remain preserved. This also resolves python-docx mistakenly treating embedded .docx as XML due to the original content type declaration. Do not use a LibreOffice save round-trip for authoring.

Cover metadata, all section-body paragraphs, nested tables, embedded reports, reference cells and evaluation prose are editable. Styles, theme, font table, header/footer and their relationships, customXml, untouched media and settings structures are preserve-only except updateFields in settings. document.xml.rels is editable only for removing obsolete aFChunk links and the other student's barcode link. Content types may remove the unused docx default. User requested content narrative derives from the supplied PDF, not the unrelated cloud scheduling subject in this DOCX.

## Content flow and fidelity gates

Order remains: cover, research introduction (background, two content tracks, technical methods, plan and progress goals), actual progress, achievements (innovation research followed by repo system and evaluations), future plan, problems/remedies, references, advisor/evaluation. Retain long-form explanatory Chinese paragraphs in the PDF's style. No fabricated research success, personal identifiers, enrollment dates, publications or signatures. Figure/screenshot placeholders are concise and editable.

Target about the original PDF's 42-page content envelope, with natural pagination under this Word template. Page count is checked after filling, rather than reproducing altChunk blank pages. Render all final pages, inspect frames, logo/header, metadata grid, cell continuation, captions, table splits and evaluation signatures. Verify package preserved parts byte-for-byte and original SHA unchanged. Set updateFields true; Word will refresh page fields on open. Rendering PDFs remain QA-only; deliver DOCX only.
