from pathlib import Path
import re,json,hashlib
from collections import Counter
from docx import Document
from docx.shared import Pt,Cm,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION_START,WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from academic_revision import BLOCKS,REFS
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/midterm/academic_text_2026-10-29';OUT.mkdir(parents=True,exist_ok=True)
doc=Document()
def page(sec,landscape=False):
 sec.orientation=WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
 sec.page_width=Cm(29.7 if landscape else 21);sec.page_height=Cm(21 if landscape else 29.7)
 sec.left_margin=sec.right_margin=Cm(2.3 if landscape else 2.5)
 sec.top_margin=sec.bottom_margin=Cm(2)
page(doc.sections[0])
for name,size in [('Normal',11),('Title',16),('Heading 1',15),('Heading 2',13),('Heading 3',12),('Heading 4',11),('Caption',10)]:
 st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
 st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Songti SC')
 st.paragraph_format.line_spacing=1.35;st.paragraph_format.space_after=Pt(6)
 if name.startswith('Heading') or name=='Title':st.font.bold=True;st.paragraph_format.keep_with_next=True
doc.styles['Normal'].paragraph_format.first_line_indent=Pt(22)
for node in list(doc.styles.element.iter(qn('w:pBdr'))):node.getparent().remove(node)
footer=doc.sections[0].footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
def math_nodes(text,parent):
 i=0
 while i<len(text):
  token=text[i];i+=1
  # Keep an operator together when it carries a lower index.
  if text[i-1:i+2]=='max':token='max';i+=2
  base=OxmlElement('m:r');t=OxmlElement('m:t');t.text=token;base.append(t)
  if i<len(text) and text[i]=='\u0302':
   i+=1;accent=OxmlElement('m:acc');pr=OxmlElement('m:accPr');mark=OxmlElement('m:chr');mark.set(qn('m:val'),'̂');pr.append(mark);accent.append(pr);e=OxmlElement('m:e');e.append(base);accent.append(e);base=accent
  while i<len(text) and text[i] in '_^':
   mode=text[i];i+=1
   if i<len(text) and text[i]=='{':
    i+=1;start=i;depth=1
    while i<len(text) and depth:
     if text[i]=='{':depth+=1
     if text[i]=='}':depth-=1
     i+=1
    sub=text[start:i-1]
   else:
    match=re.match(r'[A-Za-z0-9]+',text[i:]);sub=match.group() if match else text[i:i+1];i+=len(sub)
   outer=OxmlElement('m:sSub' if mode=='_' else 'm:sSup');e=OxmlElement('m:e');e.append(base);outer.append(e)
   script=OxmlElement('m:sub' if mode=='_' else 'm:sup');math_nodes(sub,script);outer.append(script);base=outer
  parent.append(base)
md=[];number=0
for kind,text in BLOCKS:
 if kind.startswith('h'):
  level=int(kind[1:]);p=doc.add_paragraph(text,style='Title' if level==0 else 'Heading '+str(level));p.paragraph_format.first_line_indent=Pt(0);md.append('#'*(level+1)+' '+text)
 elif kind=='eq':
  number+=1;p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.keep_together=True
  math=OxmlElement('m:oMath');math_nodes(text,math);p._p.append(math);p.add_run('  ('+str(number)+')');md.append(text+'  ('+str(number)+')')
 elif kind=='figure':
  p=doc.add_paragraph();p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.keep_with_next=True;p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  p.add_run().add_picture(str(ROOT/'docs/figures/current-method'/text['name']),width=Cm(16))
  p=doc.add_paragraph(text['caption'],style='Caption');p.paragraph_format.first_line_indent=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  md.append('!['+text['caption']+'](../../../docs/figures/current-method/'+text['name']+')')
 elif kind=='table':
  p=doc.add_paragraph(text['caption'],style='Caption');p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.keep_with_next=True
  tb=doc.add_table(rows=1,cols=len(text['headers']));tb.autofit=False
  widths=[1.4,14.6] if len(text['headers'])==2 else [3.4,6.3,6.3]
  if len(widths)!=len(text['headers']):widths=[16/len(text['headers'])]*len(text['headers'])
  for col,w in zip(tb.columns,widths):col.width=Cm(w)
  for ri,row in enumerate([text['headers']]+text['rows']):
   cells=tb.rows[0].cells if ri==0 else tb.add_row().cells
   trpr=cells[0]._tc.getparent().get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
   if ri==0:trpr.append(OxmlElement('w:tblHeader'))
   for ci,(cell,value) in enumerate(zip(cells,row)):
    cell.width=Cm(widths[ci]);cell.text=str(value)
    pr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
    for edge in ['top','left','bottom','right']:
     n=OxmlElement('w:'+edge);n.set(qn('w:val'),'single');n.set(qn('w:sz'),'4');n.set(qn('w:color'),'D9D9D9');borders.append(n)
    pr.append(borders)
    if ri==0:
     sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9EDF2');pr.append(sh)
    for p in cell.paragraphs:
     p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.15;p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4)
     for r in p.runs:r.font.size=Pt(10);r.font.bold=ri==0
     if text['caption'].startswith('算法') and ri<len(text['rows']):p.paragraph_format.keep_with_next=True
  md.append(text['caption']+'\n\n|'+'|'.join(text['headers'])+'|\n|'+'|'.join(['---']*len(text['headers']))+'|\n'+'\n'.join('|'+'|'.join(map(str,row))+'|' for row in text['rows']))
 else:
  p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
  for part in re.split(r'([CVdHJv]_t|Q_i\(S\)|I_e\(a\)|J_ef\(a\)|J_ef)',text):
   if re.fullmatch(r'[CVdHJv]_t|Q_i\(S\)|I_e\(a\)|J_ef\(a\)|J_ef',part):
    math=OxmlElement('m:oMath');math_nodes(part,math);p._p.append(math)
   else:p.add_run(part)
  if kind=='ref':
   p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.15
   for r in p.runs:r.font.size=Pt(10)
  md.append(text)
path=OUT/'中期报告_学术正文稿.docx'
doc.core_properties.title=BLOCKS[0][1];doc.core_properties.author='马天成';doc.save(path)
(OUT/'中期报告_正文.md').write_text('\n\n'.join(md))
(OUT/'outline.txt').write_text('\n'.join(t for k,t in BLOCKS if k.startswith('h')))
body='\n'.join(t if isinstance(t,str) else json.dumps(t,ensure_ascii=False) for _,t in BLOCKS)
cited=set()
for k,t in BLOCKS:
 if k=='ref':continue
 for a in re.findall(r'\[([0-9,]+)\]',str(t)):cited.update(map(int,a.split(',')))
assert cited==set(range(1,len(REFS)+1)),cited
assert not [t for t,c in Counter(t for k,t in BLOCKS if k=='p').items() if c>1]
(OUT/'quality.json').write_text(json.dumps(dict(characters=len(body),chinese_characters=len(re.findall('[\u4e00-\u9fff]',body)),bibliography=len(REFS),all_cited=True,equations=number,figures=sum(k=='figure' for k,_ in BLOCKS),tables=sum(k=='table' for k,_ in BLOCKS),docx_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
print(path)
