from pathlib import Path
import re,json,hashlib
from collections import Counter
from docx import Document
from docx.shared import Pt,Cm,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from manuscript import BLOCKS
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/midterm/academic_text_2026-10-29';OUT.mkdir(parents=True,exist_ok=True)
doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Cm(2.5)
for name,size in [('Normal',12),('Title',16),('Heading 1',15),('Heading 2',14),('Heading 3',13),('Heading 4',12),('Caption',11)]:
 st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0);st.font.italic=False;st.font.underline=False;st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体')
 st.paragraph_format.space_after=Pt(6);st.paragraph_format.line_spacing=1.3
 if name.startswith('Heading')or name=='Title':st.font.bold=True;st.paragraph_format.keep_with_next=True
for node in list(doc.styles.element.iter(qn('w:pBdr'))):node.getparent().remove(node)
normal=doc.styles['Normal'];normal.paragraph_format.first_line_indent=Pt(24)
md=[]
for kind,text in BLOCKS:
 if kind.startswith('h'):
  level=int(kind[1:]);p=doc.add_paragraph(text,style='Title'if level==0 else'Heading '+str(level));p.paragraph_format.first_line_indent=Pt(0);md.append('#'*(level+1)+' '+text)
 elif kind=='eq':
  p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.2
  # Editable text with native Word subscript and superscript runs, no raster formula.
  i=0;buf=''
  def emit(s,mode=None):
   if not s:return
   r=p.add_run(s);r.font.name='Cambria Math';r.font.size=Pt(10 if mode else 11)
   if mode=='sub':r.font.subscript=True
   if mode=='sup':r.font.superscript=True
  while i<len(text):
   if text[i] in '_^' and i+1<len(text):
    emit(buf);buf='';mode='sub'if text[i]=='_'else'sup';i+=1
    if text[i]=='(':
     start=i+1;depth=1;i+=1
     while i<len(text)and depth:
      if text[i]=='(':depth+=1
      elif text[i]==')':depth-=1
      i+=1
     emit(text[start:i-1],mode)
    else:
     m=re.match(r'-?[A-Za-z0-9]+',text[i:])
     if m:emit(m.group(),mode);i+=len(m.group())
     else:emit(text[i],mode);i+=1
   else:buf+=text[i];i+=1
  emit(buf);md.append(text)
 else:
  p=doc.add_paragraph(text);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
  if kind=='ref':
   p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.15
   for r in p.runs:r.font.size=Pt(10.5)
  md.append(text)
# Plain blank-document delivery: no school frames, figures, running heads, or forced pages.
path=OUT/'中期报告_学术正文稿.docx';doc.core_properties.title='博弈驱动的去中心化语义共识机制及在影视推荐中的应用';doc.core_properties.author='马天成';doc.save(path)
(OUT/'中期报告_正文.md').write_text('\n\n'.join(md))
(OUT/'outline.txt').write_text('\n'.join(t for k,t in BLOCKS if k.startswith('h')))
s=''.join(t for _,t in BLOCKS);refs=set()
for k,t in BLOCKS:
 if k=='ref':continue
 for a in re.findall(r'\[([0-9,]+)\]',t):refs.update(map(int,a.split(',')))
assert refs==set(range(1,47)),refs
assert not any(a in s for a in ['开题主线的继承','开题对齐','修订说明','图1','图2'])
counts=Counter(t for k,t in BLOCKS if k=='p');assert not [t for t,c in counts.items()if c>1]
(OUT/'quality.json').write_text(json.dumps({'characters':len(s),'chinese_characters':len(re.findall('[\u4e00-\u9fff]',s)),'bibliography':46,'all_cited':True,'equations':sum(k=='eq'for k,_ in BLOCKS),'headings':sum(k.startswith('h')for k,_ in BLOCKS),'narrative_paragraphs':sum(k=='p'for k,_ in BLOCKS),'blank_document':True,'figures':0,'tables':0,'docx_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
print(path)
