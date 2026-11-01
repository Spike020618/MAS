"""Reconstruct project prose in the retained school Word package and page frames."""
import sys,json,hashlib,re,html
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile,ZIP_DEFLATED
from io import BytesIO
from lxml import etree as E
from docx import Document
from docx.shared import Cm
sys.path.insert(1,str(Path(__file__).resolve().parent.parent))
import build_report_v3 as template
from content import PAGES,FIG_NUMBER
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/midterm/expanded_2026-10-29';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'output/midterm/中期报告_学校Word模板重写稿.docx';FINAL=OUT/'中期报告_开题对齐详稿.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';NS=template.NS
with ZipFile(SOURCE)as z:parts={n:z.read(n)for n in z.namelist()}
x=E.fromstring(parts['word/document.xml']);body=x.find(W+'body');original=list(body);section=deepcopy(original[-1]);cover=[deepcopy(e)for e in original[:25]]
updates={11:'学    号:    2024140876',12:'姓    名:    马天成',13:'学 生 类 别: ［待填写］',14:'学    院:    计算机学院（国家示范性软件学院）',15:'学科/专业(代码):  计算机技术',16:'研究方向:    网络服务、管理与安全技术',17:'导师姓名:    高志鹏',18:'攻 读 学 位:  电子信息硕士'}
for i,value in updates.items():template.replace_value(cover[i],value)
for p in cover:
 txt=template.text_of(p)
 if '报告日期待填写'in txt:template.replace_value(p,'2026 年 10 月 29 日')
 if '硕士研究生学位论文中期报告'in txt:
  pp=p.find(W+'pPr')
  if pp is None:pp=E.Element(W+'pPr');p.insert(0,pp)
  for st in pp.findall(W+'pStyle'):pp.remove(st)
  pp.insert(0,template.el('pStyle',{'val':'Title'}))
for e in list(body):body.remove(e)
for e in cover:body.append(e)
rels=E.fromstring(parts['word/_rels/document.xml.rels']);ct=E.fromstring(parts['[Content_Types].xml']);styles=E.fromstring(parts['word/styles.xml'])
style=E.Element(W+'style');style.set(W+'type','paragraph');style.set(W+'styleId','Title');style.append(template.el('name',{'val':'Title'}));styles.append(style)
source_tables=E.fromstring(parts['word/document.xml']).findall('w:body/w:tbl',NS);imageparts={};figures=[];markdown=[]

def para(s,role='body'):
 return template.paragraph(s,role)

def report_math(s):
 p=para('','caption')
 def run(value,sub=False):
  r=template.el('r');rp=template.el('rPr');rp.append(template.el('rFonts',{'ascii':'Cambria Math','hAnsi':'Cambria Math','eastAsia':'宋体'}));rp.append(template.el('sz',{'val':'17'if sub else '21'}))
  if sub:rp.append(template.el('vertAlign',{'val':'subscript'}))
  r.append(rp);t=template.el('t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=value;r.append(t);p.append(r)
 for token in re.split(r'([A-Za-z]+_-?[A-Za-z]+)',s):
  if not token:continue
  if '_'in token:
   base,index=token.split('_',1);run(base);run('∅'if index=='empty'else index,True)
  else:run(token)
 return p

def picture(name):
 path=OUT/'figures'/f'{name}.png';number=int(name[3:5]);doc=Document();p=doc.add_paragraph();p.add_run().add_picture(str(path),width=Cm(15.1));node=deepcopy(p._p)
 pp=node.find(W+'pPr')
 if pp is None:pp=E.Element(W+'pPr');node.insert(0,pp)
 pp.append(template.el('jc',{'val':'center'}));pp.append(template.el('spacing',{'before':'100','after':'0','line':'240','lineRule':'auto'}));pp.append(template.el('keepNext'))
 for el in node.iter():
  if E.QName(el).localname=='blip':el.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed',f'rIdReportFigure{number}')
  if E.QName(el).localname in ('docPr','cNvPr'):
   el.set('id',str(1000+number));el.set('name',name)
   if E.QName(el).localname=='docPr':el.set('descr',titles[name])
 r=E.Element('{http://schemas.openxmlformats.org/package/2006/relationships}Relationship');r.set('Id',f'rIdReportFigure{number}');r.set('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/image');r.set('Target',f'media/report_{name}.png');rels.append(r);imageparts[f'word/media/report_{name}.png']=path.read_bytes();figures.append(name)
 return node

manifest=json.loads((OUT/'figures_manifest.json').read_text());titles={f['id']:f['title']for f in manifest};titles['fig08_workspace']='影视检索与证据工作台界面'
for i,page in enumerate(PAGES):
 nodes=[]
 if page['title']:nodes.append(para(page['title'],'heading'))
 for text in page['paragraphs']:nodes.append(para(text,'ref'if page['section'].startswith('六、')else'body'))
 if page['figure']:
  name=page['figure'];nodes.append(picture(name));nodes.append(para(f'图{FIG_NUMBER[name]}  {titles[name]}','caption'))
 for text in page['equations']:nodes.append(report_math(text.replace('Yₑբ','Yₑf').replace('Yբ','Yf')))
 for text in page['after']:nodes.append(para(text,'heading'if re.match(r'^[四五六]、',text)else'body'))
 frame=template.frame(nodes,first=i==0);rows=frame.findall(W+'tr')
 if i==0:
  oldrows=source_tables[1].findall(W+'tr')
  for j in range(6):frame.replace(rows[j],deepcopy(oldrows[j]))
  rows=frame.findall(W+'tr')
  # Metadata confirmed by the user's opening proposal.
  for j,value in [(0,'博弈驱动的去中心化语义共识机制及在影视推荐中的应用'),(1,'应用研究'),(2,'2025-11-28')]:
   cs=rows[j].findall(W+'tc')
   if len(cs)>1:template.replace_value(cs[1].find(W+'p'),value)
  cs=rows[1].findall(W+'tc')
  template.replace_value(cs[3].find(W+'p'),'国家重点研发计划项目')
  cs=rows[2].findall(W+'tc')
  template.replace_value(cs[3].find(W+'p'),'是')
  cells=rows[3].findall(W+'tc')
  if len(cells)>3:template.replace_value(cells[3].find(W+'p'),'2026-10-29')
 c=rows[-1].find(W+'tc');c.insert(1 if c.find(W+'tcPr')is not None else 0,para(page['section'],'section'))
 if i>0:body.append(template.page_break())
 body.append(frame)
 markdown+=['# '+page['section'],'## '+page['title']if page['title']else'',*page['paragraphs']]
 if page['figure']:markdown+=['![图'+str(FIG_NUMBER[page['figure']])+' '+titles[page['figure']]+'](figures/'+page['figure']+'.png)']
 markdown+=list(page['equations'])+list(page['after'])+['']
# Preserve the actual school assessment form without filling another person's comments.
body.append(template.page_break());mentor=deepcopy(template.tables[5]);rows=mentor.findall(W+'tr')
for index in (1,4):
 cell=rows[index].find(W+'tc');template.clear_content(cell,{'tcPr'});cell.append(para('（由导师填写）'if index==1 else '（由学院组织考核后填写）'))
for i,row in enumerate(rows):
 pr=row.find(W+'trPr')
 if pr is None:pr=E.Element(W+'trPr');row.insert(0,pr)
 for h in pr.findall(W+'trHeight'):pr.remove(h)
 pr.append(template.el('trHeight',{'val':[650,4400,1000,650,4000][i],'hRule':'atLeast'}))
body.append(mentor);body.append(section)
if not any(n.get('Extension')=='png'for n in ct):
 n=E.SubElement(ct,'{http://schemas.openxmlformats.org/package/2006/content-types}Default');n.set('Extension','png');n.set('ContentType','image/png')
edits={'word/document.xml':E.tostring(x,xml_declaration=True,encoding='UTF-8',standalone=True),'word/_rels/document.xml.rels':E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True),'word/styles.xml':E.tostring(styles,xml_declaration=True,encoding='UTF-8',standalone=True),'[Content_Types].xml':E.tostring(ct,xml_declaration=True,encoding='UTF-8',standalone=True)}
with ZipFile(FINAL,'w',ZIP_DEFLATED)as z:
 for name,blob in parts.items():z.writestr(name,edits.get(name,blob))
 for name,blob in imageparts.items():z.writestr(name,blob)
with ZipFile(FINAL)as z:
 preserved=[n for n in parts if n not in edits];assert all(z.read(n)==parts[n]for n in preserved)
 root=E.fromstring(z.read('word/document.xml'));assert E.tostring(root.find('w:body/w:sectPr',NS))==E.tostring(section)
(OUT/'中期报告_正文.md').write_text('\n\n'.join(s for s in markdown if s))
(OUT/'content.json').write_text(json.dumps(PAGES,ensure_ascii=False,indent=2))
(OUT/'template_verification.json').write_text(json.dumps({'reference':'/Users/spike/Downloads/LwMidReport.docx','reference_sha256':hashlib.sha256(Path('/Users/spike/Downloads/LwMidReport.docx').read_bytes()).hexdigest(),'working_reference':str(SOURCE),'working_reference_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'preserved_parts':preserved,'edited_parts':list(edits),'new_parts':list(imageparts),'section_geometry_unchanged':True,'content_pages':len(PAGES),'figures':figures},ensure_ascii=False,indent=2))
print(FINAL)
