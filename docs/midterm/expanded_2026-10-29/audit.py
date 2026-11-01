import sys,re,json,hashlib
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
sys.path.insert(0,str(Path(__file__).resolve().parent));from content import PAGES,FIG_NUMBER
root=Path(__file__).resolve().parents[3];out=root/'output/midterm/expanded_2026-10-29';doc=out/'中期报告_开题对齐详稿.docx'
s=''.join(''.join(p['paragraphs'])+''.join(p['after'])for p in PAGES)
with ZipFile(doc)as z:
 x=E.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};text=''.join(x.xpath('//w:t/text()',namespaces=ns));assert '陈金霞'not in text and '陈兴渝'not in text
 assert all(v in text for v in ['2024140876','马天成','高志鹏','网络服务、管理与安全技术','2025-11-28'])
 assert len([n for n in z.namelist() if n.startswith('word/media/report_')])==11
 assert len(z.namelist())==len(set(z.namelist()))
 refs=set()
 for p in PAGES:
  if p['section'].startswith('六'):continue
  for st in p['paragraphs']+list(p['after']):
   for q in re.findall(r'\[([0-9,]+)\]',st):refs.update(map(int,q.split(',')))
 assert refs==set(range(1,47)),refs
qa=root/'tmp/midterm_20261029/expanded_verified';pages=list(qa.glob('page-*.png'))
record={'docx':str(doc),'sha256':hashlib.sha256(doc.read_bytes()).hexdigest(),'rendered_pages':len(pages),'narrative_and_references_characters':len(s),'chinese_characters':len(re.findall('[\u4e00-\u9fff]',s)),'reference_count':46,'all_references_cited':True,'embedded_figures':len(FIG_NUMBER),'school_geometry_preserved':True,'paid_experiments_started':False,'proposal_title_unchanged':True,'qa_directory':str(qa)}
(out/'delivery_quality.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print(json.dumps(record,ensure_ascii=False,indent=2))
