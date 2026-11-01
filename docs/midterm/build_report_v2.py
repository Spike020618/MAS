from pathlib import Path
import json
import re
import sys

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from report_v2_supplements import EXTRA

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/midterm/中期报告_按模板重制_Word.docx'
SRC = Path(__file__).with_name('report_v2.txt')
BODY_FONT = 'STSong'
doc = Document()
fig_no = 0
table_no = 0


def font(r, size=10.5, bold=False):
    r.font.name = BODY_FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = RGBColor(0, 0, 0)
    rf = r._element.get_or_add_rPr().rFonts
    for key in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        rf.set(qn('w:'+key), BODY_FONT)


def p(text='', size=10.5, bold=False, align=None, indent=True, before=0, after=4, leading=17):
    x = doc.add_paragraph()
    fmt = x.paragraph_format
    fmt.line_spacing = Pt(leading)
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.first_line_indent = Pt(21) if indent else Pt(0)
    fmt.widow_control = True
    if align is not None:
        x.alignment = align
    else:
        x.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    font(x.add_run(text), size, bold)
    return x


def heading(text, level):
    x = p(text, size=12 if level==1 else 10.5, bold=True, indent=False,
          before=7 if level==1 else 4, after=5, leading=18)
    x.style = doc.styles['Heading '+str(min(level,3))]
    x.paragraph_format.keep_with_next=True


def border(table, color='000000', size='4'):
    pr=table._tbl.tblPr
    old=pr.find(qn('w:tblBorders'))
    if old is not None: pr.remove(old)
    b=OxmlElement('w:tblBorders')
    for key in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+key); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),size); e.set(qn('w:color'),color); b.append(e)
    pr.append(b)


def make_table(headers, rows, widths=None, size=9.1, title=None):
    global table_no
    if title:
        table_no+=1
        c=p(f'表{table_no}  {title}',size=10,align=WD_ALIGN_PARAGRAPH.CENTER,indent=False,after=5,leading=15)
        c.paragraph_format.keep_with_next=True
    t=doc.add_table(rows=1,cols=len(headers))
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    if widths is None:
        n=len(headers)
        widths={2:[4.1,11.7],3:[3.3,6.7,5.8],4:[4.9,3.7,3.7,3.5],5:[5.2,2.5,2.7,2.7,2.7]}.get(n,[15.8/n]*n)
    for i,rowvals in enumerate([headers]+rows):
        row=t.rows[0] if i==0 else t.add_row()
        trpr=row._tr.get_or_add_trPr()
        cant=OxmlElement('w:cantSplit');trpr.append(cant)
        if i==0:
            rep=OxmlElement('w:tblHeader');trpr.append(rep)
        for j,text in enumerate(rowvals):
            cell=row.cells[j];cell.width=Cm(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for k,v in [('top','65'),('bottom','65'),('left','80'),('right','80')]:
                e=OxmlElement('w:'+k);e.set(qn('w:w'),v);e.set(qn('w:type'),'dxa');margins.append(e)
            tcpr.append(margins)
            x=cell.paragraphs[0]
            x.alignment=WD_ALIGN_PARAGRAPH.CENTER if i==0 or j==0 else WD_ALIGN_PARAGRAPH.LEFT
            x.paragraph_format.line_spacing=Pt(13)
            x.paragraph_format.space_after=Pt(0)
            font(x.add_run(text),size,i==0)
    border(t)
    p('',size=2,indent=False,after=0,leading=3)
    return t


def figure(title, desc, height):
    global fig_no
    fig_no+=1
    # All flow figures remain native Word tables and text; screenshots remain honest placeholders.
    is_placeholder= any(k in title for k in ['待补','截图','柱状图','对比图','重绘'])
    if not is_placeholder:
        nodes=[x.strip(' 。') for x in re.split('→|；',desc) if x.strip()]
        t=doc.add_table(rows=0,cols=1);t.alignment=WD_TABLE_ALIGNMENT.CENTER
        for i,node in enumerate(nodes):
            c=t.add_row().cells[0];c.width=Cm(13.8)
            x=c.paragraphs[0];x.alignment=WD_ALIGN_PARAGRAPH.CENTER
            x.paragraph_format.line_spacing=Pt(13)
            x.paragraph_format.space_before=Pt(4);x.paragraph_format.space_after=Pt(4)
            font(x.add_run(node),9.1)
        border(t,'808080')
    else:
        x=p('［图位预留］',size=9,align=WD_ALIGN_PARAGRAPH.CENTER,indent=False,after=1,leading=13)
        x.paragraph_format.space_before=Pt(8)
        p(desc,size=9,align=WD_ALIGN_PARAGRAPH.CENTER,indent=False,after=3,leading=14)
        blank=p('',size=1,indent=False,after=0,leading=1)
        blank.paragraph_format.space_after=Cm(max(1.0,min(float(height),5.2)-1.4))
    p(f'图{fig_no}  {title}',size=9.5,align=WD_ALIGN_PARAGRAPH.CENTER,indent=False,after=6,leading=14)


def setup():
    for sn in ['Normal','Title','Heading 1','Heading 2','Heading 3']:
        st=doc.styles[sn];st.font.name=BODY_FONT;st.font.color.rgb=RGBColor(0,0,0)
        st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),BODY_FONT)
        pp=st._element.get_or_add_pPr()
        for tag in ['pBdr','shd']:
            node=pp.find(qn('w:'+tag))
            if node is not None:pp.remove(node)
    s=doc.sections[0]
    s.page_width=Cm(21);s.page_height=Cm(29.7)
    s.top_margin=Cm(2.5);s.bottom_margin=Cm(2.2);s.left_margin=Cm(2.5);s.right_margin=Cm(2.1)
    s.header_distance=Cm(1.45);s.footer_distance=Cm(1.25)
    doc.core_properties.title='硕士研究生学位论文中期报告'
    doc.core_properties.subject='博弈驱动的去中心化语义共识机制及在影视推荐中的应用'
    doc.core_properties.author='马天成'


def cover():
    p('',indent=False,after=65)
    x=p('硕士研究生学位论文中期报告',size=22,bold=True,align=WD_ALIGN_PARAGRAPH.CENTER,indent=False,after=76,leading=30)
    x.style=doc.styles['Title']
    fields=[('学        号','［待填写］'),('姓        名','马天成'),('学生类别','［待填写］'),('学        院','［待填写］'),('学科/专业（代码）','［待填写］'),('研究方向','多智能体语义共识与影视推荐'),('导师姓名','高志鹏'),('攻读学位','［待填写］')]
    for a,b in fields:
        p(f'{a}：    {b}',size=12,indent=False,after=10,leading=19).paragraph_format.left_indent=Cm(3.0)
    p('北京邮电大学',size=12,bold=True,align=WD_ALIGN_PARAGRAPH.CENTER,indent=False,before=42,after=8)
    p('［报告日期待填写］',size=11,align=WD_ALIGN_PARAGRAPH.CENTER,indent=False)
    doc.add_page_break()
    p('',indent=False)
    doc.add_page_break()
    s=doc.add_section(WD_SECTION.CONTINUOUS)
    s.header.is_linked_to_previous=False;s.footer.is_linked_to_previous=False
    hp=s.header.paragraphs[0];hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    font(hp.add_run('北京邮电大学硕士研究生学位论文中期报告'),9)
    pg=OxmlElement('w:pgNumType');pg.set(qn('w:start'),'1');s._sectPr.append(pg)
    fp=s.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    font(fp.add_run('－'),9)
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');fp._p.append(fld)
    font(fp.add_run('－'),9)


def metadata():
    rows=[['论文题目','博弈驱动的去中心化语义共识机制及在影视推荐中的应用','',''],['论文类型','应用研究［待核对］','选题来源','［待填写］'],['开题日期','［待填写］','是否开题题目','［待核对］'],['论文开始日期','［待填写］','报告日期','［待填写］'],['报告地点','［待填写］','报告时间','［待填写］'],['已提交月报次数','［待填写］','已提交年报次数','［待填写］']]
    t=doc.add_table(rows=0,cols=4);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=[3.0,5.1,3.0,4.5]
    for j,w in enumerate(widths):t.columns[j].width=Cm(w)
    for i,values in enumerate(rows):
        row=t.add_row()
        if i==0:row.cells[1].merge(row.cells[3])
        for j,txt in enumerate(values):
            if i==0 and j>1:continue
            row.cells[j].width=Cm(widths[j] if i else (3.0 if j==0 else 12.6))
            x=row.cells[j].paragraphs[0];x.paragraph_format.line_spacing=Pt(14);font(x.add_run(txt),9.5,j==0)
    p('',size=2,leading=3,after=3)


def references():
    # Bibliographic entries are drawn from the project's retained bibliography, not invented metadata.
    source=(ROOT/'docs/papers/en/references.bib').read_text()
    wanted=['wang2023selfconsistency','du2024improving','zhang2024collaboration','duetting2023mechanism','saig2024incentivizing','sun2025gametheory']
    entries={}
    starts=list(re.finditer(r'@\w+\{([^,]+),',source))
    for i,m in enumerate(starts):
        chunk=source[m.end():starts[i+1].start() if i+1<len(starts) else len(source)]
        fields={}
        for k in ['author','title','journal','booktitle','year','pages','doi','url']:
            z=re.search(r'\b'+k+r'\s*=\s*\{',chunk)
            if not z:continue
            depth=1;j=z.end();end=j
            while end<len(chunk) and depth:
                if chunk[end]=='{':depth+=1
                elif chunk[end]=='}':depth-=1
                end+=1
            val=chunk[j:end-1].replace('{','').replace('}','').replace('\\_','_').replace('D\\"utting','Dütting')
            fields[k]=val
        entries[m.group(1)]=fields
    rest=['xiong2023interconsistency','han2024llm','tran2025multiagent','pitre2025consensagent','wu2025disagreement']
    picked=wanted+rest
    for key,fields in entries.items():
        if len(picked)>=20:break
        if key not in picked and any(w in fields.get('title','').lower() for w in ['redial','movielens','react:','chain-of-thought','truthfulqa','strategyqa','measuring massive','potential games','multiagent systems']):picked.append(key)
    for i,key in enumerate(picked,1):
        f=entries.get(key)
        if not f:continue
        authors=f.get('author','').split(' and ')
        a=', '.join(authors[:3])+(', et al' if len(authors)>3 else '')
        typ='[C]' if f.get('booktitle') else '[J]'
        text=f'[{i}] {a}. {f.get("title",key)}{typ}. {f.get("booktitle",f.get("journal",""))}, {f.get("year","")}.'
        if f.get('doi'):text+=' DOI: '+f['doi']+'.'
        x=p(text,size=8.8,indent=False,after=4,leading=12)
    p('说明：文献条目依据项目现有参考文献库整理；最终送审前按学校格式核对卷期、页码与标识。',size=9,indent=False,leading=13)


def build():
    setup();cover()
    raw=SRC.read_text()
    chunks={int(a):b.strip().splitlines() for a,b in re.findall(r'@@(\d+)\n(.*?)(?=\n@@|\Z)',raw,re.S)}
    # The original's 42-page envelope includes a blank verso. Consolidate related frontend content.
    audit=chunks.pop(34)
    audit=[x for x in audit if not x.startswith('!')]
    chunks[33].extend(audit)
    consumed=[]
    for idx,(num,lines) in enumerate(chunks.items()):
        if idx:doc.add_page_break()
        if num==1:metadata()
        # Add technical expansions as connected prose; no source file directives enter the report.
        extras=EXTRA.get(num,[])
        insert=len(lines)
        for j in range(len(lines)-1,-1,-1):
            if lines[j].startswith('!'):insert=j;break
        lines=lines[:insert]+extras+lines[insert:]
        if num==33:lines.extend(EXTRA.get(34,[]))
        if num==33:
            lines=[re.sub(r'\|[\d.]+$','|3.0',l) if l.startswith('!') else l for l in lines]
        j=0
        while j<len(lines):
            line=lines[j].strip();j+=1
            if not line:continue
            if line.startswith('### '):heading(line[4:],3)
            elif line.startswith('## '):heading(line[3:],2)
            elif line.startswith('# '):heading(line[2:],1)
            elif line.startswith('!'):
                title,desc,height=line[1:].split('|');figure(title,desc,height)
            elif line.startswith('T') and '|' in line:
                parts=line[1:].split('|');rows=[]
                while j<len(lines) and '|' in lines[j] and not lines[j].startswith('!'):
                    rows.append(lines[j].split('|'));j+=1
                make_table(parts[1:],rows,title=parts[0])
            elif line=='REF':references()
            else:p(line)
        consumed.append({'source_block':num,'body_chars':sum(len(x) for x in lines)})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc.save(OUT)
    (ROOT/'tmp/midterm_v2_layout.json').write_text(json.dumps(consumed,ensure_ascii=False,indent=2))
    print(OUT)
    print('Designed total pages:',2+len(chunks),'Figures:',fig_no,'Tables:',table_no)
    print('Text chars:',sum(len(x.text) for x in doc.paragraphs)+sum(len(c.text) for t in doc.tables for r in t.rows for c in r.cells))


if __name__=='__main__':build()
