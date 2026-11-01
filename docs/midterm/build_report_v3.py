from copy import deepcopy
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import re

from lxml import etree as E

from report_v2_supplements import EXTRA
from rewrite_v3 import rewrite

ROOT=Path(__file__).resolve().parents[2]
REF=Path('/Users/spike/Downloads/LwMidReport.docx')
OUT=ROOT/'output/midterm/中期报告_学校Word模板重写稿.docx'
TMP=ROOT/'tmp/midterm_v3'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W='{'+NS['w']+'}'
fig=0;tab=0
z=ZipFile(REF)
parts={n:z.read(n) for n in z.namelist()}
root=E.fromstring(parts['word/document.xml'])
body=root.find('w:body',NS)
original=list(body)
tables=body.findall('w:tbl',NS)
heading_pattern=deepcopy(tables[1].findall('w:tr',NS)[6].find('w:tc/w:p',NS))
emb=ZipFile(BytesIO(parts['word/afchunk2.docx']))
eroot=E.fromstring(emb.read('word/document.xml'))
body_pattern=next(p for p in eroot.findall('.//w:p',NS) if len(''.join(p.xpath('.//w:t/text()',namespaces=NS)))>150)


def el(name,attrs=None):
    x=E.Element(W+name)
    for k,v in (attrs or {}).items():x.set(W+k,str(v))
    return x


def text_of(x):return ''.join(x.xpath('.//w:t/text()',namespaces=NS))


def clear_content(x,keep):
    for c in list(x):
        if E.QName(c).localname not in keep:x.remove(c)


def paragraph(text='',role='body',gap=None):
    p=deepcopy(heading_pattern if role=='section' else body_pattern)
    pp=p.find('w:pPr',NS)
    if pp is None:pp=el('pPr');p.insert(0,pp)
    clear_content(p,{'pPr'})
    for node in pp.findall('w:rPr',NS)+pp.findall('w:numPr',NS)+pp.findall('w:pStyle',NS)+pp.findall('w:pageBreakBefore',NS):pp.remove(node)
    ind=pp.find('w:ind',NS)
    if ind is None:ind=el('ind');pp.append(ind)
    for k in list(ind.attrib):del ind.attrib[k]
    ind.set(W+'left','200');ind.set(W+'right','200');ind.set(W+'firstLine','420' if role=='body' else '0')
    sp=pp.find('w:spacing',NS)
    if sp is None:sp=el('spacing');pp.append(sp)
    for k in list(sp.attrib):del sp.attrib[k]
    sp.set(W+'line','312');sp.set(W+'lineRule','exact');sp.set(W+'after','65')
    jc=pp.find('w:jc',NS)
    if jc is None:jc=el('jc');pp.append(jc)
    jc.set(W+'val','center' if role in ('caption','placeholder','flow') else 'both')
    if role=='ref':
        sp.set(W+'line','240');sp.set(W+'after','70')
        ind.set(W+'firstLine','0')
    if role in ('section','heading'):
        pp.append(el('keepNext'));sp.set(W+'before','120');sp.set(W+'after','100')
    if gap is not None:sp.set(W+'after',str(gap))
    r=el('r');rp=el('rPr');rp.append(el('rFonts',{'ascii':'宋体','hAnsi':'宋体','eastAsia':'宋体','cs':'宋体'}))
    rp.append(el('sz',{'val':'18' if role=='ref' else '20' if role in ('caption','placeholder','flow') else '21'}))
    rp.append(el('szCs',{'val':'21'}))
    if role in ('section','heading'):rp.append(el('b'))
    r.append(rp);t=el('t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=text;r.append(t);p.append(r)
    return p


def data_table(headers,rows,title):
    global tab
    tab+=1
    n=len(headers);width=9000
    weights={2:[.25,.75],3:[.22,.4,.38],4:[.34,.22,.22,.22]}.get(n,[1/n]*n)
    t=el('tbl');pr=el('tblPr');pr.append(el('tblW',{'w':width,'type':'dxa'}));pr.append(el('jc',{'val':'center'}))
    pr.append(el('tblLayout',{'type':'fixed'}))
    borders=el('tblBorders')
    for k in ['top','left','bottom','right','insideH','insideV']:borders.append(el(k,{'val':'single','sz':'4','color':'000000'}))
    pr.append(borders);mar=el('tblCellMar')
    for k in ['top','bottom','left','right']:mar.append(el(k,{'w':'75','type':'dxa'}))
    pr.append(mar);t.append(pr);grid=el('tblGrid')
    for v in weights:grid.append(el('gridCol',{'w':round(width*v)}))
    t.append(grid)
    for i,values in enumerate([headers]+rows):
        tr=el('tr');trpr=el('trPr');trpr.append(el('cantSplit'))
        if i==0:trpr.append(el('tblHeader'))
        tr.append(trpr)
        for j,value in enumerate(values):
            c=el('tc');cp=el('tcPr');cp.append(el('tcW',{'w':round(width*weights[j]),'type':'dxa'}));cp.append(el('vAlign',{'val':'center'}));c.append(cp)
            p=paragraph(value,'flow',0);pp=p.find('w:pPr',NS)
            ind=pp.find('w:ind',NS);ind.set(W+'left','0');ind.set(W+'right','0')
            pp.find('w:jc',NS).set(W+'val','center' if i==0 or j==0 else 'left')
            pp.find('w:spacing',NS).set(W+'line','265')
            for sz in p.findall('.//w:sz',NS):sz.set(W+'val','18')
            if i==0:p.find('w:r/w:rPr',NS).append(el('b'))
            c.append(p);tr.append(c)
        t.append(tr)
    return [paragraph(f'表{tab}  {title}','caption'),t,paragraph('','caption',45)]


def figure(title,desc,height):
    global fig
    fig+=1
    original_title=title
    title=title.replace('按repo重绘','架构图待补').replace('待按复核口径重绘','待补图').replace('（待补真实运行界面）','（界面图待补）')
    out=[]
    placeholder=any(s in original_title for s in ['待补','截图','柱状图','对比图','重绘'])
    if placeholder:
        out.append(paragraph('［此处预留'+re.sub(r'（.*?）','',title)+'］','placeholder'))
        # Leave a physical image slot, rather than prose instructions presented as report content.
        out.append(paragraph('','placeholder',int(min(max(float(height),2.5),3.7)*567)))
    else:
        nodes=[x.strip(' 。') for x in re.split('→|；',desc) if x.strip()]
        out.append(paragraph('  →  '.join(nodes),'flow'))
        out.append(paragraph('','placeholder',max(300,int(float(height)*300))))
    out.append(paragraph(f'图{fig}  {title}','caption'))
    return out


def native_math(text):
    ns='http://schemas.openxmlformats.org/officeDocument/2006/math'
    p=paragraph('','caption');om=E.SubElement(p,'{'+ns+'}oMathPara');o=E.SubElement(om,'{'+ns+'}oMath');r=E.SubElement(o,'{'+ns+'}r');t=E.SubElement(r,'{'+ns+'}t');t.text=text
    return p


def render_lines(lines):
    out=[];i=0
    while i<len(lines):
        x=lines[i].strip();i+=1
        if not x:continue
        if x.startswith('# '):out.append(paragraph(x[2:],'section'))
        elif x.startswith('## '):out.append(paragraph(x[3:],'heading'))
        elif x.startswith('### '):out.append(paragraph(x[4:],'heading'))
        elif x.startswith('!'):
            title,desc,h=x[1:].split('|');out.extend(figure(title,desc,h))
        elif x.startswith('T') and '|' in x:
            vals=x[1:].split('|');rows=[]
            while i<len(lines) and '|' in lines[i] and not lines[i].startswith('!'):
                rows.append(lines[i].split('|'));i+=1
            out.extend(data_table(vals[1:],rows,vals[0]))
        elif x=='REF':
            old=E.fromstring(ZipFile(ROOT/'output/midterm/中期报告_按模板重制_Word.docx').read('word/document.xml'))
            for p in old.findall('./w:body/w:p',NS):
                txt=text_of(p)
                if re.match(r'^\[\d+\]',txt):out.append(paragraph(txt,'ref'))
        else:
            x=rewrite(x)
            if x:out.append(paragraph(x))
    return out


def replace_value(p,text):
    ts=p.findall('.//w:t',NS)
    if ts:
        ts[0].text=text
        for t in ts[1:]:t.text=''
    else:p.append(paragraph(text)[-1])


def page_break():
    p=paragraph('','caption',0);pp=p.find('w:pPr',NS);pp.find('w:spacing',NS).set(W+'line','20')
    r=el('r');r.append(el('br',{'type':'page'}));p.append(r);return p


def frame(content,first=False):
    t=deepcopy(tables[1])
    rows=t.findall('w:tr',NS)
    if not first:
        for r in rows[:6]:t.remove(r)
    row=t.findall('w:tr',NS)[-1]
    rp=row.find('w:trPr',NS)
    if rp is None:rp=el('trPr');row.insert(0,rp)
    for a in rp.findall('w:cantSplit',NS)+rp.findall('w:trHeight',NS):rp.remove(a)
    rp.append(el('trHeight',{'val':'9250' if first else '13200','hRule':'atLeast'}))
    c=row.find('w:tc',NS);clear_content(c,{'tcPr'})
    for x in content:c.append(x)
    if not content or E.QName(c[-1]).localname!='p':c.append(paragraph(''))
    return t


def build():
    assert hashlib.sha256(REF.read_bytes()).hexdigest()=='c8fcea4dce4384fafbbe83d0411f213cb11b5f0574162c20ecd87d8c398aa738'
    # Fill the source cover and keep its actual layout components.
    cover=[deepcopy(x) for x in original[:25]]
    barcode=cover[0].findall('w:tr/w:tc',NS)[1];clear_content(barcode,{'tcPr'});barcode.append(paragraph('［学号条码待补］','placeholder'))
    updates={11:'学    号:    ［待填写］',12:'姓    名:    马天成',13:'学 生 类 别: ［待填写］',14:'学    院:    ［待填写］',15:'学科/专业(代码):  ［待填写］',16:'研究方向:    多智能体语义共识与影视推荐',17:'导师姓名:    高志鹏',18:'攻 读 学 位: ［待填写］',23:'［报告日期待填写］'}
    for i,txt in updates.items():replace_value(cover[i],txt)
    # Source cover empty spacers caused university/date to spill onto the next page.
    for i in [4,5,6,7,8,9,10,19,20,21]:
        pp=cover[i].find('w:pPr',NS)
        if pp is None:pp=el('pPr');cover[i].insert(0,pp)
        for sp in pp.findall('w:spacing',NS):pp.remove(sp)
        pp.append(el('spacing',{'line':'150','lineRule':'exact','after':'0','before':'0'}))
    for i,txt in [(0,'博弈驱动的去中心化语义共识机制及在影视推荐中的应用'),(1,'应用研究［待核对］'),(2,'［待填写］'),(3,'［待填写］'),(4,'［待填写］'),(5,'［待填写］')]:
        cells=tables[1].findall('w:tr',NS)[i].findall('w:tc',NS)
        replace_value(cells[1].find('w:p',NS),txt)
        if len(cells)>2:replace_value(cells[3].find('w:p',NS),'［待填写］')
    for x in list(body):body.remove(x)
    for x in cover:body.append(x)
    raw=(Path(__file__).with_name('report_v2.txt')).read_text()
    chunks={int(n):b.strip().splitlines() for n,b in re.findall(r'@@(\d+)\n(.*?)(?=\n@@|\Z)',raw,re.S)}
    audit=chunks.pop(34);chunks[33].extend([x for x in audit if not x.startswith('!')])
    first=True
    for num,lines in chunks.items():
        if num==41:
            body.append(page_break())
            ev=deepcopy(tables[5])
            erows=ev.findall('w:tr',NS)
            replace_value(erows[1].find('w:tc/w:p',NS),'［由导师审阅后填写。］')
            for j,row in enumerate(erows):
                rp=row.find('w:trPr',NS)
                if rp is not None:
                    for h in rp.findall('w:trHeight',NS):h.set(W+'val',str([650,4700,1100,650,4200][j]));h.set(W+'hRule','atLeast')
            body.append(ev)
            continue
        extra=EXTRA.get(num,[])
        insert=len(lines)
        for j in range(len(lines)-1,-1,-1):
            if lines[j].startswith('!'):insert=j;break
        lines=lines[:insert]+extra+lines[insert:]
        if num==33:lines+=EXTRA.get(34,[])
        if num==23:
            # A native, editable formalization page replaces artificial padding.
            math=[paragraph('（七）语义共识对象与状态更新规范','heading'),paragraph('为使证据核查能够作用于推荐结果，本课题将候选、主张、来源与更新分别建模。设当前请求为x，合法候选集合为C，Agent集合为N。每个Agent产生报告，报告进一步拆分为具有稳定编号的主张。主张除保存文本外，还记录实体、属性、取值、适用范围与来源，便于判断同一对象是否存在相互冲突的描述。')]
            math.append(native_math('c = (id, entity, attribute, value, scope, type, provenance)'))
            math.append(paragraph('其中，type用于区分客观事实、显式约束、偏好假设和推荐预测。只有对象、属性和适用范围相同，且取值相互不兼容时，才构成可直接处理的事实争议。偏好解释与预测分歧保留各自依据，并通过外部反馈评价。'))
            math.append(native_math('v = (target_id, evidence_ids, verdict, reason)'))
            math.append(paragraph('核验结果v必须指向允许更新的目标，并给出证据引用和理由。verdict取接受、拒绝或未决。接受表示相关依据获得当前材料支持，拒绝表示应撤销相应支持或修正事实，未决表示当前资料不足。执行程序根据主张类型限制更新范围，事实核查不改写原始用户请求。'))
            math.append(native_math('s(t+1) = T(s(t), q, z)'))
            math.append(native_math('L(t+1) = Rank(C_eligible(s(t+1)), s(t+1))'))
            math.append(paragraph('s表示带版本的证据与候选状态，q为本次核查行动，z为得到的工具或来源结果。T定义支持增加、支持撤销、约束排除和未决保留等状态转移，Rank为冻结的决策器。更新后保存受影响候选、原始排序与新排序，从而评价核查是否真正改变决定。'))
            math.append(paragraph('该形式化规范描述实际执行对象及程序约束，并不直接提供推荐质量单调改善保证。核验可能发生错误，正确事实也未必能够完整表达用户效用，因此后续实验同时统计错误更正和正确误改，以外部推荐结果检验更新作用。'))
            body.append(page_break());body.append(frame(math))
            lines=[x.replace('（七）理论分析','（八）理论分析') for x in lines]
        content=render_lines(lines)
        if not first:body.append(page_break())
        body.append(frame(content,first))
        first=False
    sect=deepcopy(original[-1]);body.append(sect)
    # Original content chunks are removed; native text replaces them in the intended cells.
    rel=E.fromstring(parts['word/_rels/document.xml.rels'])
    deleted=[]
    for r in list(rel):
        if r.get('Type','').endswith('/aFChunk'):
            target=r.get('Target','').lstrip('/');deleted.append(target);rel.remove(r)
        elif r.get('Id')=='c2023110949':rel.remove(r)
    settings=E.fromstring(parts['word/settings.xml']);uf=settings.find('w:updateFields',NS)
    if uf is None:uf=el('updateFields');settings.append(uf)
    uf.set(W+'val','true')
    ct=E.fromstring(parts['[Content_Types].xml'])
    for e in list(ct):
        if e.get('Extension')=='docx':ct.remove(e)
    edits={'word/document.xml':E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True),'word/_rels/document.xml.rels':E.tostring(rel,xml_declaration=True,encoding='UTF-8',standalone=True),'word/settings.xml':E.tostring(settings,xml_declaration=True,encoding='UTF-8',standalone=True),'[Content_Types].xml':E.tostring(ct,xml_declaration=True,encoding='UTF-8',standalone=True)}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(OUT,'w',ZIP_DEFLATED) as dest:
        for info in z.infolist():
            if info.filename in deleted:continue
            dest.writestr(info,edits.get(info.filename,parts[info.filename]))
    preserved=[n for n in parts if n not in edits and n not in deleted]
    with ZipFile(OUT) as final:
        assert all(final.read(n)==parts[n] for n in preserved)
    (TMP/'package_verification.json').write_text(json.dumps({'reference_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),'preserved_parts':preserved,'edited_parts':list(edits),'removed_embedded_content':deleted,'figures':fig,'tables':tab},ensure_ascii=False,indent=2))
    print(OUT);print('Preserved package parts:',len(preserved),'Figures:',fig,'Tables:',tab)


if __name__=='__main__':build()
