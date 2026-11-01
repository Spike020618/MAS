from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "midterm"
OUT.mkdir(parents=True, exist_ok=True)
DOCX = OUT / "博弈驱动的去中心化语义共识机制_中期报告.docx"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=110, start=120, bottom=110, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def keep_with_next(p):
    p.paragraph_format.keep_with_next = True


def set_run_font(run, east="Arial Unicode MS", west="Arial Unicode MS", size=Pt(11), bold=None):
    run.font.name = west
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)
    run.font.size = size
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold


def add_text(p, text, bold=False, size=Pt(11), east="Arial Unicode MS"):
    r = p.add_run(text)
    set_run_font(r, east=east, size=size, bold=bold)
    return r


def add_body(doc, text, first=True):
    p = doc.add_paragraph()
    p.style = doc.styles["正文"]
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0.74) if first else Cm(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.5
    add_text(p, text)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    add_text(p, text, bold=True, size=Pt(15 if level == 1 else 12), east="Arial Unicode MS")
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(7)
    add_text(p, text, size=Pt(10), east="Arial Unicode MS")
    return p


def add_figure(doc, rel_path, caption, width_cm=14.8):
    path = ROOT / rel_path
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        add_caption(doc, caption)
    else:
        add_body(doc, f"［图表占位：{caption}。定稿时从实验产物导出并更新。］", first=False)


def add_table(doc, headers, rows, widths=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for j, h in enumerate(headers):
        c = hdr.cells[j]
        set_cell_shading(c, "1F4E78")
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        if widths:
            c.width = Cm(widths[j])
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(h))
        set_run_font(r, east="Arial Unicode MS", size=Pt(font_size), bold=True)
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(c)
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j, val in enumerate(row):
            c = cells[j]
            if i % 2 == 1:
                set_cell_shading(c, "EAF2F8")
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths:
                c.width = Cm(widths[j])
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            add_text(p, str(val), size=Pt(font_size))
            set_cell_margins(c)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    p._p.append(fld)


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial Unicode MS"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial Unicode MS")
    normal.font.size = Pt(11)
    if "正文" not in [s.name for s in styles]:
        styles.add_style("正文", 1)
    body = styles["正文"]
    body.font.name = "Arial Unicode MS"
    body._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial Unicode MS")
    body.font.size = Pt(11)
    for name, size in (("Heading 1", 15), ("Heading 2", 12)):
        st = styles[name]
        st.font.name = "Arial Unicode MS"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial Unicode MS")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
    title = styles["Title"]
    title.font.name = "Arial Unicode MS"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial Unicode MS")
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    ppr = title._element.get_or_add_pPr()
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is not None:
        ppr.remove(pbdr)


def build():
    doc = Document()
    configure_styles(doc)
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.2)
    add_page_number(sec)

    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(26)
    add_text(p, "硕士研究生学位论文中期报告", bold=True, size=Pt(24), east="Arial Unicode MS")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(32)
    add_text(p, "博弈驱动的去中心化语义共识机制及在影视推荐中的应用", bold=True, size=Pt(18), east="Arial Unicode MS")
    cover_rows = [
        ("学号", "［待填写］"), ("姓名", "马天成"), ("学生类别", "［待填写］"),
        ("学院", "［待填写］"), ("学科专业", "［待填写］"), ("研究方向", "多智能体系统与智能推荐"),
        ("导师姓名", "高志鹏"), ("攻读学位", "［待填写］"),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for k, v in cover_rows:
        cells = table.add_row().cells
        cells[0].width, cells[1].width = Cm(4), Cm(8.5)
        p0, p1 = cells[0].paragraphs[0], cells[1].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        add_text(p0, k + "：", bold=True, size=Pt(12), east="Arial Unicode MS")
        add_text(p1, v, size=Pt(12))
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, "2026 年 10 月", bold=True, size=Pt(13), east="Arial Unicode MS")
    doc.add_page_break()

    add_heading(doc, "基本信息", 1)
    add_table(doc, ["项目", "内容", "项目", "内容"], [
        ("论文题目", "博弈驱动的去中心化语义共识机制及在影视推荐中的应用", "论文类型", "应用基础研究"),
        ("选题来源", "学校自选项目", "开题日期", "［待填写］"),
        ("论文开始日期", "［待填写］", "中期报告日期", "2026 年 10 月"),
        ("报告地点与时间", "［待填写］", "已提交月报次数", "［待填写］"),
    ], widths=[2.5, 5.2, 2.5, 5.2], font_size=9)

    add_heading(doc, "一 研究内容简介", 1)
    add_heading(doc, "1 选题背景", 2)
    add_body(doc, "大语言模型驱动的多智能体系统能够通过独立采样、角色协作、讨论和投票处理复杂任务，但现有研究通常将重点放在通信结构与最终答案聚合上，对聚合之前的信息生产缺少可执行约束。多个Agent可能共享相似模型、提示和上下文，表面一致既可能来自独立证据，也可能来自低成本复制或共同偏差。若系统只观察最终答案，就难以识别Agent是否真实投入了推理、核验和工具使用资源。")
    add_body(doc, "本课题从机制设计视角把研究对象前移到信息生产阶段：设计者在行动前公开信用规则、行动成本、信息可见策略与验证预算，Agent依据局部状态选择资源行动，并提交由假设、证据、推断和结论构成的AEIC结构化报告。控制器验证报告、结算信用与成本、更新信誉并形成集体决策。该思路将自由文本协作转换为可观察、可解析和可审计的过程。")
    add_body(doc, "影视推荐是验证机制工程可行性的代表性应用。真实推荐系统既要处理用户偏好、年份和类型约束，也要避免生成不存在的影片实体，并保留推荐依据。课题采用先目录检索、再Agent评审、最后确定性约束门控的架构，使多智能体机制进入真实候选集、Top-K决策与Trace审计链路。")

    add_heading(doc, "2 研究目标与研究问题", 2)
    add_body(doc, "课题总体目标是构建一套博弈驱动的逻辑去中心化语义共识机制，并在通用问答和影视推荐任务中验证其行为响应、决策质量、工程可用性与适用边界。当前阶段围绕以下三个问题展开：")
    for t in [
        "如何把自由文本推理转换为可检查、可结算的语义对象，并让资源行动在执行层真实对应不同的Prompt、Token、调用次数和验证路径。",
        "生成前公开的过程合同能否改变Agent对高资源行动的选择，并在固定信息流和聚合规则下改善集体决策。",
        "该机制能否进入真实影视检索推荐链路，在保持目录实体有效与硬约束满足的同时，提供候选级证据、异议和完整审计记录。",
    ]:
        p = doc.add_paragraph(style="正文")
        p.paragraph_format.left_indent = Cm(0.74)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        p.paragraph_format.line_spacing = 1.5
        add_text(p, "• " + t)

    add_heading(doc, "3 研究内容", 2)
    add_figure(doc, "docs/papers/figures/fig_research_framework.png", "图 1 课题研究框架", 15.0)
    add_body(doc, "研究内容一是AEIC语义报告与公开过程合同。Agent报告记为r=(A,E,I,C)，分别表示假设、证据、推断与结论。合同m=(g,c,P,B)描述信用规则、行动成本、信息可见策略和验证预算。行动集合包括Low、AEIC、Verify和Imitate；执行器将不同动作映射到不同资源路径，避免把动作退化为文本标签。")
    add_body(doc, "研究内容二是合同执行、验证结算和共识形成。每轮按照合同公开、行动声明、资源执行、报告提交、验证结算和状态更新的顺序运行；同轮Agent相互不可见，只能读取策略允许的历史信息。控制器检查结构完整度、证据可验证性、无依据翻转等信号，并采用确定的聚合规则输出集体决策。")
    add_body(doc, "研究内容三是影视推荐系统应用。系统基于MovieLens正式目录和ReDial训练集共现关系执行混合召回，冻结候选集后由Preference、Diversity和Critic角色评审。年份、类型、排除项等硬约束由确定性程序执行，最终输出Top-K与可审计Trace。")

    doc.add_page_break()
    add_heading(doc, "4 关键技术", 2)
    add_table(doc, ["关键技术", "实现要点", "当前状态"], [
        ("结构化语义报告", "AEIC四字段、合法结论抽取、失败与修复日志", "已实现并用于实验"),
        ("公开过程合同", "信用信号、成本、信息权限、验证预算在行动前公开", "已完成主实验与扫描"),
        ("同步多轮审议", "五Agent、三轮、历史可见与同轮隔离", "已完成冻结实验"),
        ("目录约束推荐", "MovieLens目录、混合召回、候选冻结、硬约束门控", "已完成系统与全量评测"),
        ("可复现实验", "标准JSONL、SHA-256登记、独立复算与统计检验", "已完成验收"),
    ], widths=[3.1, 8.2, 3.2], font_size=9)

    add_heading(doc, "5 原计划与阶段目标", 2)
    add_table(doc, ["阶段", "计划内容", "阶段目标"], [
        ("阶段一", "完成相关研究调研、问题定义与总体框架", "明确机制边界与研究问题"),
        ("阶段二", "实现AEIC报告、行动资源路径、合同执行与Trace", "形成可运行机制原型"),
        ("阶段三", "完成通用问答机制实验与稳健性分析", "验证Follower行为响应及质量变化"),
        ("阶段四", "实现影视推荐系统并构建ReDial离线评测", "验证目录、约束、召回与审计链"),
        ("阶段五", "补充独立验证、人工审核与论文定稿", "收紧结论边界并完成答辩材料"),
    ], widths=[2.3, 8.2, 4.0], font_size=9)

    doc.add_page_break()
    add_heading(doc, "二 论文进展情况", 1)
    add_table(doc, ["计划任务", "实际进展", "完成度"], [
        ("文献调研与开题框架", "完成多智能体审议、结构化推理、机制设计和会话式推荐的调研，形成统一研究问题。", "已完成"),
        ("机制设计与实现", "完成AEIC接口、四类行动、合同执行、信用成本结算、信誉更新、失败降级与Trace。", "已完成"),
        ("机制实验", "完成三域450题配对主实验及奖励成本、信息环境、随机种子、面板规模、信号组合和后端边界分析。", "已完成"),
        ("系统开发", "完成MovieLens目录召回、多角色评审、硬约束、Top-K输出、Web/API与审计界面。", "已完成主体"),
        ("推荐评测", "完成1,203条ReDial测试样本的全量召回评测和100条固定样本Agent重排评测。", "已完成"),
        ("论文写作", "已形成毕业论文整合稿与中英文论文材料；参考文献、附录与学校格式仍需统一。", "进行中"),
        ("独立补验", "人工盲审、长期用户画像、无历史锚点语义召回和跨后端合同校准仍待补充。", "待完成"),
    ], widths=[3.1, 9.1, 2.3], font_size=9)
    add_body(doc, "总体上，课题已完成从理论框架、机制原型、确认性实验到应用系统的主体闭环。当前工作重点已经从“能否运行”转向“结论边界是否充分、应用薄弱环节如何补强以及论文材料如何统一”。")

    add_heading(doc, "三 工作成果", 1)
    add_heading(doc, "1 博弈驱动的去中心化语义共识机制", 2)
    add_body(doc, "课题形成了公开合同驱动的信息生产机制。Leader先承诺规则，Follower Agent再依据任务、合同、允许访问的历史和局部状态选择行动。当前所谓去中心化是决策与语义报告层面的逻辑去中心化：各Agent独立生成报告，统一控制器仍负责任务调度、权限控制、验证与聚合，不声称已经实现跨机器P2P基础设施。")
    add_figure(doc, "docs/papers/figures/fig1_aeic_md_overview_rich.png", "图 2 公开合同与语义报告机制示意", 14.5)
    add_body(doc, "机制记录complete_aeic、consistency、novel_evidence、verified_evidence、useful_correction和unsupported_flip等可观察信号。报告信用为各信号的加权和，Agent效用为信用减成本，信誉按衰减累计。信用和成本是系统内部记账量，不解释为真实货币或人类效用。")

    add_heading(doc, "2 通用问答机制实验", 2)
    add_body(doc, "主实验从StrategyQA、MMLU-Hard和TruthfulQA各选150题，共450个配对问题。每个条件使用相同问题、五个Agent、三轮同步审议、相同模型和信息配置。No Mechanism不提供显式信用；Outcome Only只按最终结果结算；Static AEIC按行动前公开的过程信号结算。")
    add_table(doc, ["条件", "高资源行动率", "第三轮多数准确率", "相对基线"], [
        ("No Mechanism", "0.46%", "68.00%", "基线"),
        ("Outcome Only", "3.08%", "67.33%", "准确率 -0.67pp"),
        ("Static AEIC", "56.93%", "82.22%", "准确率 +14.22pp"),
    ], widths=[3.4, 3.4, 4.0, 3.7], font_size=9)
    add_caption(doc, "表 1 主实验结果")
    add_figure(doc, "docs/papers/en/figures/fig_main_results.png", "图 3 机制主实验结果", 14.8)
    add_body(doc, "Static AEIC相对基线有78题由错变对、14题由对变错，Holm校正后的精确McNemar检验p=1.23×10^-11。每个Agent-round平均增加145.5 Token，因此该结果不能描述为零成本提升。Outcome Only没有产生相同行动和准确率改善，说明仅给出结果奖励语言不足以解释主效应。")
    add_body(doc, "奖励提高时高资源行动上升，成本提高时高资源行动下降；额外随机种子、不同面板规模、信号组合与历史策略扰动支持主方向稳定。等计算量对照没有复现相同响应，AEIC相对通用结构化接口也未显示显著准确率优势；留出GLM后端未复现主后端效果。这些结果将结论限定为：公开合同、执行提示与分级资源路径的联合配置在已测后端上有效，AEIC已证实的作用主要是提供统一、可解析和可结算的语义接口。")

    add_heading(doc, "3 影视推荐系统", 2)
    add_figure(doc, "docs/papers/figures/fig4_system_architecture.png", "图 4 影视推荐系统总体架构", 15.2)
    add_body(doc, "系统以18,430部MovieLens影片为正式目录，采用词法分数与ReDial训练集影片共现分数的混合召回。候选集冻结后，Preference Agent评估显式偏好匹配，Diversity Agent评估列表覆盖与同质化风险，Critic Agent检查约束冲突和无依据理由。生成式Agent只能评审目录候选，不能自由生成影片实体。")
    add_body(doc, "Trace记录request_id、计划、公开合同、候选集、行动、AEIC报告、Token、延迟、信誉变化、硬约束检查和最终排名。模型超时、空响应或结构错误会进入重试与降级路径；无法恢复时使用确定性Agent完成安全输出，并明确记录降级事件。")

    add_heading(doc, "4 ReDial应用评测", 2)
    add_body(doc, "ReDial原始训练集包含8,004段对话，测试集包含1,342段。完成MovieLens映射和构造条件筛选后，得到7,348段训练对话和1,203段测试对话。每条样本只保留首个正向目标影片出现之前的上下文，开发集与测试集零重叠，目标影片均属于目录，输入上下文不含目标ID。")
    add_table(doc, ["方法", "Hit@5", "Recall@5", "NDCG@5"], [
        ("词法召回", "0.75%", "0.26%", "0.24%"),
        ("训练集共现", "10.47%", "5.83%", "4.88%"),
        ("混合召回", "10.81%", "5.95%", "5.01%"),
    ], widths=[4.8, 3.2, 3.2, 3.2], font_size=9)
    add_caption(doc, "表 2 全量1,203条测试样本召回结果")
    add_body(doc, "混合召回相对词法提高10.06个百分点，Bootstrap 95%置信区间为[8.40, 11.89]个百分点；123条只被混合方法命中，2条只被词法命中，精确McNemar检验p=3.70×10^-34。")
    add_table(doc, ["重排方法", "Hit@5", "Recall@5", "NDCG@5", "平均资源"], [
        ("直接排序", "6%", "4.17%", "2.66%", "确定性"),
        ("单Agent", "7%", "5.17%", "3.11%", "5,842 Token / 7.44秒"),
        ("多Agent共识", "7%", "5.17%", "2.96%", "16,935 Token / 12.68秒"),
    ], widths=[3.0, 2.2, 2.4, 2.4, 4.5], font_size=9)
    add_caption(doc, "表 3 100条固定样本Agent重排结果")
    add_body(doc, "多Agent共识与单Agent的命中集合完全相同，McNemar p=1.0。三种方法的目录实体有效率和硬约束满足率均为100%。共识报告提供51.2%的可核验证据覆盖，但没有显示推荐准确率优势。该负结果是中期结论的重要组成部分：机制在推荐场景中的当前价值是约束执行、证据组织与审计，而不是已证实的相关性提升。")

    add_heading(doc, "5 阶段性成果汇总", 2)
    add_table(doc, ["类别", "阶段成果", "证据状态"], [
        ("方法", "公开过程合同、AEIC接口、四类行动、验证结算与信誉更新", "代码与实验已落地"),
        ("实验", "三域450题主实验及多组稳健性、消融和后端边界检查", "冻结结果可复算"),
        ("系统", "影视搜索推荐API、Web界面、多角色评审、硬约束和Trace", "已完成工程验收"),
        ("数据", "MovieLens目录与ReDial无标签泄漏评测基准", "验证脚本通过"),
        ("论文", "毕业论文整合稿、英文论文与图表材料", "内容成形，格式待统一"),
    ], widths=[2.2, 9.6, 2.8], font_size=9)
    add_body(doc, "［成果证明占位：定稿时可补充论文投稿信息、专利或软著受理信息、系统部署截图、代码仓库提交记录及导师确认材料。］", first=False)

    add_heading(doc, "四 计划及进度安排", 1)
    add_heading(doc, "1 下一步工作计划", 2)
    add_body(doc, "下一阶段将优先完成研究收口，而不是继续扩展未经验证的新机制。具体包括：补齐跨后端合同校准或明确不迁移边界；围绕无历史锚点样本加强语义召回、协同过滤与用户画像；对推荐理由和AEIC证据组织实施独立人工审核；统一实验登记、图表、参考文献和学校格式；完成论文修改、预答辩与送审材料。")
    add_body(doc, "［图表占位：后续若完成人工审核，在此加入审核流程图与一致性统计表；若完成新召回模块，在此加入同一冻结测试集上的配对对比表。］", first=False)
    add_heading(doc, "2 进度安排", 2)
    add_table(doc, ["时间", "计划", "验收标准"], [
        ("2026.10-2026.11", "统一中期报告、论文叙事与冻结结果口径", "数字、图表、代码入口一致"),
        ("2026.11-2026.12", "补充人工审核与必要的独立验证", "预先锁定协议，不追结果调参"),
        ("2026.12-2027.01", "改进冷启动与无锚点召回，完成系统补测", "同一测试集配对比较并报告资源开销"),
        ("2027.01-2027.02", "完成毕业论文初稿修改与导师审阅", "章节、附录、参考文献齐备"),
        ("2027.02-2027.03", "预答辩、定稿与送审准备", "学校格式检查与重复率检查通过"),
        ("2027.04-2027.06", "论文送审与毕业答辩", "完成归档材料"),
    ], widths=[3.2, 7.4, 4.0], font_size=9)

    add_heading(doc, "五 问题及整改方案", 1)
    add_heading(doc, "1 当前困难与问题", 2)
    add_table(doc, ["问题", "影响", "当前判断"], [
        ("跨后端泛化不足", "同一合同在留出后端未复现主效应", "合同需要面向后端校准，不能宣称通用激励相容"),
        ("推荐准确率增益不足", "多Agent共识未超过单Agent", "主要瓶颈是冷启动与候选覆盖，而非聚合层"),
        ("AEIC真实性验证有限", "结构完整不等于事实正确", "需增加可定位来源与独立人工审核"),
        ("资源成本较高", "多Agent Token与延迟显著增加", "必须同时报告质量、成本与延迟"),
        ("用户建模较弱", "依赖显式请求和短期影片锚点", "长期画像与语义召回仍待建设"),
    ], widths=[3.0, 5.3, 6.3], font_size=9)
    add_heading(doc, "2 整改方案", 2)
    add_body(doc, "针对跨后端问题，在开发集上估计Follower响应面，冻结合同后再进行独立测试；若无法复现，则将结论明确限定到主后端。针对推荐问题，优先增加语义召回和协同过滤基线，并在相同候选预算下比较，不用增加Agent数量替代候选覆盖。针对真实性与成本问题，建立引用可定位性、证据充分性和理由一致性的人工审核协议，同时把Token、调用次数、平均延迟和P95延迟作为主要资源指标。")
    add_body(doc, "所有后续补验均应在执行前固定数据划分、主要指标、停止规则与验收条件；负结果和失败路径保留在论文中，不以事后筛选样本或调整指标的方式制造正向结论。")

    doc.add_page_break()
    add_heading(doc, "六 参考文献", 1)
    refs = [
        "[1] Wang L, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 2024.",
        "[2] Du Y, et al. Improving factuality and reasoning in language models through multiagent debate. ICML, 2024.",
        "[3] Wei J, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022.",
        "[4] Yao S, et al. ReAct: Synergizing reasoning and acting in language models. ICLR, 2023.",
        "[5] Shoham Y, Leyton-Brown K. Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations. Cambridge University Press, 2009.",
        "[6] Fudenberg D, Tirole J. Game Theory. MIT Press, 1991.",
        "[7] Koren Y, Bell R, Volinsky C. Matrix factorization techniques for recommender systems. Computer, 2009.",
        "[8] Li R, et al. Towards deep conversational recommendations. NeurIPS, 2018.",
        "[9] Harper F M, Konstan J A. The MovieLens datasets: History and context. ACM TiiS, 2015.",
        "[10] Dietterich T G. Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 1998.",
        "[11] Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 1979.",
        "[12] McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 1947.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="正文")
        p.paragraph_format.first_line_indent = Cm(-0.74)
        p.paragraph_format.left_indent = Cm(0.74)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(3)
        add_text(p, ref, size=Pt(10))
    add_body(doc, "注：最终定稿时需根据学校参考文献格式与论文BibTeX库逐条核对作者、卷期、页码和DOI。", first=False)

    doc.save(DOCX)
    print(DOCX)


if __name__ == "__main__":
    build()
