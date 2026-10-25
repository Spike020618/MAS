from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "博弈驱动的多智能体语义共识方法及影视推荐系统_专利交底书.docx"
FIGURE_ROOT = ROOT.parent / "papers" / "figures"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
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


def set_repeat_table_headers(table, header_rows=1):
    for row in table.rows[:header_rows]:
        set_repeat_table_header(row)


def set_cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    tr_pr.append(cant)


def set_east_asia_font(run, name="宋体"):
    name = {"宋体": "Songti SC", "黑体": "STHeiti"}.get(name, name)
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)


def configure_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Songti SC")
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.first_line_indent = Cm(0.74)
    pf.space_after = Pt(3)
    pf.widow_control = True

    specs = {
        "Title": ("黑体", 20, True, RGBColor(0, 0, 0), 0, 0),
        "Heading 1": ("黑体", 16, True, RGBColor(0, 0, 0), 12, 6),
        "Heading 2": ("黑体", 14, True, RGBColor(0, 0, 0), 9, 4),
        "Heading 3": ("黑体", 12, True, RGBColor(0, 0, 0), 6, 3),
        "Caption": ("宋体", 10, False, RGBColor(80, 80, 80), 3, 6),
    }
    for style_name, (font, size, bold, color, before, after) in specs.items():
        st = doc.styles[style_name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.color.rgb = color
        st._element.rPr.rFonts.set(qn("w:eastAsia"), {"宋体": "Songti SC", "黑体": "STHeiti"}.get(font, font))
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.widow_control = True
    doc.styles["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_ppr = doc.styles["Title"]._element.get_or_add_pPr()
    p_bdr = title_ppr.find(qn("w:pBdr"))
    if p_bdr is not None:
        title_ppr.remove(p_bdr)
    doc.styles["Caption"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.styles["Caption"].paragraph_format.first_line_indent = Cm(0)


def add_run(p, text, bold=False, italic=False, color=None, size=None, font="宋体"):
    r = p.add_run(text)
    set_east_asia_font(r, font)
    r.bold = bold
    r.italic = italic
    if color:
        r.font.color.rgb = RGBColor(*color)
    if size:
        r.font.size = Pt(size)
    return r


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.keep_together = False
    if bold_prefix and text.startswith(bold_prefix):
        add_run(p, bold_prefix, bold=True)
        add_run(p, text[len(bold_prefix):])
    else:
        add_run(p, text)
    return p


def add_numbered(doc, number, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.2)
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_together = False
    add_run(p, f"{number} ", bold=True, color=(46, 88, 120))
    add_run(p, text)
    return p


def add_heading(doc, text, level):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.first_line_indent = Cm(0)
    add_run(p, text, bold=True, font="黑体")
    return p


def add_caption(doc, text):
    p = doc.add_paragraph(style="Caption")
    add_run(p, text, font="宋体")
    return p


def add_figure(doc, image_name, caption, width_cm=16.0):
    """Insert an actual technical drawing followed by its caption."""
    image_path = FIGURE_ROOT / image_name
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(image_path), width=Cm(width_cm))
    add_caption(doc, caption)
    return p


def add_key_value_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [Cm(2.6), Cm(7.2), Cm(2.6), Cm(4.7)]
    for i, row_data in enumerate(rows):
        row = table.rows[i]
        set_cant_split(row)
        for j, value in enumerate(row_data):
            cell = row.cells[j]
            cell.width = widths[j]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell, 120, 120, 120, 120)
            if j % 2 == 0:
                set_cell_shading(cell, "D9E7F2")
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Cm(0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j % 2 == 0 else WD_ALIGN_PARAGRAPH.LEFT
            add_run(p, value, bold=(j % 2 == 0), font="黑体" if j % 2 == 0 else "宋体")
    return table


def add_matrix(doc, headers, rows, widths=None, font_size=9.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = widths is None
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        set_cell_shading(cell, "D9E7F2")
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.paragraph_format.first_line_indent = Cm(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run(p, h, bold=True, size=font_size, font="黑体")
        if widths:
            cell.width = widths[j]
    for row_data in rows:
        row = table.add_row()
        set_cant_split(row)
        for j, value in enumerate(row_data):
            cell = row.cells[j]
            set_cell_margins(cell)
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.15
            add_run(p, str(value), size=font_size)
            if widths:
                cell.width = widths[j]
    set_repeat_table_headers(table)
    return table


def add_flow(doc, labels, caption):
    cols = len(labels) * 2 - 1
    table = doc.add_table(rows=1, cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for idx, label in enumerate(labels):
        c = table.cell(0, idx * 2)
        set_cell_shading(c, "EAF2F8" if idx % 2 == 0 else "E8F4F1")
        set_cell_margins(c, 180, 80, 180, 80)
        p = c.paragraphs[0]
        p.paragraph_format.first_line_indent = Cm(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run(p, label, bold=True, size=9, font="黑体")
        if idx < len(labels) - 1:
            a = table.cell(0, idx * 2 + 1)
            p2 = a.paragraphs[0]
            p2.paragraph_format.first_line_indent = Cm(0)
            p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_run(p2, "→", bold=True, size=14, color=(46, 88, 120))
    add_caption(doc, caption)
    return table


def add_math(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    omath = OxmlElement("m:oMath")
    mr = OxmlElement("m:r")
    mt = OxmlElement("m:t")
    mt.text = text
    mr.append(mt)
    omath.append(mr)
    p._p.append(omath)
    return p


def add_page_field(paragraph):
    add_run(paragraph, "    第 ", size=8.5, color=(120, 120, 120))
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text_run = OxmlElement("w:r")
    text_node = OxmlElement("w:t")
    text_node.text = "1"
    text_run.append(text_node)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    begin_run = OxmlElement("w:r")
    begin_run.append(begin)
    instr_run = OxmlElement("w:r")
    instr_run.append(instr)
    separate_run = OxmlElement("w:r")
    separate_run.append(separate)
    end_run = OxmlElement("w:r")
    end_run.append(end)
    paragraph._p.extend([begin_run, instr_run, separate_run, text_run, end_run])
    add_run(paragraph, " 页", size=8.5, color=(120, 120, 120))


doc = Document()
configure_styles(doc)
sec = doc.sections[0]
sec.page_width = Cm(21)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.3)
sec.bottom_margin = Cm(2.0)
sec.left_margin = Cm(2.6)
sec.right_margin = Cm(2.3)

# Header/footer
header = sec.header.paragraphs[0]
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
header.paragraph_format.first_line_indent = Cm(0)
add_run(header, "技术交底书", size=9, color=(100, 100, 100))
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.paragraph_format.first_line_indent = Cm(0)
add_run(footer, "博弈驱动的多智能体语义共识方法及影视推荐系统", size=8.5, color=(120, 120, 120))
add_page_field(footer)

# Cover
p = doc.add_paragraph(style="Title")
p.paragraph_format.space_before = Pt(20)
p.paragraph_format.space_after = Pt(26)
add_run(p, "专 利 技 术 交 底 书", bold=True, size=24, font="黑体", color=(0, 0, 0))
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.paragraph_format.first_line_indent = Cm(0)
p2.paragraph_format.space_after = Pt(20)
add_run(p2, "一种博弈驱动的多智能体语义共识方法及影视推荐系统", bold=True, size=18, font="黑体")

add_key_value_table(doc, [
    ("交底书名称", "一种博弈驱动的多智能体语义共识方法及影视推荐系统", "专利类型", "发明专利"),
    ("发明人", "马天成、高志鹏", "专利负责人", "高志鹏"),
    ("交底书撰写人", "马天成", "所属单位", "北京邮电大学"),
    ("联系电话", "", "E-mail", ""),
    ("版本日期", "2026年10月", "材料状态", "技术交底稿"),
])

note = doc.add_paragraph()
note.paragraph_format.first_line_indent = Cm(0)
note.paragraph_format.space_before = Pt(18)
add_run(note, "说明：", bold=True, font="黑体")
add_run(note, "本交底书依据现有研究代码、实验记录与影视推荐系统实现重写。联系人信息留待提交前补充。")

doc.add_page_break()

add_heading(doc, "0  缩略语和关键术语定义", 1)
add_matrix(doc, ["术语", "定义"], [
    ("AEIC", "由假设（Assumption）、证据（Evidence）、推断（Inference）和结论（Conclusion）四部分组成的结构化语义报告接口。"),
    ("公开合同", "在 Agent 生成报告之前公开的规则集合，至少包含信用规则、行动成本、信息可见策略和验证预算。"),
    ("行动", "Agent 在公开合同下选择的资源执行方式，包括低资源回答、完整 AEIC 报告、验证以及模仿/沿用同伴信息。"),
    ("信用/效用", "系统根据可观察报告信号计算信用，并扣除所选行动的公开成本，得到本轮已实现效用。"),
    ("信誉", "Agent 跨轮次保留的历史状态，由衰减后的既有信誉与本轮效用共同更新，用于后续加权或资源控制。"),
    ("逻辑去中心化", "各 Agent 根据局部可见状态独立选行动并形成报告；统一控制器只负责合同执行、同步、验证、聚合和日志，不等同于跨机器 P2P 部署。"),
    ("冻结候选集", "在多 Agent 评审开始前由目录检索器确定、并在同一次请求内保持不变的候选集合，用于保证评审对象一致。"),
    ("证据共识", "在保留支持、反对和弃权意见的前提下，对经门控的候选及其证据进行加权聚合形成的候选级决策。"),
], widths=[Cm(3.3), Cm(12.6)], font_size=10)

add_heading(doc, "1  相关技术背景及最接近的现有实现方案", 1)
add_heading(doc, "1.1  技术背景", 2)
add_body(doc, "大语言模型驱动的多智能体系统通常让多个 Agent 分别生成自然语言答案，再通过多数投票、固定权重或中心化裁判模型给出最终结论。此类方法关注聚合结果，却很少约束 Agent 在聚合前是否投入了验证资源、是否独立形成证据、是否只是复述其他 Agent 的观点。自由文本还会把假设、证据和结论混在一起，使得系统难以区分“有新证据的修正”和“无依据的立场翻转”。")
add_body(doc, "在影视搜索与推荐场景中，用户请求同时包含年份、类型、排除项等硬约束，以及题材、风格、多样性等软偏好。如果直接把请求交给生成式模型，可能出现不存在的影片、遗漏排除条件、不同 Agent 评审不同候选或无法解释 Top-K 排序等问题。仅在最终结果处增加一个裁判模型，既不能保证候选实体来自真实目录，也无法审计每个 Agent 的证据生产过程。")

add_heading(doc, "1.2  现有技术方案一：结果级投票或加权聚合", 2)
add_heading(doc, "1.2.1  现有实现方式", 3)
add_body(doc, "多个 Agent 接收同一问题，各自输出结论；控制器对结论进行多数投票，或根据预设权重、置信度、历史正确率计算加权结果。部分系统在生成后再对回答完整度、置信度或文本相似度评分。")
add_heading(doc, "1.2.2  现有方案的不足", 3)
add_numbered(doc, "（1）", "规则在生成之后才作用于聚合，无法改变 Agent 在生成前的资源投入选择。")
add_numbered(doc, "（2）", "自由文本缺少统一可解析字段，完整度、证据增量、无依据翻转和有用纠错难以比较。")
add_numbered(doc, "（3）", "只奖励最终正确结果时，在真实答案不可立即获得的场景中难以结算，也容易忽略可核验过程。")
add_numbered(doc, "（4）", "中心裁判模型形成新的单点判断，其自身偏差和幻觉难以追溯。")

add_heading(doc, "1.3  现有技术方案二：多 Agent 辩论或拓扑通信", 2)
add_heading(doc, "1.3.1  现有实现方式", 3)
add_body(doc, "系统按照并行、顺序、星型或辩论拓扑传递 Agent 消息，通过多轮互评获得一致结论；部分系统根据分歧程度调整通信轮数。")
add_heading(doc, "1.3.2  现有方案的不足", 3)
add_numbered(doc, "（1）", "通信拓扑决定信息流向，但不能保证 Agent 愿意执行成本更高的检索、核验和独立推理。")
add_numbered(doc, "（2）", "过早共享同伴结论会诱发模仿或从众，使表面一致掩盖证据多样性下降。")
add_numbered(doc, "（3）", "不同候选被不同数量或不同角色的 Agent 审查时，候选分数不可直接比较，Top-K 可能由覆盖差异而非质量差异决定。")
add_numbered(doc, "（4）", "若缺少事件时序和资源日志，无法证明合同先于行动、行动确实触发了对应资源路径。")

add_heading(doc, "1.4  现有技术方案三：生成式影视推荐", 2)
add_heading(doc, "1.4.1  现有实现方式", 3)
add_body(doc, "系统用关键词或向量检索召回影片，再由一个或多个生成式模型解释用户偏好并重排候选。典型实现把类型、年份、相似度和模型评分合并为一个最终分数。")
add_heading(doc, "1.4.2  现有方案的不足", 3)
add_numbered(doc, "（1）", "硬约束与软偏好未分层处理，生成模型可能为了满足 Top-K 数量而放松年份、类型或排除条件。")
add_numbered(doc, "（2）", "候选实体不一定来自真实影片目录，存在影片名称、年份或类型幻觉。")
add_numbered(doc, "（3）", "偏好、多样性和风险意见被压缩为单一分数，反对理由和剩余分歧不可见。")
add_numbered(doc, "（4）", "系统无法从最终推荐反向定位检索、评审、验证和聚合阶段的具体责任。")

add_heading(doc, "1.5  本发明与现有方案的实质区别", 2)
add_body(doc, "本发明不是把结构化提示、多数投票和影视检索简单并列，而是以可验证时序把规则、行动、资源、报告和决策连接成闭环。各区别特征均改变计算机系统中的信息可见范围、模型调用路径、候选处理顺序或状态写入方式，并产生可检查的中间数据。")
add_matrix(doc, ["现有方案", "本发明的区别特征", "产生的技术效果"], [
    ("生成后评分或投票", "合同先于行动提交并绑定请求", "可以验证规则是否真正作用于生成策略"),
    ("Agent自报已努力", "行动标识强制映射提示、调用、Token和验证路径", "声明行为与真实计算资源一致"),
    ("自由文本互评", "A/E/I/C及证据来源采用版本化结构", "报告可解析、可比较、可重算"),
    ("不同候选接受不同评审", "先冻结候选，再执行统一Agent覆盖", "候选得分具备同一比较基准"),
    ("约束作为软分数", "目录硬约束作为排序前门控条件", "高偏好分不能抵消年份或排除项违规"),
    ("只保存最终答案", "请求级事件链保存合同、行动、资源、报告和结算", "能够复现结果并定位故障阶段"),
], widths=[Cm(4.0), Cm(6.3), Cm(5.6)], font_size=9.2)

add_heading(doc, "2  本发明技术方案的详细阐述", 1)
add_heading(doc, "2.1  本发明要解决的技术问题", 2)
add_body(doc, "本发明解决的核心技术问题是：在多个大语言模型 Agent 存在资源成本、信息不完全和相互影响的条件下，如何使 Agent 在生成报告之前观察到统一规则并选择真实资源行动，使自然语言推理转化为可解析、可验证、可结算的数据对象，并在不依赖单一中心裁判模型的情况下形成可审计的集体决策。")
add_body(doc, "在影视推荐实施例中，本发明进一步解决：如何保证所有 Agent 在同一冻结候选集上完成一致覆盖；如何将目录事实和硬约束从生成式判断中分离；如何同时利用偏好支持、证据质量、历史信誉和反对意见得到 Top-K；以及如何保存从请求到最终排序的完整证据链。")

add_heading(doc, "2.2  完整技术方案", 2)
add_figure(doc, "fig1_aeic_md_overview.png", "图1  博弈驱动的多智能体语义共识总体架构")
add_body(doc, "如图1所示，本发明将“先承诺、后响应”的博弈时序固化到控制流程中。公开合同写入 Trace 后，Agent 才能选择行动；行动选择随后控制实际提示模板、模型调用次数、Token 预算、验证调用和可见同伴信息。系统拒绝仅在输出完成后给回答贴上“高资源”标签的实现方式。")

add_heading(doc, "2.2.1  任务对象和系统状态定义", 3)
add_body(doc, "设一次待决策任务为 q，参与任务的 Agent 集合为 N={1,…,n}，同步审议轮次为 t={1,…,T}。控制器为任务建立状态 S_t，状态至少包括请求内容、公开合同、轮次、各 Agent 的信誉快照、信息可见矩阵、剩余验证预算以及已经生成的报告索引。任务真实标签或事后质量信息不属于 Agent 的可见状态，除非合同明确允许在结算阶段使用。")
add_math(doc, "S_t = (q, m, t, ρ(t−1), P_t, B_t, H_t)")
add_body(doc, "其中 ρ(t−1) 为进入第 t 轮前的信誉向量，P_t 为信息可见矩阵，B_t 为剩余验证预算，H_t 为符合可见策略的历史记录。每个 Agent i 只获得由投影函数 view_i(S_t) 产生的局部状态，不直接读取其他 Agent 的私有提示、原始推理或被策略屏蔽的报告。该设计使 Agent 在语义报告层独立行动，同时由统一控制器完成同步和日志，不要求底层采用跨机器对等网络。")
add_body(doc, "系统区分三个时间点：合同提交时间 τ_c、行动选择时间 τ_a 和报告完成时间 τ_r，并要求 τ_c<τ_a<τ_r。任一记录不满足该时序，或合同摘要与 Agent 实际看到的版本不一致，控制器均将该报告标记为不可结算。通过该约束，可防止系统在看到回答后修改奖励，或把事后分类伪装成 Agent 的事前行动。")
add_matrix(doc, ["符号", "含义"], [
    ("q、N、T", "任务、参与Agent集合和同步审议总轮数"),
    ("S_t、H_t", "第t轮全局控制状态和策略允许可见的历史"),
    ("m=(g,c,P,B)", "公开合同及其信用规则、行动成本、可见策略和验证预算"),
    ("a_i、R_i", "Agent i选择的离散行动及控制器实际执行的资源路径"),
    ("r_i、s_i", "AEIC结构化报告及由报告和日志提取的信号向量"),
    ("ρ_i、ω_i", "Agent信誉状态及聚合时使用的受限权重"),
    ("C_frozen、z_ij", "冻结候选集及Agent i对候选j的有效覆盖标记"),
], widths=[Cm(4.0), Cm(11.9)], font_size=9.4)

add_heading(doc, "2.2.2  步骤S1：构造并公开合同", 3)
add_body(doc, "控制器针对当前任务生成公开合同 m。合同至少包含四类参数：信用规则 g、各行动成本 c、信息可见策略 P 和验证预算 B。合同在任一 Agent 生成任务报告之前，以机器可解析形式写入不可变请求日志，并与请求标识、合同版本和时间戳绑定。")
add_math(doc, "m = (g, c, P, B)")
add_body(doc, "信用规则定义哪些可观察信号获得信用；行动成本表示各执行路径的相对资源消耗；信息策略限定 Agent 在每轮可见的问题、历史信誉和同伴报告；验证预算规定允许的额外校验次数或 Token 上限。控制器可使用固定合同，也可从有限合同集合中选择，但在一次请求内不得在 Agent 行动后追溯修改。")
add_body(doc, "合同中的信用和成本是控制器使用的机器可读评分与资源参数，不代表货币支付。信用决定日志中的效用、后续聚合权重或资源权限；成本对应模型调用、Token、验证次数和延迟等计算开销。通过把数值和可执行路径绑定，合同从抽象话术转化为计算机资源控制配置。")
add_body(doc, "合同以版本化 JSON、数据库记录或等价结构存储。信用规则 g 包含信号名称、权重、取值范围和无法计算时的处理方式；成本表 c 为每一种行动给出公开成本；P 可以按 Agent、轮次和字段规定可见性；B 可以同时限制总验证次数、总 Token、并发数和最长延迟。为保证可复现，合同还包含随机种子、提示版本、聚合器版本和模式版本的哈希。")
add_matrix(doc, ["合同分量", "内容", "执行约束"], [
    ("信用规则 g", "信号集合、权重、惩罚项、结算上限", "报告生成前固定；结算时逐项输出分解"),
    ("行动成本 c", "Low、AEIC、Verify、Imitate等行动的公开成本", "成本与真实资源路径绑定，不随结果追溯改变"),
    ("可见策略 P", "Agent×轮次×字段的可见矩阵", "只向 Agent 暴露允许的信息和来源标识"),
    ("验证预算 B", "调用、Token、延迟和并发预算", "超预算时停止额外验证并记录降级原因"),
], widths=[Cm(3.0), Cm(6.4), Cm(6.5)], font_size=9.3)

add_heading(doc, "2.2.3  步骤S2：Agent独立选择行动", 3)
add_body(doc, "每个 Agent 接收公开合同、允许可见的局部状态和任务输入，并从行动集合 A 中返回唯一行动标识。优选行动集合包括 Low、AEIC、Verify 和 Imitate。行动选择响应与任务报告分两次调用或两个逻辑阶段完成，以避免控制器根据已生成内容倒推行动。")
add_matrix(doc, ["行动", "真实执行路径", "可记录资源"], [
    ("Low", "短提示、低 Token 上限、单次生成，仅返回最小结论或简化报告。", "调用次数、输入/输出 Token、延迟"),
    ("AEIC", "使用完整字段模板生成假设、证据、推断和结论。", "字段完整度、证据数量、Token、延迟"),
    ("Verify", "先生成 AEIC 报告，再执行独立校验或审计修订。", "额外调用、核验证据、修订轨迹"),
    ("Imitate", "在信息策略允许时读取指定同伴信息，并明确记录被引用对象。", "可见同伴、引用关系、复制/翻转信号"),
], widths=[Cm(2.2), Cm(9.4), Cm(4.3)], font_size=9.5)
add_body(doc, "Agent 的行动响应至少包含 action、reason_code、expected_credit 和 expected_cost 四个字段。reason_code 只描述选择理由，不允许提前输出任务答案；expected_credit 和 expected_cost 用于审计 Agent 是否读取了合同，不作为真实信用的替代。控制器对 action 执行枚举校验，对非法值按照合同规定回退到 Low 或拒绝本轮。")
add_math(doc, "a_i(t) ∈ {Low, AEIC, Verify, Imitate}")
add_body(doc, "本发明不要求大语言模型精确求解经典效用最大化，也不将自然语言模型视为完全理性主体。可检验的技术效果是：当公开信用或行动成本变化时，所实现模型的行动分布是否发生方向一致的变化。因此，系统保存每次合同与行动的联合记录，用于估计有限合同条件下的经验响应，而不把固定提示下的一次输出夸大为一般博弈均衡。")

add_heading(doc, "2.2.4  步骤S3：按行动路由实际资源", 3)
add_body(doc, "控制器依据行动标识选择预先注册的资源执行器，不允许 Agent 仅声明 Verify 而不触发校验调用，也不允许 Low 行动使用高资源提示。执行器记录模型标识、提示版本、温度、调用次数、Token、延迟、重试和格式修复。格式修复只用于恢复结构，不得引入新事实或改变原始结论。")
add_body(doc, "在仅提供一个 DeepSeek API 的部署条件下，不同逻辑 Agent 可共享同一模型端点，但必须使用相互独立的角色提示、局部状态、可见信息和日志标识。该实现不要求不同厂商模型，也不把“异构角色”错误表述为“异构基础模型”。")
add_body(doc, "资源路由器维护 action 到 executor 的一一映射。Low 执行器只允许一次短生成；AEIC 执行器分配完整字段模板和较高输出上限；Verify 执行器在 AEIC 生成后调用目录、检索器、规则引擎或第二次模型审计；Imitate 执行器只读取 P 允许的同伴字段，并把来源 Agent、轮次和报告哈希写入引用清单。若执行器失败，重试次数和回退路径必须计入同一 action_trace。")
add_math(doc, "R_i = route(a_i, P_i, B_i)")
add_body(doc, "R_i 表示 Agent i 的实际资源路径。控制器在结算前比较声明行动 a_i 与实际路径 R_i：如果 Verify 未产生任何验证事件，或 Imitate 未记录来源，则对应行动信号置为无效；如果实际调用超出 B_i，超出部分不能获得额外信用。通过声明与执行的双重校验，避免 Agent 仅通过输出格式获得高资源奖励。")

add_heading(doc, "2.2.5  步骤S4：生成AEIC结构化语义报告", 3)
add_body(doc, "资源执行器输出版本化 AEIC 报告。Assumption 字段记录成立条件、约束解释和不确定前提；Evidence 字段记录可引用的事实、目录记录、检索片段或同伴来源；Inference 字段记录从假设与证据到判断的推导；Conclusion 字段记录规范化结论、支持/反对/弃权状态及可选置信度。")
add_matrix(doc, ["字段", "主要内容", "可计算信号"], [
    ("A 假设", "问题解释、适用条件、约束和潜在风险", "field_A、与请求的一致性"),
    ("E 证据", "带来源标识的事实、检索结果或核验记录", "field_E、novel_evidence、verified_evidence"),
    ("I 推断", "证据到结论的步骤和反事实/反对意见处理", "field_I、consistency、useful_correction"),
    ("C 结论", "规范化标签、候选支持状态或排序意见", "field_C、unsupported_flip"),
], widths=[Cm(2.4), Cm(8.4), Cm(5.1)], font_size=9.5)
add_body(doc, "AEIC 完整并不等同于事实正确。本发明将“结构可观察性”和“真实性验证”分开：完整度用于判断报告是否可解析；事实性由目录约束、外部证据或额外验证路径判断。这样可以避免仅凭形式完整就奖励错误报告。")
add_body(doc, "Evidence 中每一项证据使用 evidence_id、claim、source_type、source_ref、status 和 span 字段表示。source_type 用于区分目录记录、检索文档、规则结果、同伴报告和模型内部判断；source_ref 保存可回溯标识；status 为未验证、已验证或被否定；span 指明该证据支持或反对的具体结论。没有来源标识的流畅陈述只能作为主张，不能自动计入 verified_evidence。")
add_body(doc, "Conclusion 使用任务相关的有限标签空间。例如问答任务采用选项或真/假/未知；影视候选评审采用 support、oppose、abstain；排序任务增加 preference_score 和 risk_flags。控制器只聚合规范化标签，不直接比较长文本表面相似度，从而减少表达风格对投票的干扰。原始文本仍保存在 Trace 中供人工复核。")

add_heading(doc, "2.2.6  步骤S5：提取可观察语义信号", 3)
add_body(doc, "报告解析器首先执行模式校验，再在允许的信息范围内计算信号向量 s_i。字段完整度不是简单判断字段是否存在，而是要求字段非空、类型正确且结论可规范化；一致性检查 A、E、I、C 之间是否存在直接冲突；证据新颖度以相对于 Agent 可见证据集合的增量计算，重复复制同一证据不能重复得分。")
add_math(doc, "completeAEIC_i = 1[A_i≠∅ ∧ E_i≠∅ ∧ I_i≠∅ ∧ C_i∈Y]")
add_math(doc, "novelEvidence_i = |tokens(E_i) ∖ tokens(E_visible,i)| / max(1, |tokens(E_i)|)")
add_body(doc, "useful_correction 用于识别 Agent 在增加新证据或完成验证后对先前结论作出的可解释修正；unsupported_flip 用于识别在没有新增证据、没有验证事件且仅看到同伴结论时发生的立场翻转。两者均需要比较相邻轮次报告及当轮可见信息，不能仅根据最终结论是否改变来判断。")
add_math(doc, "unsupportedFlip_i(t)=1[C_i(t)≠C_i(t−1) ∧ ΔE_i(t)=0 ∧ verify_i(t)=0]")
add_body(doc, "verified_evidence 由确定性工具、真实目录、带定位信息的检索文档或人工审核结果产生。对于不能自动验证的任务，系统可以只奖励可检查的报告过程，并把真实性状态记为 unknown，而不能用模型自报置信度替代验证。所有信号都记录计算版本和输入报告哈希，以便重新计算。")

add_heading(doc, "2.2.7  步骤S6：计算信用、成本和已实现效用", 3)
add_body(doc, "控制器从报告、行动日志、允许可见的历史和可获得的验证结果中提取信号。信用由各信号的权重和得分组成，随后扣除合同中对应行动的成本，得到本轮效用。优选信号包括 AEIC 完整度、字段一致性、新颖证据、已验证证据、有用纠错和无依据翻转惩罚。")
add_math(doc, "credit_i = Σ_k w_k · s_k(r_i, h_i, y)")
add_math(doc, "utility_i = credit_i − cost_i(action_i)")
add_body(doc, "其中 r_i 为 Agent i 的结构化报告，h_i 为该 Agent 按信息策略可见的历史，y 为仅在允许结果核验时提供给结算环境的验证信息。结算器保存每个信号的原始值、权重、信用分解、成本和最终效用，以便复核。")
add_body(doc, "信用函数可以包含正向项和惩罚项。例如完整 AEIC、一致推断、新颖且核验通过的证据和有效纠错获得正信用，重复证据、无来源主张、无依据翻转和超预算资源消耗获得零信用或负信用。为避免某一项支配全部结果，结算器可以对各项截断到合同规定区间，并对总信用设置上下限。")
add_math(doc, "positive_i = w_c·complete_i + w_n·novel_i + w_v·verified_i + w_u·correction_i")
add_math(doc, "credit_i = clip(positive_i − w_f·flip_i, L, U)")
add_body(doc, "若环境在任务结束后能够取得真实标签 y，可以另外计算 outcome 信号，但该信号不得倒流到已经完成的行动选择。若真实标签不可得，系统仍可依赖过程信号完成结算。由此既支持可验证问答，也支持没有即时标准答案的搜索推荐和人工决策场景。")

add_heading(doc, "2.2.8  步骤S7：批量更新信誉", 3)
add_body(doc, "信誉以衰减累积方式更新。为避免同一请求内较早候选影响较晚候选，系统在处理同一冻结候选集时先读取统一的信誉快照，全部候选完成后再批量提交更新。")
add_math(doc, "reputation_i(t) = decay · reputation_i(t−1) + utility_i(t)")
add_body(doc, "当一个请求包含多个候选 j 时，控制器使用请求开始时的 reputation_before 为所有候选计算权重，并收集 Agent 在各候选上的贡献 u_ij。请求完成后，以均值、截断均值或合同指定的稳健统计量合并贡献，再一次性写入 reputation_after。该方式避免候选列表顺序造成先评影片和后评影片使用不同信誉。")
add_math(doc, "ρ_i(after)=decay·ρ_i(before)+mean_j(u_ij)")
add_body(doc, "信誉按命名空间隔离。优选键至少包含 agent_id、任务域和模型版本；在多用户服务中还可以加入租户或会话标识。更新采用事务、乐观锁或互斥锁，防止并发请求覆盖。若模型、提示或角色发生实质变化，系统创建新的信誉版本，不把旧信誉无条件继承给新主体。")

add_heading(doc, "2.2.9  步骤S8：形成可审计集体决策", 3)
add_body(doc, "集体决策不直接把所有文本交给中心裁判，而是对规范化结论、效用、置信度、验证证据和历史信誉进行确定性聚合。聚合结果同时保留支持证据、反对证据、弃权、参与 Agent、覆盖状态和剩余分歧。若报告覆盖不足或硬约束失败，则候选不能进入最终结果。")
add_body(doc, "对于离散结论 y，系统为每个 Agent 计算非负权重 ω_i。权重由置信度、结算效用和信誉快照共同确定，并设置下限和上限以防止单个 Agent 完全支配。支持某结论的权重进入 support，反对权重进入 oppose，弃权只进入覆盖统计。最终结论取净支持最大的标签，并同时输出第一名与第二名的差值作为共识裕量。")
add_math(doc, "ω_i = clip(conf_i · (1+ρ_i/K) · max(0,utility_i), ω_min, ω_max)")
add_math(doc, "consensus(y)=Σ_i ω_i·1[C_i=y] − λΣ_i ω_i·1[C_i opposes y]")
add_body(doc, "当共识裕量低于阈值、有效证据不足或反对意见来自高信誉 Agent 时，控制器可以触发验证升级、返回不确定结果或提交人工审核，而不是强制输出一致结论。该处理保留少数反对意见的审计价值，避免把“达成一致”本身误当成正确。")

add_heading(doc, "2.2.10  数据结构和审计Trace", 3)
add_figure(doc, "fig2_runtime_sequence.png", "图2  单次任务的执行时序与可追溯记录链")
add_body(doc, "每次任务创建唯一 request_id。Trace 按事件时间顺序保存请求解析、公开合同提交、行动选择、资源调用、AEIC 原始报告、验证结果、信用成本结算、信誉前后快照和最终决策。系统通过模式版本号校验必需字段，通过事件顺序校验防止行动早于合同或结算早于报告。")
add_matrix(doc, ["记录对象", "必需字段示例"], [
    ("请求", "request_id、原始文本、结构化约束、预算、创建时间"),
    ("合同", "contract_version、g、c、P、B、committed_at"),
    ("行动", "agent_id、action、visible_state、selected_at"),
    ("报告", "A/E/I/C、evidence_refs、raw_response、tokens、latency"),
    ("结算", "signals、credit_breakdown、cost、utility、reputation_before/after"),
    ("决策", "normalized_result、support/oppose、residual_dissent、rank、score_components"),
], widths=[Cm(3.0), Cm(12.9)], font_size=9.5)
add_body(doc, "Trace 采用只追加事件序列。每条事件包含 event_id、request_id、event_type、actor、timestamp、payload_hash 和前一事件哈希 prev_hash。通过哈希链可以检测日志被删除、替换或重排；在不需要密码学防篡改的轻量部署中，至少执行事件类型和时间顺序校验。敏感提示和用户数据可以单独加密保存，Trace 主链仅记录摘要和访问引用。")
add_math(doc, "hash_e = H(request_id || event_type || payload_hash || prev_hash_e)")
add_body(doc, "控制器在任务结束时执行完整性矩阵检查：应存在的 Agent×轮次×候选记录与实际记录逐项比对，并检查合同、行动、资源、报告和结算是否闭合。只有模式有效、路由有效、覆盖完整且无意外记录的任务才能标记为 complete；否则标记为 incomplete 并保存缺失原因。")

add_heading(doc, "2.2.11  预算约束和执行终止条件", 3)
add_body(doc, "控制器为每次请求维护资源账本。一次调用的资源消耗可由输入Token、输出Token、调用次数、验证次数和延迟的加权和表示；合同分别规定总预算和各 Agent 子预算。调度器在执行前预留预算，在调用完成后按实际使用量结算并释放未使用部分。")
add_math(doc, "resource_i = η_in·tokenIn_i + η_out·tokenOut_i + η_call·calls_i + η_v·verify_i")
add_math(doc, "Σ_i resource_i ≤ B_total  and  resource_i ≤ B_i")
add_body(doc, "当剩余预算不足以执行已选择行动时，控制器按照合同中的确定性顺序处理：优先保留硬约束验证，其次保留已经开始的AEIC报告，最后减少附加解释或后续轮次；不得在未记录的情况下偷偷更换行动。终止原因包括达到轮数、共识裕量超过阈值、预算耗尽、无有效候选或人工审核接管。")
add_body(doc, "该预算机制使系统能够在高风险请求上分配更多验证资源，在低风险请求上避免无必要的全量辩论，同时保证资源调整只依赖请求时可见状态。预算、预留、实耗和回退事件全部进入Trace，以支持成本复核和性能评测。")

add_heading(doc, "2.3  影视推荐系统实施例", 2)
add_body(doc, "影视推荐系统由交互层、请求规划器、目录检索器、公开合同控制器、多角色 Agent 执行器、确定性验证器、证据共识引擎、信誉存储和 Trace 存储组成。模块之间传递结构化对象，不直接依赖前端页面文本。交互层负责请求和结果展示；目录与验证模块负责可确定事实；大语言模型 Agent 负责偏好解释、证据组织和风险意见；共识引擎负责可重复的候选排序。")
add_matrix(doc, ["模块", "输入", "输出", "关键约束"], [
    ("请求规划器", "用户文本、Top-K、会话信息", "结构化请求、风险、预算、角色集合", "不得使用真实标签和事后效用"),
    ("目录检索器", "结构化请求、目录索引", "候选及检索分、目录证据", "候选必须具有真实目录ID"),
    ("合同控制器", "风险、预算、合同模板", "已提交公开合同", "必须先于Agent行动"),
    ("Agent执行器", "合同、局部状态、冻结候选", "行动与候选级AEIC报告", "同一候选集、独立状态"),
    ("验证器", "硬约束、目录证据、报告", "通过/拒绝及核验证据", "确定性约束优先"),
    ("共识引擎", "报告、信誉快照、检索分", "排序、分解、异议", "统一覆盖、批量信誉更新"),
    ("Trace存储", "全流程事件", "可查询审计链", "版本化、顺序校验、请求隔离"),
], widths=[Cm(2.4), Cm(4.0), Cm(4.6), Cm(4.9)], font_size=8.8)

add_heading(doc, "2.3.1  请求解析与风险预算规划", 3)
add_body(doc, "用户通过网页或其他终端输入自然语言影视请求。交互层可以保留键盘、语音输入和图片上传入口，但本发明不要求当前模型具备视觉理解能力；当后端仅支持文本时，图片入口只负责取得用户补充的文本描述。请求解析器把年份下限/上限、必选类型、排除类型等识别为硬约束，把题材、情绪、相似影片和多样性要求识别为软偏好。")
add_body(doc, "规划器只使用请求时可见特征估计风险和预算，例如硬约束数量、否定条件、Top-K 大小和请求复杂度，不使用真实答案、事后效用或最终候选标签。规划结果决定启用的逻辑 Agent、验证预算和候选短名单上限。")
add_body(doc, "结构化请求至少包含 query、required_genres、preferred_genres、excluded_genres、min_year、max_year 和 top_k。解析器先执行规则和词典匹配，再允许语言模型补充软偏好；硬约束只有在用户文本中存在明确依据时才能写入。对于“不要动作片和灾难片”，系统生成两个 excluded_genres；对于“最好是科幻”，只写入 preferred_genres，不把偏好误转成必须类型。")
add_body(doc, "风险分由硬约束数量、否定条件、多意图、结果规模和事实核验需求构成。风险分不直接决定推荐影片，只用于分配 Agent 和验证资源。当风险较低时，可只启用检索、偏好和验证角色；当存在多个排除条件或冲突偏好时，增加多样性和批评角色，并提高验证预算。")
add_math(doc, "risk = clip(b_0 + b_h·n_hard + b_n·n_negative + b_q·complexity + b_k·topK, 0, 1)")

add_heading(doc, "2.3.2  真实目录召回与冻结候选集", 3)
add_body(doc, "目录检索器在真实影片目录中执行实体召回和硬条件预过滤。一个实施例使用约18,430部 MovieLens 影片的标识、标题、年份和类型信息。召回结果按确定性检索分数排序，截取不少于 Top-K 的短名单并冻结。冻结后，所有启用 Agent 必须审查同一候选集合，控制器保存 candidate_id 列表和哈希，防止评审阶段悄然替换候选。")
add_body(doc, "硬约束由目录字段和确定性规则验证，不由大语言模型自行决定。若满足硬约束的影片少于 Top-K，系统返回实际可用数量并明确说明，不为凑满结果放松排除条件。所有最终影片必须能回指目录 ID，从而消除生成式影片实体幻觉。")
add_body(doc, "目录检索可以采用关键词、倒排索引、向量召回或混合召回。无论采用何种方法，检索器均输出 movie_id、title、year、genres、retrieval_score 和 catalog_evidence。硬约束过滤在冻结之前执行一次，在最终输出前再次执行一次，以防候选在中间处理过程中被错误替换。")
add_math(doc, "C_frozen = topL({x∈Catalog | hardPass(x,q)=1}, retrievalScore)")
add_body(doc, "L 至少等于 top_k，并可根据风险和预算扩大。冻结事件记录有序 candidate_id、L、目录版本和候选摘要哈希。后续 Agent 报告只允许引用 C_frozen 中的候选；任何目录外影片或未经授权新增候选都被验证器拒绝。")

add_heading(doc, "2.3.3  多角色Agent对统一候选集进行AEIC评审", 3)
add_body(doc, "检索 Agent 提供目录事实和检索匹配证据；偏好 Agent 判断候选与用户显式偏好的匹配；多样性 Agent 判断候选组合的覆盖和重复风险；批评 Agent 主动寻找硬约束冲突、证据不足和排序风险；验证 Agent 对年份、类型和排除项执行确定性复核。每个角色均输出候选级 AEIC 报告。")
add_body(doc, "控制器计算每个候选的 Agent 覆盖映射。只有当所有必需角色完成或按合同明确弃权时，候选才进入共识排序。该完整覆盖检查解决不同候选因接受评审数量不同而导致分数不可比的问题。")
add_body(doc, "为每个候选 j 构造覆盖向量 z_j，其中 z_ij=1 表示 Agent i 已提交模式有效的候选级报告，z_ij=0 表示缺失。必需角色集合 N_req 由执行计划确定。若任一 j 不满足所有必需角色覆盖，系统先在预算内重试缺失角色；仍失败时，把该候选标记为 coverage_incomplete，或者让全部候选统一降级到同一可用角色集合，禁止只对部分候选使用更多 Agent。")
add_math(doc, "coverage(j)=Σ_{i∈N_req} z_ij / |N_req|")
add_body(doc, "不同角色的任务边界保持明确。检索角色不能凭空扩展目录事实；偏好角色只判断候选与用户偏好的关系；多样性角色在候选集合层比较重复性；批评角色优先寻找约束冲突和反例；验证角色只输出可重复的规则结果。角色提示和输出模式均带版本号，从而能够定位某类错误来自请求解析、目录事实、偏好解释还是共识聚合。")

add_heading(doc, "2.3.4  硬约束门控与候选级共识排序", 3)
add_body(doc, "候选先经过硬约束门控；通过后再计算软偏好共识分。一个实施例中的得分分量包括冻结短名单检索分、Agent 效用汇总、已接受证据质量和反对意见扣分。各分量单独保存并显示，避免把“所有 Top-K 都是相同数字”的占位分数误当成真实共识结果。")
add_math(doc, "score_j = α·retrieval_j + β·agentUtility_j + γ·evidence_j − μ·opposition_j")
add_body(doc, "其中 opposition_j 由有效反对报告及其信誉权重计算。排序器输出候选的综合分、得分分解、支持证据、反对证据、剩余分歧和完整角色覆盖。Top-K 仅从门控通过的候选中截取。")
add_body(doc, "hardPass_j 是进入排序的先决条件，不作为可以被高软分抵消的惩罚项。也就是说，只要候选年份、必选类型或排除类型任一项不满足，候选直接被剔除，而不是通过偏好得分补偿。检索分、Agent 效用和证据分在同一请求内归一化；反对扣分根据反对报告的置信度、信誉和核验状态计算。")
add_math(doc, "hardPass_j = yearPass_j ∧ requiredGenrePass_j ∧ excludedGenrePass_j")
add_math(doc, "TopK = argsort_K({score_j | j∈C_frozen ∧ hardPass_j=1 ∧ coverage(j)=1})")
add_body(doc, "若所有候选均被门控拒绝，系统返回空结果和未满足原因；若有效候选少于 K，只返回有效候选。排序结果中保留 score_components，使前端能够分别显示短名单检索、Agent 效用、证据质量和反对扣分，避免使用统一占位数字或把目录相似度误称为综合共识分。")

add_figure(doc, "fig_system_architecture.png", "图3  影视搜索推荐实施例的处理链与系统模块")
add_figure(doc, "fig1_aeic_md_overview_rich.png", "图4  Agent执行、AEIC报告、验证结算与审计数据关系")

add_heading(doc, "2.3.5  用户交互与最近一次检索过程展示", 3)
add_body(doc, "面向用户的主页面保持常规影视检索形态，展示影片封面、标题、年份、类型、综合分和推荐理由。语音与图片等入口作为可折叠辅助能力，不占用主检索流程。检索完成后，用户可点击“查看检索过程”按钮打开侧边层或悬浮层，读取最近一次请求的执行计划、审计时间线、Agent AEIC 报告、候选级共识分解和完整 Trace，而不把内部调试界面作为首页。")
add_body(doc, "系统将面向用户的解释与内部审计分开。用户侧理由只引用已经接受的目录或验证证据，并控制长度；审计层展示合同版本、行动、Token、延迟、完整 A/E/I/C、覆盖和异议。默认页面不暴露原始系统提示、密钥或其他用户的历史信誉。最近一次检索过程以当前 request_id 查询，避免多个浏览器会话之间串用 Trace。")

add_heading(doc, "2.3.6  会话隔离、并发和异常处理", 3)
add_body(doc, "HTTP 服务为每个请求创建独立工作流上下文。信誉存储按命名空间读取快照，候选报告保存在请求局部对象中；共享缓存只保存只读目录索引和模型连接池。写入信誉和任务完成状态时使用事务，防止并发请求更新丢失。前端断线不取消已经提交的任务，用户重新连接后可按 request_id 获取状态。")
add_body(doc, "模型超时、格式错误和验证失败采用分级处理。首次格式错误可调用不增加事实的结构修复；模型超时在预算内重试；单个 Agent 失败触发覆盖检查；外部模型整体不可用时，系统可退化为目录检索与确定性验证，但必须把 mechanism_status 标记为 degraded，并禁止显示“多 Agent 已完成”的误导状态。")
add_matrix(doc, ["异常", "处理", "Trace标记"], [
    ("行动响应非法", "按合同回退Low或拒绝本轮", "invalid_action、fallback"),
    ("AEIC模式错误", "限次结构修复，不增加新事实", "schema_repair、raw_hash"),
    ("Agent超时", "预算内重试，随后执行覆盖降级", "timeout、retry、coverage"),
    ("目录或验证失败", "停止受影响候选进入Top-K", "verification_unavailable"),
    ("模型服务不可用", "仅返回目录检索结果或错误", "degraded、missing_agents"),
    ("前端断开", "后台保留任务，允许按request_id恢复", "client_disconnected"),
], widths=[Cm(3.3), Cm(7.6), Cm(5.0)], font_size=9.2)

add_heading(doc, "2.4  具体实施过程", 2)
add_heading(doc, "2.4.1  通用多智能体判断实施例", 3)
add_body(doc, "以一个具有确定答案的判断题为例，控制器加载固定公开合同：完整 AEIC 和 Verify 行动获得较高的过程信用，Low 成本最低，Imitate 只有在明确引用并新增核验信息时获得信用。五个 Agent 在第一轮只看到问题、合同和各自信誉，不看到同伴结论。每个 Agent 先返回行动，再由资源路由器执行对应提示。")
add_body(doc, "第一轮完成后，解析器将每份报告拆分为 A、E、I、C，并把结论规范化为真、假或未知。控制器计算字段完整度和内部一致性，验证器检查可定位证据。第二轮依据 P 只向 Agent 暴露允许的证据摘要，不直接暴露全部原始推理；Agent 可以维持结论、增加证据或在说明理由后修正。第三轮重复上述过程并形成最终报告。")
add_body(doc, "若某 Agent 在未获得新证据、未执行 Verify 的情况下，看到多数结论后从真改为假，则 unsupported_flip 被记录；若其通过额外验证发现原证据错误并完成修正，则 useful_correction 和 verified_evidence 被记录。结算器在每轮结束后计算信用和成本，全部问题级记录完成后更新信誉。最终决策只使用第三轮规范化结论及结算权重，同时保留少数反对意见。")
add_matrix(doc, ["阶段", "控制器动作", "Agent或工具输出", "审计重点"], [
    ("合同", "提交g、c、P、B", "无", "提交时间和版本"),
    ("选择", "发送局部状态", "唯一action", "是否先于任务报告"),
    ("执行", "路由真实资源", "原始响应、Token、延迟", "行动与路径是否一致"),
    ("解析", "模式校验和规范化", "A/E/I/C及结论标签", "字段完整与原文哈希"),
    ("验证", "查证证据和轮次变化", "信号向量", "新证据、纠错、翻转"),
    ("结算", "信用减成本", "效用、信誉更新", "逐项分解与预算"),
    ("聚合", "确定性加权", "结论、裕量、异议", "无中心生成式裁判"),
], widths=[Cm(2.2), Cm(4.8), Cm(4.7), Cm(4.2)], font_size=8.9)

add_heading(doc, "2.4.2  多约束影视请求实施例", 3)
add_body(doc, "用户请求为“推荐2010年以后的思想性科幻剧情片，不要动作片和灾难片”，Top-K 为3。请求解析器生成 min_year=2010、required_genres={Science Fiction, Drama}、excluded_genres={Action, Disaster}，并把“思想性”作为软偏好。由于同时存在年份、两个必选类型和两个排除类型，规划器将风险设为高，启用检索、偏好、多样性、批评和验证角色。")
add_body(doc, "目录检索器先从约18,430部影片中过滤年份和类型，按检索分取得长度 L 的短名单，并记录 movie_id、目录字段和候选哈希。公开合同随后提交。全部角色在同一短名单上逐候选输出 AEIC：偏好角色解释思想性与题材匹配，多样性角色检查候选之间的重复，批评角色主动查找动作或灾难类型冲突，验证角色再次根据目录字段执行硬约束。")
add_body(doc, "假设某候选同时包含 Science Fiction、Drama 和 Action。即使偏好角色给出高支持，hardPass 仍因排除类型失败，该候选不能进入 Top-K。另一个候选满足硬约束但批评角色指出推荐理由缺少可核验证据，则该候选保留，但 evidence 分下降并保存异议。最终三个候选的综合分由各自真实分量计算，不再使用相同占位分。")
add_body(doc, "返回前，控制器确认每个候选均有全部必需角色报告、全部目录引用可解析、合同事件早于行动事件，并批量更新信誉。主页面只展示最终影片和简要理由；用户点击“查看检索过程”后，可以看到执行计划、冻结短名单、各角色 A/E/I/C、硬约束结果、得分分解及剩余异议。")

add_heading(doc, "2.4.3  关键算法流程", 3)
add_numbered(doc, "步骤1：", "接收任务 q，创建 request_id，解析可见请求特征并选择执行计划。")
add_numbered(doc, "步骤2：", "生成合同 m，写入 committed 事件和合同哈希，建立信誉与预算快照。")
add_numbered(doc, "步骤3：", "向各 Agent 发送 view_i(S_t)，接收并校验行动 a_i；按照 route(a_i) 执行真实资源路径。")
add_numbered(doc, "步骤4：", "解析 AEIC 报告，保存原始响应，执行证据验证并计算信号向量 s_i。")
add_numbered(doc, "步骤5：", "按照合同 g 计算 credit_i，扣除 c(a_i) 得到 utility_i；检查预算和声明行动一致性。")
add_numbered(doc, "步骤6：", "完成规定轮次；对规范化结论执行确定性聚合或候选共识排序，保留异议。")
add_numbered(doc, "步骤7：", "以请求开始时的信誉快照计算全部候选，再批量写入 reputation_after。")
add_numbered(doc, "步骤8：", "执行期望记录矩阵、事件顺序、模式和路由完整性检查，标记 complete 或 incomplete。")

add_heading(doc, "2.5  实施例的运行与验证结果", 2)
add_body(doc, "为验证公开过程规则是否改变 Agent 的资源行动及集体决策，在 StrategyQA、MMLU-Hard 和 TruthfulQA 三个数据集上各选取150题，共450题。每个条件使用5个 Agent、3轮同步审议，相同问题、Seed、模型、信息设置和多数决策方式，仅改变公开规则。")
add_matrix(doc, ["比较条件", "高资源行动率", "第三轮多数决策准确率"], [
    ("无机制（No Mechanism）", "0.46%", "68.00%"),
    ("静态AEIC公开规则（Static AEIC）", "56.93%", "82.22%"),
    ("变化", "+56.47个百分点", "+14.22个百分点"),
], widths=[Cm(7.0), Cm(4.3), Cm(4.6)], font_size=10)
add_body(doc, "问题级配对的 Exact McNemar 检验经 Holm 校正后 p=1.23×10⁻¹¹。仅奖励最终结果的条件没有得到同样的准确率提升。该结果支持“公开的可观察过程规则能够改变已测试后端的行动分布并改善集体决策”，但不主张已证明一般激励相容、全局最优合同或跨所有模型的普遍有效性。")
add_body(doc, "影视系统还对30条多约束请求执行工程验收，共检查150个 Top-5 结果。目录实体有效性、硬约束满足、证据引用、合同先于行动的事件顺序、完整 Agent 覆盖和统一候选覆盖均通过验收。该验收说明系统链路正确和可追溯，不等同于已经证明推荐质量显著优于其他推荐算法。")
add_body(doc, "进一步地，在完整1,203条ReDial测试样本上，以测试对话切分点之前出现的影片作为可见历史、切分点之后首次推荐影片作为隐藏目标，并仅使用ReDial训练集统计影片共现，避免标签泄漏。词法召回的Hit@5为0.75%，冻结权重后的混合召回为10.81%，问题级Exact McNemar检验p=3.70×10⁻³⁴。100条Agent排序评测中，直接排序、单Agent和多Agent共识的Hit@5分别为6%、7%和7%，共识与单Agent命中集合相同。因此，本实施例支持混合召回、实体约束和审计链路的有效性，但不据此主张多Agent共识必然提高推荐准确率。")

add_body(doc, "三个任务域的效果并非相同常数：StrategyQA 的准确率由73.33%提高到77.33%，MMLU-Hard由71.33%提高到91.33%，TruthfulQA由59.33%提高到78.00%。该差异说明额外证据和验证的边际收益与任务结构有关；结果用于说明已实现机制的适用表现，不据此主张所有任务都获得相同提升。")
add_body(doc, "扩展检查包括奖励扫描、成本扫描、四类信息环境、额外随机种子、3至5个 Agent 的面板规模、AEIC字段组合扰动和历史策略扰动。奖励提高时高资源行动总体增加，成本提高时总体下降；额外随机种子和不同面板规模未改变主方向。字段和历史扰动主要表明机制在已测条件下保持稳定，不能据此声称四个字段已经逐项证明不可替代。")

add_heading(doc, "2.6  本发明的有益效果", 2)
add_numbered(doc, "（1）", "通过在生成前公开合同并绑定日志，使规则能够影响 Agent 的行动选择，而不是只在生成后对结果重加权。")
add_numbered(doc, "（2）", "通过 AEIC 报告把自由文本转化为可解析的假设、证据、推断和结论，为完整度、证据新颖性、核验证据、有用纠错和无依据翻转提供统一计算界面。")
add_numbered(doc, "（3）", "通过行动到真实资源路径的强制映射，使 Verify、AEIC 等高资源行为可由调用次数、Token、延迟和审计记录验证。")
add_numbered(doc, "（4）", "通过信用—成本结算和批量信誉更新，在保留历史可靠性的同时，避免同一请求内候选顺序造成信誉串扰。")
add_numbered(doc, "（5）", "通过确定性验证与证据共识分离，不依赖单一中心裁判模型，并能保留支持、反对、弃权和剩余分歧。")
add_numbered(doc, "（6）", "在影视推荐中通过真实目录、冻结短名单、完整覆盖和硬约束门控，降低虚构影片、漏审候选、分数不可比和约束违规。")
add_numbered(doc, "（7）", "通过请求级 Trace 将公开合同、行动、报告、资源、结算和决策连接为完整证据链，便于复现、审计和错误定位。")

add_heading(doc, "2.7  可替代实施方式", 2)
add_numbered(doc, "（1）", "基础模型可为同一模型端点的多角色实例，也可为不同模型组成的混合面板；无论采用哪种方式，均保持独立状态和明确 Agent 标识。")
add_numbered(doc, "（2）", "合同可为固定规则、基于任务风险从有限集合选择的规则，或根据历史响应面选择的规则；只要本轮合同在行动前提交且请求内不可追溯修改。")
add_numbered(doc, "（3）", "验证源可为数据库、搜索索引、规则引擎、人工审核、外部工具或第二模型；验证结果必须与证据标识绑定。")
add_numbered(doc, "（4）", "集体决策可用于问答、内容审核、学术评审、医疗辅助筛查等其他多 Agent 场景；影视推荐只是优选实施例。")
add_numbered(doc, "（5）", "信誉可按 Agent、任务域、用户会话或组织隔离存储，并采用事务或锁避免并发请求互相污染。")
add_numbered(doc, "（6）", "当外部模型调用失败时，可由确定性检索、规则验证和本地 Agent 降级完成；Trace 中必须记录降级路径，避免将降级结果伪装成完整多 Agent 共识。")
add_numbered(doc, "（7）", "AEIC 可以编码为 JSON、协议缓冲区、关系表或其他可解析结构；只要四类语义成分及其来源关系能够稳定解析和版本化。")
add_numbered(doc, "（8）", "共识聚合可以采用加权多数、概率融合、排序学习或约束优化；但应继续保留硬门控、证据来源、反对意见和得分分解，不能退化为不可解释的单一裁判输出。")
add_numbered(doc, "（9）", "信息策略可以采用首轮隔离、后续只共享证据、共享结论但隐藏理由或按拓扑传播；所有可见信息均需记录，使模仿和纠错信号可以基于真实可见历史计算。")

add_heading(doc, "3  本发明的关键技术点和拟保护点", 1)
add_numbered(doc, "1.", "一种公开合同先于多智能体行动的控制方法：合同包含信用规则、行动成本、信息可见策略和验证预算，并在报告生成之前与请求标识一起提交。")
add_numbered(doc, "2.", "一种行动—资源执行绑定方法：Agent 返回离散行动标识，控制器据此选择不同提示、Token、调用次数、同伴信息和验证路径，并记录可核验资源消耗。")
add_numbered(doc, "3.", "一种基于 AEIC 的结构化语义报告及信号提取方法：从假设、证据、推断和结论字段计算完整度、一致性、证据增量、核验证据、有用纠错及无依据翻转。")
add_numbered(doc, "4.", "一种基于可观察信号的信用—成本结算和信誉更新方法：保存信用分解、行动成本、已实现效用和信誉前后快照，并对同一候选批次采用统一信誉快照和批量更新。")
add_numbered(doc, "5.", "一种不依赖单一生成式裁判的集体决策方法：对规范化结论、证据、效用、置信度和信誉进行确定性聚合，同时保留支持、反对、弃权和剩余分歧。")
add_numbered(doc, "6.", "一种面向影视推荐的统一候选评审方法：从真实目录召回并冻结候选集合，使全部必需 Agent 对相同候选完成评审，通过完整覆盖检查后再进行共识排序。")
add_numbered(doc, "7.", "一种硬约束与软偏好分层的影视推荐方法：先以目录字段和确定性规则执行硬约束门控，再根据检索分、Agent 效用、证据质量和反对扣分计算 Top-K。")
add_numbered(doc, "8.", "一种端到端审计方法：以 request_id 连接请求、合同、行动、资源调用、AEIC 报告、验证、结算、信誉和最终决策，并校验关键事件顺序及模式完整性。")
add_numbered(doc, "9.", "一种多智能体语义共识系统：包括合同控制模块、行动路由模块、结构化报告模块、验证结算模块、信誉模块、共识模块和审计模块，各模块按照合同提交、行动选择、资源执行、报告、结算和决策的先后顺序交换数据。")
add_numbered(doc, "10.", "一种影视推荐系统：在上述多智能体语义共识系统基础上增加真实影片目录、请求规划器、候选冻结模块、完整覆盖检查和硬约束门控模块，并输出带证据、分数分解和剩余异议的Top-K。")
add_numbered(doc, "11.", "一种电子设备及计算机可读存储介质：处理器执行存储的程序时，实现本交底书所述公开合同提交、行动资源绑定、AEIC报告处理、验证结算、信誉更新、候选共识和Trace生成步骤。")

add_body(doc, "建议独立权利要求围绕“合同在报告生成前提交”“行动选择决定真实资源路径”“AEIC报告和资源日志共同产生可结算信号”“根据结算结果形成集体决策并保存时序Trace”组织。影视推荐可设置另一项独立权利要求，突出真实目录召回、冻结候选、统一覆盖、硬约束门控和候选级共识。字段名称、权重数值、Agent数量和具体基础模型宜放入从属权利要求或实施例，避免不必要缩小保护范围。")

add_heading(doc, "4  附图说明及技术材料索引", 1)
add_heading(doc, "4.1  附图说明", 2)
add_numbered(doc, "（1）", "图1为总体时序图，示出合同提交、行动选择、资源执行、AEIC报告、验证结算、信誉更新、集体决策和Trace固化。")
add_numbered(doc, "（2）", "图2为核心数据关系图，示出request_id对合同、行动、资源、报告、证据、结算、信誉和决策的关联。")
add_numbered(doc, "（3）", "图3为影视推荐处理链，示出约束解析、目录召回、短名单冻结、统一评审、硬门控、候选共识及Top-K输出。")
add_numbered(doc, "（4）", "图4为多Agent执行与审计数据关系图，示出Agent行动、AEIC字段、验证结算、跨轮状态、共识聚合以及行为、决策与成本审计指标之间的关系。")

add_heading(doc, "4.2  技术材料索引", 2)
add_matrix(doc, ["材料", "位置/说明"], [
    ("研究论文与实验材料", "项目目录 docs/papers/、experiments/results/；用于支持 AEIC 公开规则与多智能体实验结果。"),
    ("影视推荐系统实现", "项目目录 system/game_consensus_recsys/；包含请求规划、目录检索、Agent、共识引擎、HTTP服务和 Trace。"),
    ("系统工程验收", "experiments/results/movie_acceptance/acceptance_results.jsonl；包含30条多约束请求的逐项结果。"),
    ("旧版专利稿", "docs/patent/source/旧版专利.docx；仅作历史材料，不作为当前事实口径。"),
    ("专利交底书样例", "docs/patent/source/专利交底书样例.docx；仅作栏目和书面表达参考。"),
], widths=[Cm(4.2), Cm(11.7)], font_size=8.8)

add_heading(doc, "4.3  拟保护技术特征与说明书支撑关系", 2)
add_body(doc, "下表表明各拟保护技术特征均有流程、数据对象、计算关系和实施例支撑。具体模型名称、固定权重、Agent数量或单一数据集属于可替换实施参数，不宜限定为实现本发明的必要条件。")
add_matrix(doc, ["拟保护技术特征", "主要说明书支撑", "可验证载体"], [
    ("行动前提交且请求内不可追溯修改的公开合同", "2.2.2、2.2.3及图1", "合同版本、提交时间和request_id"),
    ("离散行动选择与真实资源执行路径绑定", "2.2.4、2.2.5及图1", "Prompt版本、Token、调用次数、工具和验证日志"),
    ("AEIC结构化报告与可观察信号提取", "2.2.6、2.2.7及图2", "字段对象、证据标识、信号分解"),
    ("信用—成本结算、批量信誉更新与确定性共识", "2.2.8至2.2.10", "信用分解、成本项、信誉前后快照和投票明细"),
    ("真实目录召回、候选冻结、统一覆盖和硬约束门控", "2.3.2至2.3.5及图3", "候选ID及哈希、覆盖矩阵、门控原因和分数分解"),
    ("Agent行动、AEIC报告、验证结算和审计指标", "2.2.3至2.2.6及图4", "行动记录、字段报告、信用/成本结算和可回放Trace"),
    ("贯穿完整执行链的机器可读审计Trace", "2.2.10、2.3.6及图2", "事件序号、时间戳、输入输出摘要和模式完整性检查"),
], widths=[Cm(5.0), Cm(4.8), Cm(6.1)], font_size=8.0)

doc.core_properties.title = "一种博弈驱动的多智能体语义共识方法及影视推荐系统——专利技术交底书"
doc.core_properties.subject = "多智能体语义共识、AEIC结构化报告、机制设计、影视推荐"
doc.core_properties.author = "马天成；高志鹏"
doc.core_properties.keywords = "AEIC, 多智能体, 公开合同, 语义共识, 影视推荐, 可追溯"

settings = doc.settings._element
update_fields = settings.find(qn("w:updateFields"))
if update_fields is None:
    update_fields = OxmlElement("w:updateFields")
    settings.append(update_fields)
update_fields.set(qn("w:val"), "true")

doc.save(OUT)
print(OUT)
