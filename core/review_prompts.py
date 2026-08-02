"""
AEIC-Review 结构化评审协议 — 提示词与解析器

将 AEIC 四层结构映射到论文评审任务：
  A (Assessment): 论文核心主张 + 评判框架（主张成立需要满足什么条件）
  E (Evidence):   2-4 条具体优点/缺点，每条锚定论文内容并标注 (+)/(-)
  I (Inference):  权衡推理 —— soundness/significance/novelty/clarity 的取舍
  C (Conclusion): 必须 "Accept" 或 "Reject" 开头 + "Rating: X/10" + 一句理由

设计原则：
  - 与 QA 版 AEIC 同构（JSON 输出、C 字段立场前缀），复用现有解析与信号计算管线
  - 对齐 ICLR 评审 rubric（soundness, significance, novelty, clarity）
  - 评审员人设仅作温和多样性（同质后端下的人格扰动），不设 stance 偏向
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

# ── 评审员人设（温和多样性，无立场偏向）─────────────────────
REVIEWER_PERSONAS = [
    {
        "id": 0,
        "name": "Soundness-Focused Reviewer",
        "system": (
            "You are a reviewer who prioritizes technical soundness and rigor. "
            "You carefully check whether claims are supported by evidence and methodology. "
            "You are strict about experimental design but fair in weighing contributions."
        ),
        "temperature": 0.3,
    },
    {
        "id": 1,
        "name": "Significance-Focused Reviewer",
        "system": (
            "You are a reviewer who prioritizes significance and potential impact. "
            "You evaluate whether the work addresses important problems and could "
            "change how the community thinks or builds systems."
        ),
        "temperature": 0.5,
    },
    {
        "id": 2,
        "name": "Novelty-Focused Reviewer",
        "system": (
            "You are a reviewer who prioritizes novelty and originality. "
            "You assess how the work differs from prior art and whether it offers "
            "genuinely new insights rather than incremental variations."
        ),
        "temperature": 0.5,
    },
    {
        "id": 3,
        "name": "Clarity-Focused Reviewer",
        "system": (
            "You are a reviewer who prioritizes clarity and reproducibility. "
            "You check whether the paper is well-organized, claims are precisely stated, "
            "and whether others could reproduce the results from the description."
        ),
        "temperature": 0.4,
    },
    {
        "id": 4,
        "name": "Balanced Reviewer",
        "system": (
            "You are a balanced reviewer who weighs all criteria — soundness, "
            "significance, novelty, and clarity — without letting any single one dominate. "
            "You explicitly consider both strengths and weaknesses before deciding."
        ),
        "temperature": 0.4,
    },
]

# ── AEIC-Review 主提示词 ────────────────────────────────────
AEIC_REVIEW_SYSTEM = """You are one reviewer on a peer-review panel for a top ML conference. Review the submission using the strict AEIC four-layer structure:

A (Assessment): State the paper's core claim in 1-2 sentences, then your evaluation frame — what would need to be true for this paper to be a strong acceptance.
E (Evidence): List 2-4 concrete observations. Each MUST be anchored to specific content of the submission (a claim, method choice, experiment, missing baseline, unclear notation, etc.). Prefix each with "(+)" for a strength or "(-)" for a weakness.
I (Inference): Weigh the evidence across the four criteria — soundness, significance, novelty, clarity. First argue the case for acceptance, then the case for rejection, then resolve the tension (2-4 sentences).
C (Conclusion): MUST start with exactly "Accept" or "Reject" (no other leading word). Then on the same line write "Rating: X/10" (integer 1-10, where 6+=accept-worthy, 5=borderline, <=4=reject-worthy), followed by a one-sentence core justification.

Principles: judge by the evidence in the submission itself, not by reputation or gut feeling. A borderline paper should receive a borderline rating.

{persona_instruction}

Output STRICT JSON (no markdown fences):
{{"A": "assessment text", "E": ["(+) strength ...", "(-) weakness ...", ...], "I": "inference text", "C": "Accept/Reject Rating: X/10 justification"}}"""

AEIC_REVIEW_USER = """Title: {title}

Abstract:
{abstract}

Keywords: {keywords}

Write your AEIC review of this submission."""

# ── 面板讨论（debate）上下文模板 ────────────────────────────
PANEL_DISCUSSION_CONTEXT = """[Panel discussion — Round {round}]

You are discussing this submission with {n_peers} fellow reviewers. Their current positions are:

{peer_positions}

Title: {title}

Re-examine your AEIC review in light of your peers' positions. You may defend your stance with evidence, or update it if peers raised points you genuinely missed. Both holding and changing your position are acceptable — what matters is that your final judgment is evidence-driven.

Output your updated AEIC review in the same strict JSON format."""


def build_review_messages(title: str, abstract: str, keywords: List[str],
                          persona_id: int = 0,
                          discussion_context: Optional[str] = None) -> List[Dict[str, str]]:
    """构建评审生成的 messages。discussion_context 非空时为讨论轮。"""
    persona = REVIEWER_PERSONAS[persona_id % len(REVIEWER_PERSONAS)]
    system = AEIC_REVIEW_SYSTEM.format(
        persona_instruction=f"Your reviewing style: {persona['system']}"
    )
    user = AEIC_REVIEW_USER.format(
        title=title,
        abstract=abstract,
        keywords=", ".join(keywords) if keywords else "N/A",
    )
    if discussion_context:
        user = discussion_context + "\n\n---\n\n" + user
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def format_peer_positions(reviews: List[Dict], exclude_id: Optional[int] = None) -> str:
    """把其他评审员的 C+I 压缩成讨论上下文（只暴露立场与核心推理，不暴露证据细节以控制长度）。"""
    lines = []
    for r in reviews:
        if exclude_id is not None and r["reviewer_id"] == exclude_id:
            continue
        stance, rating = parse_stance_rating(r["aeic"]["C"])
        stance_str = f"{stance or 'neutral'} (rating {rating if rating is not None else '?'}/10)"
        inference = r["aeic"]["I"][:300]
        lines.append(f"Reviewer {r['reviewer_id']}: {stance_str} — {inference}")
    return "\n".join(lines) if lines else "(no peer positions available)"


# ── 受调控讨论（Study 2b 干预指令，方向中立、证据锚定）────────────
MINORITY_REGULATION = """
[Panel moderation] You currently hold a MINORITY position in this panel.
- Change your position ONLY if the majority has rebutted specific items in your Evidence (E) with concrete evidence of their own.
- If you change your stance or rating, explicitly name the specific evidence that persuaded you.
- If no specific rebuttal was offered, keep your position and state briefly why it survives.
"""

MAJORITY_REGULATION = """
[Panel moderation] You currently hold the MAJORITY position in this panel.
- Before urging convergence, address the minority's Evidence (E) items with specific rebuttals grounded in the paper.
- Do not cite the majority count itself as a reason to agree.
"""


# ── Meta-Reviewer（star 拓扑的 hub 综合）────────────────────
META_REVIEW_SYSTEM = """You are the Area Chair writing a meta-review. You receive the structured AEIC reviews of {n_reviewers} reviewers for one submission.

Synthesize a final panel decision by:
1. Summarizing the key strengths and weaknesses raised (note which reviewers raised them).
2. Weighing the credibility of each point — specific evidence-anchored claims count more than generic impressions.
3. Arriving at a final recommendation.

Your output MUST end with a final line in exactly this format:
DECISION: Accept|Reject Rating: X/10

Be independent: do not simply average the reviewers' ratings; resolve disagreements by examining which side has stronger evidence."""

META_REVIEW_USER = """Title: {title}

Abstract:
{abstract}

Reviewer reports:
{review_reports}

Write the meta-review and final DECISION line."""


def build_meta_review_messages(title: str, abstract: str,
                               reviews: List[Dict]) -> List[Dict[str, str]]:
    """构建 meta-reviewer（AC 综合）的 messages。reviews 为 [{reviewer_id, aeic}]。"""
    reports = []
    for r in reviews:
        aeic = r["aeic"]
        stance, rating = parse_stance_rating(aeic["C"])
        reports.append(
            f"--- Reviewer {r['reviewer_id']} (stance: {stance or '?'}, rating: {rating if rating is not None else '?'}/10) ---\n"
            f"Assessment: {aeic['A']}\n"
            f"Evidence: {'; '.join(aeic['E'])}\n"
            f"Inference: {aeic['I']}\n"
            f"Conclusion: {aeic['C']}"
        )
    system = META_REVIEW_SYSTEM.format(n_reviewers=len(reviews))
    user = META_REVIEW_USER.format(title=title, abstract=abstract,
                                   review_reports="\n\n".join(reports))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def parse_meta_decision(raw: str) -> tuple:
    """解析 meta-review 的 DECISION 行 → (stance, rating)。"""
    if not raw:
        return None, None
    m = re.search(r'(?i)decision\s*:\s*(accept|reject)\b', raw)
    stance = m.group(1).lower() if m else None
    m2 = re.search(r'(?i)decision[^\n]*rating\D{0,12}(\d{1,2})', raw)
    if not m2:
        m2 = re.search(r'(?i)rating\D{0,12}(\d{1,2})\s*/\s*10', raw)
    rating = int(m2.group(1)) if m2 and 1 <= int(m2.group(1)) <= 10 else None
    return stance, rating


# ── 解析器 ──────────────────────────────────────────────────
def parse_stance_rating(c_text: str) -> tuple:
    """从 C 字段提取 (stance, rating)。
    stance ∈ {"accept", "reject", None}; rating ∈ [1,10] int or None。
    骑墙写法（如 "Accept/Reject"）→ stance=None，回退由 rating>=6 派生
    （与 ICLR 评分语义及 DeepSeek 自身行为 100% 一致）。
    """
    if not c_text:
        return None, None
    text = c_text.strip()
    # 截断 JSON 救援：C 字段被回退为整段原始输出时，提取最后一个 "C" 字段碎片
    if text.startswith("```") or text.startswith('{'):
        mC = re.search(r'"C"\s*:\s*"(.*)', text, re.S)
        if mC:
            text = mC.group(1)
    stance = None
    hedged = bool(re.match(
        r'(?i)^\s*(weak\s+)?(accept|reject)\s*[/|&]\s*(accept|reject)\b', text))
    if not hedged:
        m = re.match(r'(?i)^\s*(weak\s+)?(accept|reject)\b', text)
        if m:
            stance = m.group(2).lower()
    rating = None
    m2 = re.search(r'(?i)rating\D{0,12}(\d{1,2})\s*(?:/\s*10)?', text)
    if m2:
        v = int(m2.group(1))
        if 1 <= v <= 10:
            rating = v
    if stance is None and rating is not None:
        stance = "accept" if rating >= 6 else "reject"
    return stance, rating


def parse_aeic_review(raw: str) -> Dict:
    """解析 LLM 的 AEIC-Review JSON 输出，带多级回退。"""
    if raw is None:
        return {"A": "", "E": [], "I": "", "C": ""}
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # 平衡括号回退
        result = None
        for m in re.finditer(r'\{', text):
            depth = 0
            for k in range(m.start(), len(text)):
                if text[k] == '{':
                    depth += 1
                elif text[k] == '}':
                    depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[m.start():k + 1])
                    except (json.JSONDecodeError, TypeError):
                        result = None
                    break
            if result is not None:
                break
    if not isinstance(result, dict):
        # 正则字段级提取（容忍非法 JSON：未转义换行、缺引号、截断等）
        def _str_field(name: str) -> str:
            m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % name, text, re.S)
            if m:
                return m.group(1)
            m = re.search(r'"%s"\s*:\s*"(.*)' % name, text, re.S)  # 截断碎片
            return m.group(1) if m else ""
        a = _str_field("A")
        i_ = _str_field("I")
        c = _str_field("C")
        e_list: list = []
        mE = re.search(r'"E"\s*:\s*\[(.*?)\]', text, re.S)
        if mE:
            e_list = re.findall(r'"((?:[^"\\]|\\.)*)"', mE.group(1), re.S)
        if a or i_ or c or e_list:
            return {"A": a, "E": [str(x) for x in e_list], "I": i_, "C": c}
        return {"A": "", "E": [], "I": raw[:500], "C": raw}
    e = result.get("E", [])
    if isinstance(e, str):
        e = [e]
    return {
        "A": str(result.get("A", "")),
        "E": [str(x) for x in e],
        "I": str(result.get("I", "")),
        "C": str(result.get("C", "")),
    }
