"""
信任评估模块 — AEIC-Stackelberg 核心

功能:
  - IC Consistency: 中英双语方向检测 + 语义相似度
  - EC Relevance: 证据-结论方向性评估
  - Evidence Score: 模式感知（parallel=consensus, sequential=novelty）
  - Neighbor Agreement: 拓扑感知邻居一致性
  - Composite Trust: power transform (无min-max归一化)
  - Quality-Gated Decision: IC×EC质量门控，少数派可override多数派
  - Stackelberg Bid构建: 4个独立信号 → 4个bid参数
"""
from __future__ import annotations

import re
import time as _time
from typing import Dict, List, Tuple

import numpy as np

from core.stackelberg import AgentBid, StackelbergGame


# ── 方向检测信号词 ─────────────────────────────────────────────

POS_SIGNALS = [
    "支持", "确认", "成立", "正确", "肯定", "能够", "可以",
    "证据充分", "说明", "证实", "吻合", "一致",
    "有利于", "符合", "验证", "证明", "确实", "的确",
    "答案是肯定", "是正确的", "是成立的",
    "supports", "affirms", "confirms", "consistent with",
    "suggests yes", "clearly yes", "correct",
    "converge on", "strongly suggests",
    "affirmative", "indeed", "feasible", "is true",
    "the answer is yes", "answer is affirmative",
]

NEG_SIGNALS = [
    "反对", "否定", "不成立", "不正确", "无法", "不能", "不支持",
    "缺乏", "没有", "不是", "矛盾", "排除", "不足以",
    "不符合", "不一致", "反驳", "推翻", "否认", "不可能",
    "答案是否定", "是错误的", "不成立的",
    "contradicts", "refutes", "does not", "cannot", "inconsistent",
    "suggests no", "clearly no", "false", "incorrect",
    "against", "fails to", "lack of", "no evidence",
    "does not follow", "does not support",
    "indicates against", "impossible", "infeasible",
    "not feasible", "unable", "never",
    "the answer is no", "answer is negative",
]


def extract_stance(conclusion: str) -> str:
    """从结论文本提取 yes/no stance。"""
    c = str(conclusion).strip().lower()
    # AEIC prompt要求C以Yes/No开头
    if c.startswith("yes"):
        return "yes"
    if c.startswith("no"):
        return "no"

    # 模式匹配: "the answer is Yes/No", "答案是肯定/否定"
    import re
    yes_patterns = [
        r"\banswer is yes\b", r"\banswer is true\b",
        r"\byes\b", r"\baffirmative", r"\bconcluding affirmatively",
        r"答案是肯定", r"答案为是", r"结论是肯定",
    ]
    no_patterns = [
        r"\banswer is no\b", r"\banswer is false\b",
        r"\bno\b", r"\bnegative", r"\bconcluding negatively",
        r"答案是否定", r"答案为否", r"结论是否定",
    ]
    pos_hits = sum(1 for p in yes_patterns if re.search(p, c))
    neg_hits = sum(1 for p in no_patterns if re.search(p, c))

    if pos_hits > neg_hits:
        return "yes"
    if neg_hits > pos_hits:
        return "no"

    # 最后回退: 中文关键词
    pos_kw = ["是", "正确", "肯定", "成立", "支持", "true"]
    neg_kw = ["否", "不", "错误", "否定", "不成立", "false"]
    pos_hits2 = sum(1 for k in pos_kw if k in c[:80])
    neg_hits2 = sum(1 for k in neg_kw if k in c[:80])
    if pos_hits2 > neg_hits2:
        return "yes"
    if neg_hits2 > pos_hits2:
        return "no"
    return "neutral"


def _count_signals(text: str, signals: List[str]) -> int:
    """计算文本中信号词出现次数（长短语优先）。"""
    t = text.lower()
    count = 0
    matched_spans = []
    # Sort by length descending so longer phrases match first
    for s in sorted(signals, key=len, reverse=True):
        sl = s.lower()
        idx = t.find(sl)
        while idx >= 0:
            span = (idx, idx + len(sl))
            if not any(ms <= idx < me or ms < idx + len(sl) <= me
                       for ms, me in matched_spans):
                count += 1
                matched_spans.append(span)
            idx = t.find(sl, idx + 1)
    return count


def detect_direction(text: str) -> str:
    """检测文本的整体方向倾向。"""
    pos = _count_signals(text, POS_SIGNALS)
    neg = _count_signals(text, NEG_SIGNALS)
    if pos > neg:
        return "yes"
    if neg > pos:
        return "no"
    return "neutral"


# ── IC Consistency (语义对齐) ─────────────────────────────────

# 结论标志词：I文本中指示最终推理方向的过渡词
CONCLUSION_MARKERS = [
    "therefore", "thus", "hence", "consequently",
    "this leads to", "this suggests", "this means",
    "we conclude", "the conclusion is", "in conclusion",
    "as a result", "it follows that", "indicating that",
    "综上", "因此", "所以", "由此可见", "总结", "结论是",
    "这表明", "这说明", "据此", "可以得出", "由此",
    "总之", "最终", "归纳", "概言之",
]


def _get_concluding_segment(inference: str) -> str:
    """提取I文本中结论性语句段落（最后一个结论标志词之后的内容）。

    原理：推理文本的最终方向通常在结论标志词之后明确表达。
    "证据表明...无法捕获...因此答案是肯定的" → 结论段="因此答案是肯定的"
    """
    text_lower = inference.lower()
    last_pos = -1
    for marker in CONCLUSION_MARKERS:
        pos = text_lower.rfind(marker.lower())
        if pos > last_pos:
            last_pos = pos

    if last_pos >= 0:
        return inference[last_pos:]

    # 无标志词时，取最后30%文本作为结论段近似
    cutoff = int(len(inference) * 0.7)
    return inference[cutoff:] if cutoff < len(inference) else inference


def _has_signal_word_boundary(text: str, signal: str) -> bool:
    """检查signal是否作为完整词/短语出现在text中（非子串）。

    中文信号词使用简单包含匹配（中文无空格分隔，不存在英文的子串歧义）。
    英文信号词使用词边界匹配（避免"correct"匹配"incorrect"）。
    """
    if any('一' <= ch <= '鿿' for ch in signal):
        return signal in text
    pattern = r'(?<![a-zA-Z])' + re.escape(signal) + r'(?![a-zA-Z])'
    return bool(re.search(pattern, text))


def _shared_concept_alignment(inference: str, conclusion: str) -> float:
    """检测I和C是否共享相同的关键概念（语义对齐检测）。

    核心原理：当I说"cannot capture, impossible"而C说"Yes: impossible"时，
    双方共享"impossible/cannot"等关键词 → 语义方向一致，即使表面极性看似矛盾。

    解决的问题：双重否定场景下，如"Is X impossible?"
    - 诚实agent: I="evidence shows cannot..." C="Yes: it is impossible"
    - 旧方法: I方向=no(否定词多), C=yes → 误判为不一致
    - 新方法: I和C都含"impossible/cannot" → 共享概念 → 判定为一致

    使用词边界匹配避免"correct"匹配"incorrect"等子串误报。
    """
    i_lower = inference.lower()
    c_lower = conclusion.lower()
    c_match = re.match(r'^(yes|no)\s*[:：]?\s*', c_lower)
    c_body = c_lower[c_match.end():] if c_match else c_lower

    if not c_body.strip():
        return 0.0

    all_signals = POS_SIGNALS + NEG_SIGNALS
    i_signals = set()
    c_signals = set()
    for s in all_signals:
        sl = s.lower()
        if _has_signal_word_boundary(i_lower, sl):
            i_signals.add(sl)
        if _has_signal_word_boundary(c_body, sl):
            c_signals.add(sl)

    shared = i_signals & c_signals
    if not shared:
        return 0.0

    denom = min(len(i_signals), len(c_signals))
    if denom == 0:
        return 0.0
    return len(shared) / denom


def compute_ic_consistency(inference: str, conclusion: str, sim_calc) -> float:
    """
    计算Inference→Conclusion语义对齐度。

    核心思路（语义对齐）：
    推理文本的"方向"不应该由全文的正/负词总数决定，因为推理体中的
    否定词往往描述的是问题本身（如"X不可能"），而非推理的结论方向。

    正确做法：只看I的**结论性语句**的方向（"因此..."之后），
    辅以**共享概念对齐**检测（I和C说的是同一件事则视为一致）。

    三层机制（短路优先级）：
    1. 共享概念对齐 → I和C共享关键词 → 必然语义一致
    2. 结论段方向匹配 → 只看I末尾的推理方向
    3. 语义相似度兜底 → 当以上都无法判断时

    最终: direction_score(0.5) + semantic_sim(0.5)
    """
    c_stance = extract_stance(conclusion)

    # ── Layer 1: 共享概念对齐 ──
    # 解决双重否定: I说"impossible,cannot" + C说"Yes: impossible" → 对齐
    concept_align = _shared_concept_alignment(inference, conclusion)
    if concept_align > 0.3:
        direction_score = 0.7 + 0.2 * min(concept_align, 1.0)
    else:
        # ── Layer 2: 结论段方向检测（只看I的结论部分） ──
        concluding = _get_concluding_segment(inference)
        conc_pos = _count_signals(concluding, POS_SIGNALS)
        conc_neg = _count_signals(concluding, NEG_SIGNALS)

        if conc_pos + conc_neg > 0:
            i_direction = "yes" if conc_pos > conc_neg else "no"
            confidence = abs(conc_pos - conc_neg) / (conc_pos + conc_neg)
            if c_stance == "neutral":
                direction_score = 0.5
            elif i_direction == c_stance:
                direction_score = 0.8 + 0.2 * confidence
            else:
                direction_score = 0.2 - 0.1 * confidence
        else:
            # ── Layer 3: 无信号时依赖语义相似度 ──
            direction_score = 0.5

    direction_score = max(0.0, min(1.0, direction_score))
    semantic_sim = sim_calc.compute(inference, conclusion)

    return 0.5 * direction_score + 0.5 * semantic_sim


# ── EC Relevance ───────────────────────────────────────────────

def compute_ec_relevance(evidence, conclusion: str, sim_calc) -> float:
    """
    计算Evidence→Conclusion支持度。

    方向一致性(0.5) + 语义相似度(0.5)
    方向一致时 direction_score 高，矛盾时低。
    """
    e_text = " ".join(evidence) if isinstance(evidence, list) else str(evidence)
    c_text = str(conclusion)

    semantic = sim_calc.compute(e_text, c_text)

    c_stance = extract_stance(c_text)
    e_direction = detect_direction(e_text)

    if c_stance == "neutral" or e_direction == "neutral":
        direction_score = 0.5
    elif e_direction == c_stance:
        direction_score = 0.85
    else:
        direction_score = 0.15

    return float(np.clip(0.5 * direction_score + 0.5 * semantic, 0.0, 1.0))


# ── Evidence Score (模式感知) ──────────────────────────────────

def _flatten_evidence(node) -> str:
    e = node.get("E", [])
    if isinstance(e, list):
        return " ".join(str(x) for x in e)
    return str(e)


def compute_evidence_consensus(nodes: List[Dict], W: np.ndarray, sim_calc) -> List[float]:
    """
    Parallel/Debate模式: 独立产出的证据相似 = 可靠。
    计算每个agent的证据与邻居证据的平均相似度。
    """
    n = len(nodes)
    scores = []
    evidence_texts = [_flatten_evidence(nd) for nd in nodes]

    for i in range(n):
        neighbors = [j for j in range(n) if j != i and W[i, j] > 0]
        if not neighbors or not evidence_texts[i].strip():
            scores.append(0.5)
            continue
        sims = []
        for j in neighbors:
            if evidence_texts[j].strip():
                s = sim_calc.compute(evidence_texts[i], evidence_texts[j])
                sims.append(s * W[i, j])
        if sims:
            w_sum = sum(W[i, j] for j in neighbors if evidence_texts[j].strip())
            scores.append(sum(sims) / max(w_sum, 1e-9))
        else:
            scores.append(0.5)
    return scores


def compute_evidence_novelty(nodes: List[Dict], sim_calc) -> List[float]:
    """
    Sequential模式: 新信息 = 有价值。
    计算每个agent的证据相对于之前所有agent的新颖度。
    """
    n = len(nodes)
    scores = []
    evidence_texts = [_flatten_evidence(nd) for nd in nodes]

    for i in range(n):
        if i == 0:
            scores.append(1.0)
            continue
        prev_e = " ".join(evidence_texts[j] for j in range(i))
        cur_e = evidence_texts[i]
        if not cur_e.strip() or not prev_e.strip():
            scores.append(0.5)
            continue
        sim_to_prev = sim_calc.compute(cur_e, prev_e)
        novelty = 1.0 - sim_to_prev
        scores.append(max(0.1, novelty))
    return scores


def compute_evidence_score(nodes: List[Dict], W: np.ndarray, sim_calc,
                           mode: str = "parallel") -> List[float]:
    """模式感知的证据评估。"""
    if mode == "sequential":
        return compute_evidence_novelty(nodes, sim_calc)
    else:
        return compute_evidence_consensus(nodes, W, sim_calc)


# ── Neighbor Agreement ─────────────────────────────────────────

def compute_neighbor_agreement(nodes: List[Dict], W: np.ndarray) -> List[float]:
    """拓扑感知的邻居一致性。"""
    n = len(nodes)
    stances = [extract_stance(nd.get("C", "")) for nd in nodes]
    scores = []
    for i in range(n):
        neighbors = [j for j in range(n) if j != i and W[i, j] > 0]
        if not neighbors:
            scores.append(0.5)
            continue
        agree_count = sum(1 for j in neighbors if stances[j] == stances[i])
        scores.append(agree_count / len(neighbors))
    return scores


# ── Composite Trust ────────────────────────────────────────────

def compute_composite_trust(ic: List[float], ec: List[float],
                            ev: List[float], ag: List[float],
                            mode: str = "parallel") -> List[float]:
    """
    合成信任值，使用power transformation增加区分度。
    无min-max归一化，保持绝对标准。
    """
    if mode == "sequential":
        weights = (0.40, 0.25, 0.20, 0.15)
    elif mode == "debate":
        weights = (0.35, 0.25, 0.20, 0.20)
    else:
        weights = (0.35, 0.25, 0.25, 0.15)

    trust = []
    for i in range(len(ic)):
        raw = (weights[0] * ic[i] + weights[1] * ec[i] +
               weights[2] * ev[i] + weights[3] * ag[i])
        trust.append(raw ** 1.5)
    return trust


# ── Stackelberg Bid构建 ────────────────────────────────────────

def build_stackelberg_bids(ic: List[float], ec: List[float],
                           ev: List[float], agreement: List[float],
                           n: int) -> List[AgentBid]:
    """
    从4个独立信号构建Stackelberg bid参数。

    IC → alignment_willingness (内部一致 = 愿意配合)
    agreement → revision_cost (少数派 = 昂贵)
    EC → max_revision_depth (证据好 = 有修改空间)
    IC+EC → historical_alignment (综合可靠性)
    """
    bids = []
    for i in range(n):
        bids.append(AgentBid(
            agent_id=f"agent-{i}",
            model="deepseek-chat",
            alignment_willingness=max(0.05, ic[i]),
            revision_cost=max(5.0, 30.0 * (1.0 - agreement[i])),
            max_revision_depth=float(np.clip(ec[i], 0.1, 0.95)),
            historical_alignment=0.5 * ic[i] + 0.5 * ec[i],
        ))
    return bids


# ── Quality-Gated Decision ────────────────────────────────────

def quality_gated_decision(stances: List[str], trust_weights: List[float],
                           ic_scores: List[float], ec_scores: List[float]) -> Tuple[str, str]:
    """
    质量门控决策：当少数派质量显著高于多数派时可override。

    Uses relative quality ratio (robust across different similarity methods)
    rather than absolute thresholds.

    Returns: (decision_stance, mechanism)
      mechanism: "unanimous" | "quality_override" | "weighted_vote"
    """
    n = len(stances)
    if n == 0:
        return "neutral", "empty"

    vote_scores = {}
    for i in range(n):
        vote_scores[stances[i]] = vote_scores.get(stances[i], 0.0) + trust_weights[i]
    majority_stance = max(vote_scores, key=vote_scores.get)

    quality = [ic * ec for ic, ec in zip(ic_scores, ec_scores)]

    majority_idx = [i for i in range(n) if stances[i] == majority_stance]
    minority_idx = [i for i in range(n) if stances[i] != majority_stance]

    if not minority_idx:
        return majority_stance, "unanimous"

    majority_quality = float(np.mean([quality[i] for i in majority_idx]))
    minority_quality = float(max(quality[i] for i in minority_idx))
    quality_gap = minority_quality - majority_quality
    quality_ratio = minority_quality / max(majority_quality, 0.01)
    majority_fraction = len(majority_idx) / n

    # Override conditions (relative + modest absolute):
    # 1. minority best agent quality > 2× majority average (relative dominance)
    # 2. quality gap > 0.05 (prevents noise-level overrides)
    # 3. majority not overwhelming (<=80%)
    if quality_ratio > 2.0 and quality_gap > 0.05 and majority_fraction <= 0.8:
        best_minority = minority_idx[int(np.argmax([quality[i] for i in minority_idx]))]
        return stances[best_minority], "quality_override"

    return majority_stance, "weighted_vote"


# ── 完整Pipeline入口 ──────────────────────────────────────────

def evaluate_and_decide(nodes: List[Dict], W: np.ndarray, sim_calc,
                        mode: str = "parallel",
                        use_stackelberg: bool = True) -> Dict:
    """
    完整的信任评估+决策pipeline。

    Args:
        nodes: List of AEIC dicts [{"A":..,"E":..,"I":..,"C":..}, ...]
        W: adjacency/weight matrix (n×n)
        sim_calc: SimilarityCalculator instance
        mode: "parallel" | "sequential" | "debate"
        use_stackelberg: 是否启用Stackelberg博弈调整权重

    Returns:
        {
            "decision": str,  # "yes" or "no"
            "mechanism": str,  # "unanimous"/"quality_override"/"weighted_vote"
            "trust": List[float],
            "ic": List[float],
            "ec": List[float],
            "ev": List[float],
            "agreement": List[float],
            "theta": List[float] (if stackelberg),
        }
    """
    n = len(nodes)
    if n == 0:
        return {"decision": "neutral", "mechanism": "empty",
                "trust": [], "ic": [], "ec": [], "ev": [], "agreement": [],
                "timing": {}}

    # Step 1: 计算4个信任维度
    _t0 = _time.time()
    ic = [compute_ic_consistency(nd.get("I", ""), nd.get("C", ""), sim_calc)
          for nd in nodes]
    ec = [compute_ec_relevance(nd.get("E", []), nd.get("C", ""), sim_calc)
          for nd in nodes]
    _t_ic_ec = _time.time() - _t0

    _t1 = _time.time()
    ev = compute_evidence_score(nodes, W, sim_calc, mode=mode)
    _t_ev = _time.time() - _t1

    agreement = compute_neighbor_agreement(nodes, W)

    # Step 2: 合成信任
    trust = compute_composite_trust(ic, ec, ev, agreement, mode=mode)

    # Step 3: Stackelberg博弈调整（可选）
    # W全零时跳过：peer_pressure=0导致θ退化为无意义噪声
    _t2 = _time.time()
    theta = [0.5] * n
    w_has_edges = W.any() if hasattr(W, 'any') else np.any(W > 0)
    if use_stackelberg and n >= 2 and w_has_edges:
        bids = build_stackelberg_bids(ic, ec, ev, agreement, n)
        game = StackelbergGame(num_agents=n, W=W)
        history = game.run(bids)
        theta = list(history[-1]["theta"])
        theta_arr = np.array(theta)
        if theta_arr.max() > theta_arr.min():
            theta_norm = (theta_arr - theta_arr.min()) / (theta_arr.max() - theta_arr.min())
        else:
            theta_norm = np.ones(n) * 0.5
        trust = [t * (0.5 + 0.5 * tn) for t, tn in zip(trust, theta_norm)]
    _t_sg = _time.time() - _t2

    # Step 4: 质量门控决策
    stances = [extract_stance(nd.get("C", "")) for nd in nodes]
    decision, mechanism = quality_gated_decision(stances, trust, ic, ec)

    return {
        "decision": decision,
        "mechanism": mechanism,
        "trust": trust,
        "ic": ic,
        "ec": ec,
        "ev": ev,
        "agreement": agreement,
        "theta": theta,
        "stances": stances,
        "timing": {
            "ic_ec_time_s": round(_t_ic_ec, 4),
            "evidence_time_s": round(_t_ev, 4),
            "stackelberg_time_s": round(_t_sg, 4),
        },
    }


# ── 简单方法对照 ──────────────────────────────────────────────

def majority_vote(nodes: List[Dict]) -> str:
    """简单多数投票 baseline。"""
    stances = [extract_stance(nd.get("C", "")) for nd in nodes]
    from collections import Counter
    counts = Counter(stances)
    if not counts:
        return "neutral"
    return counts.most_common(1)[0][0]


def quality_weighted_vote(nodes: List[Dict], sim_calc) -> Tuple[str, List[float]]:
    """质量加权投票（无Stackelberg、无门控）。"""
    n = len(nodes)
    ic = [compute_ic_consistency(nd.get("I", ""), nd.get("C", ""), sim_calc)
          for nd in nodes]
    ec = [compute_ec_relevance(nd.get("E", []), nd.get("C", ""), sim_calc)
          for nd in nodes]
    weights = [i * e for i, e in zip(ic, ec)]

    stances = [extract_stance(nd.get("C", "")) for nd in nodes]
    vote_scores = {}
    for i in range(n):
        vote_scores[stances[i]] = vote_scores.get(stances[i], 0.0) + weights[i]

    if not vote_scores:
        return "neutral", weights
    decision = max(vote_scores, key=vote_scores.get)
    return decision, weights
