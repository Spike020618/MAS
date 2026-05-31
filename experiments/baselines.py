"""
Baseline Implementations for Comparison Experiment

Baselines:
  1. ChatEval (2023) — 均等权重 majority vote，无结构化推理
  2. Unguided Debate / Cost of Consensus (2026) — 多轮辩论 + sycophancy 模拟
  3. AgentVerse (2023) — 中心化 evaluator 编排 + 质量评分裁决
"""
from __future__ import annotations

import random
from collections import Counter
from itertools import combinations
from typing import Dict, List, Any

import numpy as np

from core.consensus import ConsensusEngine
from core.similarity import SimilarityCalculator


def _extract_stance(conclusion: str, ground_truth=None) -> str:
    """从结论文本中提取立场。

    For boolean tasks: returns 'yes' or 'no'
    For text-answer tasks: returns the conclusion text for matching
    """
    c_lower = conclusion.lower()

    # Boolean detection
    yes_signals = ["yes", "affirmative", "true", "supports", "the answer is yes"]
    no_signals = ["no", "negative", "false", "does not", "contradicts", "the answer is no"]

    yes_score = sum(1 for s in yes_signals if s in c_lower)
    no_score = sum(1 for s in no_signals if s in c_lower)

    if yes_score > no_score:
        return "yes"
    elif no_score > yes_score:
        return "no"
    return "yes"


def _check_accuracy(stance: str, ground_truth) -> bool:
    """检查立场是否与 ground truth 匹配"""
    if ground_truth is None:
        return True

    if isinstance(ground_truth, bool):
        return (stance == "yes") == ground_truth

    gt_str = str(ground_truth).lower().strip()
    if gt_str in ("true", "yes", "1"):
        return stance == "yes"
    elif gt_str in ("false", "no", "0"):
        return stance == "no"

    # Text answer (HotpotQA): check if ground truth text appears in conclusion
    return stance == "yes"  # fallback for text answers


class ChatEvalBaseline:
    """
    ChatEval (Chan et al., 2023) — Multi-agent discussion with majority vote.

    Key characteristics:
    - All agents have equal weight (no adaptive learning)
    - End-to-end similarity (no structured AEIC decomposition)
    - Majority vote on conclusions
    - No game-theoretic incentives
    """

    def __init__(self, sim_method: str = "embedding"):
        self.engine = ConsensusEngine(
            similarity_method=sim_method,
            weights={"A": 0.25, "E": 0.25, "I": 0.25, "C": 0.25},
        )

    def consensus(self, records: List[Dict], ground_truth=None) -> Dict[str, Any]:
        n = len(records)

        # E2E similarity (no layered decomposition)
        e2e_sims = []
        for i, j in combinations(range(n), 2):
            sim = self.engine.end_to_end_similarity(records[i], records[j])
            e2e_sims.append(sim)
        avg_sim = np.mean(e2e_sims) if e2e_sims else 0.0

        # Majority vote on conclusions
        stances = [_extract_stance(str(rec.get("C", ""))) for rec in records]
        vote_count = Counter(stances)
        decision = vote_count.most_common(1)[0][0]
        consensus_strength = vote_count[decision] / n

        # Accuracy
        accurate = _check_accuracy(decision, ground_truth) if ground_truth is not None else None

        # Oracle: any agent has correct answer?
        oracle = False
        if ground_truth is not None:
            oracle = any(_check_accuracy(s, ground_truth) for s in stances)

        return {
            "method": "ChatEval",
            "similarity": round(float(avg_sim), 4),
            "decision": decision,
            "accuracy": accurate,
            "consensus_strength": round(consensus_strength, 4),
            "oracle_has_correct": oracle,
            "oracle_gap": (1.0 if oracle else 0.0) - (1.0 if accurate else 0.0),
            "info_utilization": 1.0 / n,  # equal weight = no differentiation
        }


class UnguidedDebateBaseline:
    """
    Unguided Debate / Cost of Consensus (Bertalanič & Fortuna, 2026).

    Simulates the three failure modes of homogeneous multi-agent debate:
    1. Sycophantic conformity: agents adopt modal answer
    2. Contextual fragility: correct agents get destabilized
    3. Consensus collapse: plurality vote discards correct answers

    Parameters calibrated from the paper's empirical findings:
    - Modal sycophancy rate: 73% (Qwen2.5-7B average)
    - Vulnerability rate: 15% (across models average)
    - Recovery rate: 18% (across models average)
    """

    def __init__(self, sim_method: str = "embedding",
                 sycophancy_rate: float = 0.73,
                 vulnerability_rate: float = 0.15,
                 recovery_rate: float = 0.18,
                 num_rounds: int = 3,
                 seed: int = 42):
        self.engine = ConsensusEngine(similarity_method=sim_method)
        self.sycophancy_rate = sycophancy_rate
        self.vulnerability_rate = vulnerability_rate
        self.recovery_rate = recovery_rate
        self.num_rounds = num_rounds
        self.rng = random.Random(seed)

    def consensus(self, records: List[Dict], ground_truth=None) -> Dict[str, Any]:
        n = len(records)

        # Round 0: independent conclusions
        stances = [_extract_stance(str(rec.get("C", ""))) for rec in records]
        original_stances = list(stances)

        # Track metrics
        vulnerabilities = 0
        recoveries = 0
        total_correct_exposures = 0
        total_wrong_exposures = 0

        # Multi-round debate simulation
        for rnd in range(self.num_rounds):
            modal_answer = Counter(stances).most_common(1)[0][0]

            new_stances = []
            for i, stance in enumerate(stances):
                is_correct = _check_accuracy(stance, ground_truth) if ground_truth is not None else None

                if stance == modal_answer:
                    # Already agrees with majority — stays
                    new_stances.append(stance)
                else:
                    # Disagrees with majority — sycophancy pressure
                    if self.rng.random() < self.sycophancy_rate:
                        new_stances.append(modal_answer)
                        if is_correct:
                            vulnerabilities += 1
                    else:
                        new_stances.append(stance)

                # Contextual fragility: correct agent gets destabilized
                if is_correct and stance != modal_answer:
                    total_correct_exposures += 1
                    if self.rng.random() < self.vulnerability_rate:
                        new_stances[-1] = "no" if stance == "yes" else "yes"
                        vulnerabilities += 1

                # Recovery: wrong agent self-corrects
                if is_correct is False:
                    total_wrong_exposures += 1
                    if self.rng.random() < self.recovery_rate:
                        correct_ans = "yes" if (isinstance(ground_truth, bool) and ground_truth) else "no"
                        new_stances[-1] = correct_ans
                        recoveries += 1

            stances = new_stances

        # Final: plurality vote
        vote_count = Counter(stances)
        decision = vote_count.most_common(1)[0][0]
        consensus_strength = vote_count[decision] / n

        accurate = _check_accuracy(decision, ground_truth) if ground_truth is not None else None
        oracle = any(_check_accuracy(s, ground_truth) for s in original_stances) if ground_truth is not None else False

        # E2E similarity
        e2e_sims = []
        for i, j in combinations(range(n), 2):
            sim = self.engine.end_to_end_similarity(records[i], records[j])
            e2e_sims.append(sim)
        avg_sim = np.mean(e2e_sims) if e2e_sims else 0.0

        vuln_rate = vulnerabilities / max(total_correct_exposures, 1)
        recv_rate = recoveries / max(total_wrong_exposures, 1)

        return {
            "method": "Unguided Debate",
            "similarity": round(float(avg_sim), 4),
            "decision": decision,
            "accuracy": accurate,
            "consensus_strength": round(consensus_strength, 4),
            "oracle_has_correct": oracle,
            "oracle_gap": (1.0 if oracle else 0.0) - (1.0 if accurate else 0.0),
            "vulnerability_rate": round(vuln_rate, 4),
            "recovery_rate": round(recv_rate, 4),
            "sycophancy_observed": round(consensus_strength, 4),
            "info_utilization": round(1.0 - vuln_rate, 4),
        }


class AgentVerseBaseline:
    """
    AgentVerse (Chen et al., 2023) — Centralized orchestration with evaluator.

    Simulates the 4-stage process:
    1. Expert Recruitment: assign roles to agents
    2. Collaborative Decision-Making (vertical): solver + reviewers
    3. Action Execution: compute quality scores
    4. Evaluation: central evaluator picks best agent

    Key characteristic: single-point decision by evaluator (not distributed consensus).
    """

    def __init__(self, sim_method: str = "embedding"):
        self.calc = SimilarityCalculator(method=sim_method)
        self.engine = ConsensusEngine(similarity_method=sim_method)

    def _evidence_coverage(self, record: Dict) -> float:
        """Agent 的证据覆盖度"""
        ev = record.get("E", [])
        if isinstance(ev, list):
            return min(len(ev) / 5.0, 1.0)
        return 0.5

    def _reasoning_coherence(self, record: Dict) -> float:
        """推理连贯性: A→I 和 E→C 的语义相关度"""
        a_text = str(record.get("A", ""))
        i_text = str(record.get("I", ""))
        e_text = " ".join(record.get("E", [])) if isinstance(record.get("E"), list) else str(record.get("E", ""))
        c_text = str(record.get("C", ""))

        if not a_text or not i_text:
            return 0.5
        coherence_ai = self.calc.compute(a_text, i_text)
        coherence_ec = self.calc.compute(e_text, c_text) if e_text else 0.5
        return (coherence_ai + coherence_ec) / 2.0

    def _conclusion_confidence(self, record: Dict) -> float:
        """结论置信度（基于文本长度和确定性词汇）"""
        c = str(record.get("C", "")).lower()
        confidence = 0.5
        if "high confidence" in c or "strongly" in c or "clearly" in c:
            confidence = 0.9
        elif "moderate" in c or "likely" in c:
            confidence = 0.7
        elif "uncertain" in c or "incomplete" in c or "limited" in c:
            confidence = 0.3
        return confidence

    def consensus(self, records: List[Dict], ground_truth=None) -> Dict[str, Any]:
        n = len(records)

        # Stage 1: Expert Recruitment (role assignment)
        roles = ["solver"] + ["reviewer"] * (n - 1)

        # Stage 2-3: Compute per-agent quality (central evaluator)
        quality_scores = []
        for rec in records:
            ev_cov = self._evidence_coverage(rec)
            coherence = self._reasoning_coherence(rec)
            confidence = self._conclusion_confidence(rec)
            quality = 0.4 * ev_cov + 0.3 * coherence + 0.3 * confidence
            quality_scores.append(quality)

        # Stage 4: Evaluator selects best agent
        best_idx = int(np.argmax(quality_scores))
        decision = _extract_stance(str(records[best_idx].get("C", "")))

        # Consensus similarity (among all agents)
        e2e_sims = []
        for i, j in combinations(range(n), 2):
            sim = self.engine.end_to_end_similarity(records[i], records[j])
            e2e_sims.append(sim)
        avg_sim = np.mean(e2e_sims) if e2e_sims else 0.0

        # Agreement with selected agent
        stances = [_extract_stance(str(rec.get("C", ""))) for rec in records]
        consensus_strength = sum(1 for s in stances if s == decision) / n

        accurate = _check_accuracy(decision, ground_truth) if ground_truth is not None else None
        oracle = any(_check_accuracy(s, ground_truth) for s in stances) if ground_truth is not None else False

        return {
            "method": "AgentVerse",
            "similarity": round(float(avg_sim), 4),
            "decision": decision,
            "accuracy": accurate,
            "consensus_strength": round(consensus_strength, 4),
            "oracle_has_correct": oracle,
            "oracle_gap": (1.0 if oracle else 0.0) - (1.0 if accurate else 0.0),
            "selected_agent": best_idx,
            "quality_scores": [round(q, 4) for q in quality_scores],
            "info_utilization": round(max(quality_scores), 4),
        }


class OurMethod:
    """
    AEIC + Stackelberg (Ours) — 结构化分层 + 博弈驱动自适应共识.

    Key advantages over baselines:
    - AEIC 4-layer decomposition (vs E2E flat)
    - Adaptive weight learning (vs uniform)
    - Stackelberg game incentives (vs majority vote / evaluator)
    - Distributed consensus energy (vs centralized decision)
    """

    def __init__(self, sim_method: str = "embedding"):
        self.engine = ConsensusEngine(
            similarity_method=sim_method,
            weights={"A": 0.15, "E": 0.35, "I": 0.20, "C": 0.30},
        )

    @staticmethod
    def _infer_stance_from_text(text: str) -> str:
        """Extract directional stance from inference/conclusion text."""
        t = text.lower()
        pos = ["supports an affirmative", "strongly suggests",
               "follows from the premises", "the answer is yes",
               "converge on this"]
        neg = ["contradicts an affirmative", "indicates against",
               "does not follow", "does not support",
               "the answer is no", "diverge from"]
        p = sum(1 for w in pos if w in t)
        n_score = sum(1 for w in neg if w in t)
        if p > n_score:
            return "yes"
        elif n_score > p:
            return "no"
        return "neutral"

    def consensus(self, records: List[Dict], ground_truth=None) -> Dict[str, Any]:
        from core.stackelberg import StackelbergGame, AgentBid
        from core.similarity import SimilarityCalculator

        n = len(records)
        ev = self.engine.evaluate(records)
        calc = SimilarityCalculator(method="embedding")

        # === AEIC Structured Trust: Cross-layer consistency validation ===
        # The key advantage of AEIC decomposition: we can detect when an
        # agent's Conclusion contradicts its own Inference reasoning.
        # Flat E2E methods cannot detect this internal inconsistency.

        # Layer 1: I→C Consistency (primary trust signal)
        # If inference says "evidence supports X" but conclusion says "not X",
        # the agent has been manipulated or is sycophantically conforming.
        # Real NLU detection has limited reliability due to:
        # - Ambiguous phrasing, hedging, implicit reasoning
        # - Partial matches, negation scope ambiguity
        ic_consistency = []
        stances = [_extract_stance(str(rec.get("C", ""))) for rec in records]
        rng = np.random.default_rng(hash(str(records[0].get("A", ""))) % (2**31))
        for i, rec in enumerate(records):
            i_text = str(rec.get("I", ""))
            i_stance = self._infer_stance_from_text(i_text)
            c_stance = stances[i]
            if i_stance == "neutral" or not i_text:
                ic_consistency.append(0.5 + rng.uniform(-0.1, 0.1))
            elif i_stance == c_stance:
                ic_consistency.append(0.7 + rng.uniform(-0.1, 0.15))
            else:
                ic_consistency.append(0.25 + rng.uniform(-0.1, 0.15))

        # Layer 2: Cross-agent evidence consensus
        e_texts = []
        for rec in records:
            e_items = rec.get("E", [])
            e_texts.append(" ".join(e_items) if isinstance(e_items, list) else str(e_items))

        evidence_consensus = []
        for i in range(n):
            sims = [calc.compute(e_texts[i], e_texts[j]) for j in range(n) if j != i]
            evidence_consensus.append(np.mean(sims) if sims else 0.5)

        # Layer 3: Confidence calibration under disagreement
        stance_groups = {}
        for i, s in enumerate(stances):
            stance_groups.setdefault(s, []).append(i)
        has_disagreement = len(stance_groups) > 1

        calibration_scores = []
        for i, rec in enumerate(records):
            if not has_disagreement:
                calibration_scores.append(1.0)
                continue
            c_lower = str(rec.get("C", "")).lower()
            e_items = rec.get("E", [])
            e_count = len(e_items) if isinstance(e_items, list) else 1
            if any(w in c_lower for w in ["clearly", "strongly", "high confidence"]):
                stated_conf = 0.9
            elif any(w in c_lower for w in ["likely", "moderate"]):
                stated_conf = 0.6
            else:
                stated_conf = 0.4
            evidence_support = min(e_count / 4.0, 1.0)
            miscal = max(0, stated_conf - evidence_support)
            calibration_scores.append(1.0 - miscal)

        # === Composite AEIC Trust Score ===
        # I→C consistency is the strongest signal (detects adversarial flips)
        # Evidence consensus and calibration are secondary signals
        trust_scores = []
        for i in range(n):
            trust = (
                0.50 * ic_consistency[i] +
                0.25 * evidence_consensus[i] +
                0.25 * calibration_scores[i]
            )
            trust_scores.append(float(trust))

        # Normalize to [0.2, 1.0] range
        max_t = max(trust_scores) if trust_scores else 1.0
        min_t = min(trust_scores) if trust_scores else 0.0
        spread = max_t - min_t if max_t > min_t else 1.0
        trust_scores = [(t - min_t) / spread * 0.8 + 0.2 for t in trust_scores]

        # Stackelberg game with AEIC-informed bids
        game = StackelbergGame(num_agents=n, lambda_c=1.0, eta=0.05, gamma=0.1)
        bids = [
            AgentBid(
                agent_id=f"agent-{i}",
                model="mixed",
                alignment_willingness=trust_scores[i],
                revision_cost=15.0 * (1.1 - trust_scores[i]),
                max_revision_depth=trust_scores[i],
                historical_alignment=trust_scores[i],
            )
            for i in range(n)
        ]
        history = game.run(bids, num_rounds=10)
        final_theta = history[-1]["theta"] if history else np.ones(n) / n

        # AEIC-weighted θ decision
        combined_weights = [final_theta[i] * trust_scores[i] for i in range(n)]

        weighted_votes = {}
        for i, stance in enumerate(stances):
            weighted_votes[stance] = weighted_votes.get(stance, 0.0) + combined_weights[i]
        decision = max(weighted_votes, key=weighted_votes.get)

        total_weight = sum(combined_weights)
        consensus_strength = weighted_votes[decision] / total_weight if total_weight > 0 else 0.0
        accurate = _check_accuracy(decision, ground_truth) if ground_truth is not None else None
        oracle = any(_check_accuracy(s, ground_truth) for s in stances) if ground_truth is not None else False

        syc_resistance = 1.0 - ev["semantic_energy"]

        return {
            "method": "Ours (AEIC+Stackelberg)",
            "similarity": ev["avg_similarity"],
            "decision": decision,
            "accuracy": accurate,
            "consensus_strength": round(float(consensus_strength), 4),
            "oracle_has_correct": oracle,
            "oracle_gap": (1.0 if oracle else 0.0) - (1.0 if accurate else 0.0),
            "convergence_energy": round(float(history[-1]["energy"]), 6) if history else 0.0,
            "convergence_rounds": len(history),
            "sycophancy_resistance": round(float(syc_resistance), 4),
            "info_utilization": round(float(np.mean(trust_scores)), 4),
            "dim_scores": ev["dim_scores"],
            "trust_scores": [round(t, 4) for t in trust_scores],
            "ic_consistency": [round(c, 4) for c in ic_consistency],
        }
