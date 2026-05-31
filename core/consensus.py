"""
AEIC 共识引擎 — 多节点语义共识评估与自适应权重学习

公式：
  SIM_AEIC(R_i, R_j) = w_A·sim_A + w_E·sim_E + w_I·sim_I + w_C·sim_C
  U = avg_sim · R - C
  权重更新: v_X ← v_X + η · (U/R) · w_X, 然后 w = softmax(v)
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

import numpy as np

from .similarity import SimilarityCalculator, tokenize


class ConsensusEngine:
    """多节点 AEIC 语义共识引擎"""

    def __init__(
        self,
        similarity_method: str = "bm25",
        weights: Dict[str, float] = None,
        reward: float = 100.0,
        cost: float = 25.0,
    ):
        self.sim_method = similarity_method
        self.R = reward
        self.C = cost
        self._calc = SimilarityCalculator(method=similarity_method)

        # AEIC 权重（logits 参数化）
        if weights:
            self.w_logits = {k: np.log(max(v, 0.01)) for k, v in weights.items()}
        else:
            self.w_logits = {"A": np.log(0.2), "E": np.log(0.3),
                            "I": np.log(0.2), "C": np.log(0.3)}
        self.w = self._softmax_weights()
        self.w_history: List[Dict[str, float]] = []

    def _softmax_weights(self) -> Dict[str, float]:
        logits = np.array([self.w_logits["A"], self.w_logits["E"],
                           self.w_logits["I"], self.w_logits["C"]])
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        w = exp / np.sum(exp)
        return {"A": float(w[0]), "E": float(w[1]), "I": float(w[2]), "C": float(w[3])}

    def update_weights(self, utility: float, lr: float = 0.01,
                       dim_scores: Dict[str, float] = None) -> Dict[str, float]:
        """基于维度贡献的自适应权重更新

        dim_scores: 各维度平均相似度 {"A": ..., "E": ..., "I": ..., "C": ...}
        """
        self.w_history.append(dict(self.w))
        signal = max(utility / self.R, 0.01)
        if dim_scores:
            # 差异化更新：高相似度维度获得更大权重提升
            total_dim = sum(dim_scores.values()) or 1.0
            for k in self.w_logits:
                contribution = dim_scores.get(k, 0.25) / total_dim
                self.w_logits[k] += lr * signal * contribution
        else:
            for k in self.w_logits:
                self.w_logits[k] += lr * signal * self.w[k]
        self.w = self._softmax_weights()
        return dict(self.w)

    # ── 相似度计算 ───────────────────────────────────

    def _sim(self, t1: str, t2: str) -> float:
        return self._calc.compute(t1, t2, self.sim_method)

    def _sim_evidence(self, ev_i, ev_j) -> float:
        if isinstance(ev_i, list):
            text_i = " ".join(str(e) for e in ev_i)
            set_i = set(str(e) for e in ev_i)
        else:
            text_i = str(ev_i)
            set_i = set(tokenize(str(ev_i)))
        if isinstance(ev_j, list):
            text_j = " ".join(str(e) for e in ev_j)
            set_j = set(str(e) for e in ev_j)
        else:
            text_j = str(ev_j)
            set_j = set(tokenize(str(ev_j)))
        sem = self._sim(text_i, text_j)
        ovl = len(set_i & set_j) / len(set_i | set_j) if (set_i | set_j) else 0.0
        return 0.6 * sem + 0.4 * ovl

    def aeic_similarity(self, rec_i: Dict, rec_j: Dict) -> Dict[str, float]:
        """计算两节点 AEIC 各层 + 加权总相似度"""
        sim_a = self._sim(str(rec_i.get("A", "")), str(rec_j.get("A", "")))
        sim_e = self._sim_evidence(rec_i.get("E", ""), rec_j.get("E", ""))
        sim_i = self._sim(str(rec_i.get("I", "")), str(rec_j.get("I", "")))
        sim_c = self._sim(str(rec_i.get("C", "")), str(rec_j.get("C", "")))
        total = (self.w["A"] * sim_a + self.w["E"] * sim_e +
                 self.w["I"] * sim_i + self.w["C"] * sim_c)
        return {"sim_a": sim_a, "sim_e": sim_e, "sim_i": sim_i, "sim_c": sim_c,
                "total": total}

    def end_to_end_similarity(self, rec_i: Dict, rec_j: Dict) -> float:
        """端到端：直接拼接所有层计算整体相似度"""
        def concat(rec):
            parts = [str(rec.get("A", "")),
                     " ".join(rec.get("E", [])) if isinstance(rec.get("E"), list) else str(rec.get("E", "")),
                     str(rec.get("I", "")),
                     str(rec.get("C", ""))]
            return " ".join(parts)
        return self._sim(concat(rec_i), concat(rec_j))

    # ── 主接口 ───────────────────────────────────────

    def evaluate(self, records: List[Dict]) -> Dict[str, Any]:
        """
        多节点共识评估

        Args:
            records: N 个节点的 AEIC 记录 [{"A": ..., "E": [...], "I": ..., "C": ...}]

        Returns:
            avg_similarity, utility, decision, pairwise details, conflict_attribution
        """
        n = len(records)
        if n < 2:
            raise ValueError("至少需要2个节点")

        pairwise = {}
        conflicts = {}
        all_sims = []

        for i, j in combinations(range(n), 2):
            result = self.aeic_similarity(records[i], records[j])
            key = f"{i}x{j}"
            pairwise[key] = result
            all_sims.append(result["total"])

            # 冲突归因
            layers = {"A": result["sim_a"], "E": result["sim_e"],
                      "I": result["sim_i"], "C": result["sim_c"]}
            conflicts[key] = min(layers, key=layers.get)

        avg_sim = sum(all_sims) / len(all_sims)
        utility = avg_sim * self.R - self.C

        # 各维度平均相似度
        n_pairs = len(pairwise)
        dim_scores = {
            "A": sum(p["sim_a"] for p in pairwise.values()) / max(n_pairs, 1),
            "E": sum(p["sim_e"] for p in pairwise.values()) / max(n_pairs, 1),
            "I": sum(p["sim_i"] for p in pairwise.values()) / max(n_pairs, 1),
            "C": sum(p["sim_c"] for p in pairwise.values()) / max(n_pairs, 1),
        }

        return {
            "avg_similarity": round(avg_sim, 4),
            "utility": round(utility, 2),
            "decision": "ESS" if utility > 55 else ("Audit" if utility > 0 else "Reject"),
            "pairwise": pairwise,
            "conflict_attribution": conflicts,
            "semantic_energy": sum((1 - s) ** 2 for s in all_sims) / (2 * max(n - 1, 1)),
            "dim_scores": dim_scores,
        }
