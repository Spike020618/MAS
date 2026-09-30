"""
TrustTracker — 跨问题信任积累

滑动窗口记录每个agent的历史质量信号（θ, IC, EC, EQ, consensus_agreement），
计算可靠性分数，用于动态agent选择。

所有信号均为无监督（无需ground truth）。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class TrustTracker:

    def __init__(self, n_agents: int, window: int = 20, warmup: int = 3,
                 decay: float = 0.9):
        self.n_agents = n_agents
        self.window = window
        self.warmup = warmup
        self.decay = decay
        self.history: Dict[int, List[Dict]] = {i: [] for i in range(n_agents)}
        self.question_count = 0

    def update(self, agent_id: int, theta: float, ic: float, ec: float,
               eq: float, agreed_with_consensus: bool):
        entry = {
            "theta": theta, "ic": ic, "ec": ec, "eq": eq,
            "agreed": float(agreed_with_consensus),
        }
        h = self.history[agent_id]
        h.append(entry)
        if len(h) > self.window:
            self.history[agent_id] = h[-self.window:]

    def increment_question(self):
        self.question_count += 1

    def reliability_score(self, agent_id: int) -> float:
        h = self.history[agent_id]
        if not h:
            return 0.5
        n = len(h)
        weights = [self.decay ** (n - 1 - i) for i in range(n)]
        w_sum = sum(weights)

        avg_theta = sum(w * e["theta"] for w, e in zip(weights, h)) / w_sum
        avg_ic = sum(w * e["ic"] for w, e in zip(weights, h)) / w_sum
        avg_ec = sum(w * e["ec"] for w, e in zip(weights, h)) / w_sum
        avg_eq = sum(w * e["eq"] for w, e in zip(weights, h)) / w_sum
        avg_agreed = sum(w * e["agreed"] for w, e in zip(weights, h)) / w_sum

        structural = avg_ic * avg_ec * avg_eq
        return 0.4 * structural + 0.3 * avg_theta + 0.3 * avg_agreed

    @property
    def warmed_up(self) -> bool:
        return self.question_count >= self.warmup

    def select_agents(self, max_n: int, min_n: int = 2,
                      threshold: float = 0.3) -> List[int]:
        if not self.warmed_up:
            return list(range(min(max_n, self.n_agents)))

        scores = [(self.reliability_score(i), i) for i in range(self.n_agents)]
        scores.sort(reverse=True)

        qualified = [(s, i) for s, i in scores if s > threshold]
        n = max(min_n, min(len(qualified), max_n))
        return [i for _, i in scores[:n]]

    def get_all_scores(self) -> Dict[int, float]:
        return {i: round(self.reliability_score(i), 4)
                for i in range(self.n_agents)}

    def __repr__(self):
        return (f"TrustTracker(n_agents={self.n_agents}, "
                f"questions={self.question_count}, "
                f"warmed_up={self.warmed_up})")
