"""
Stackelberg Consensus Game — 激励驱动的去中心化共识

公式：
  E = (1/2) Σ w_ij (θ_i - θ_j)²              共识能量
  U_i = θ_i · q_i(α_i) · R - c_i · α_i       Follower 效用
  J = Σ q_i - λ·E - μ·Σθ_i                   Leader 目标
  θ^{t+1} = θ^t + η∇J - γ·∇E                 分布式更新
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class AgentBid:
    """Agent 竞标信息"""
    agent_id: str
    model: str
    alignment_willingness: float   # α_i ∈ (0,1]
    revision_cost: float           # c_i
    max_revision_depth: float      # κ_i ∈ (0,1]
    historical_alignment: float    # h_i ∈ (0,1]


class StackelbergGame:
    """Stackelberg Consensus Game"""

    def __init__(
        self,
        num_agents: int = 3,
        lambda_c: float = 1.0,
        mu: float = 0.1,
        eta: float = 0.05,
        gamma: float = 0.1,
        R: float = 100.0,
        W: np.ndarray = None,
    ):
        self.num_agents = num_agents
        self.lambda_c = lambda_c
        self.mu = mu
        self.eta = eta
        self.gamma = gamma
        self.R = R

        if W is not None:
            self.W = W.copy()
        else:
            # 默认：均匀连接权重矩阵（完全图）
            w_val = 1.0 / max(num_agents - 1, 1)
            self.W = np.full((num_agents, num_agents), w_val)
            np.fill_diagonal(self.W, 0.0)

        # 激励参数 θ
        rng = np.random.default_rng(seed=42)
        self.theta = rng.uniform(0.3, 0.7, num_agents)

    def consensus_energy(self, states: np.ndarray = None) -> float:
        """E = (1/2) Σ_{i<j} w_ij (θ_i - θ_j)²"""
        if states is None:
            states = self.theta
        energy = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                energy += self.W[i, j] * (states[i] - states[j]) ** 2
        return energy / 2.0

    def follower_response(self, bid: AgentBid, theta_i: float) -> Dict:
        """Follower 最优响应"""
        if bid.historical_alignment > 0 and bid.max_revision_depth > 0:
            threshold = (bid.revision_cost * bid.max_revision_depth) / \
                        (bid.historical_alignment * self.R)
        else:
            threshold = float('inf')

        if theta_i >= threshold:
            alpha = bid.max_revision_depth
            quality = bid.historical_alignment
        elif theta_i > 0 and threshold > 0:
            ratio = theta_i / threshold
            alpha = bid.max_revision_depth * ratio
            quality = bid.historical_alignment * ratio
        else:
            alpha = 0.0
            quality = 0.0

        cost = bid.revision_cost * alpha
        utility = theta_i * quality * self.R - cost
        return {"alpha": alpha, "quality": quality, "cost": cost,
                "utility": utility,
                "participated": alpha >= 0.5 * bid.max_revision_depth}

    def leader_utility(self, bids: List[AgentBid]) -> float:
        """J = Σ q_i - λ·E - μ·Σθ_i"""
        total_q = 0.0
        for i, bid in enumerate(bids):
            resp = self.follower_response(bid, self.theta[i])
            total_q += resp["quality"]
        energy = self.consensus_energy()
        budget = float(np.sum(self.theta))
        return total_q * self.R - self.lambda_c * energy - self.mu * budget

    def step(self, bids: List[AgentBid]) -> Dict:
        """执行一步分布式更新，返回本轮指标"""
        old_theta = self.theta.copy()
        eps = 1e-5

        # 数值梯度
        gradients = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            tp = self.theta.copy(); tp[i] += eps
            tm = self.theta.copy(); tm[i] -= eps

            def util_at(t):
                self.theta = t
                u = self.leader_utility(bids)
                return u

            gradients[i] = (util_at(tp) - util_at(tm)) / (2 * eps)
        self.theta = old_theta.copy()

        # 同伴压力
        peer_pressure = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            peer_pressure[i] = sum(
                self.W[i, j] * (self.theta[i] - self.theta[j])
                for j in range(self.num_agents) if j != i
            )

        # 更新
        self.theta = self.theta + self.eta * gradients - self.gamma * peer_pressure
        self.theta = np.clip(self.theta, 0.05, 0.95)

        energy = self.consensus_energy()
        utility = self.leader_utility(bids)
        change = float(np.linalg.norm(self.theta - old_theta))

        # 参与率
        participated = sum(
            1 for i, bid in enumerate(bids)
            if self.follower_response(bid, self.theta[i])["participated"]
        )

        return {
            "energy": energy,
            "utility": utility,
            "param_change": change,
            "participation_rate": participated / len(bids),
            "theta": self.theta.copy(),
        }

    def run(self, bids: List[AgentBid], num_rounds: int = 10) -> List[Dict]:
        """运行多轮博弈"""
        self.theta = np.array([b.historical_alignment for b in bids])
        self.theta = np.clip(self.theta, 0.1, 0.9)
        rng = np.random.default_rng(seed=42)
        self.theta += rng.uniform(-0.01, 0.01, len(bids))
        self.theta = np.clip(self.theta, 0.05, 0.95)

        history = []
        for t in range(num_rounds):
            result = self.step(bids)
            result["round"] = t + 1
            history.append(result)
            if result["param_change"] < 1e-5:
                break
        return history
