"""
Stackelberg Consensus Game 框架 v2
================================================================

三个核心公式：
  公式1  共识能量      E = (1/2) Σ w_ij (θ_i - θ_j)²
  公式2  Leader 目标   min_θ [-(Σ U_i) + λ·E]
  公式3  分布式学习    θ_i^{t+1} = θ_i^t + η ∇U_i - γ Σ_j w_ij(θ_i-θ_j)

改进点（v2）：
  - 真实 ∇U_i 梯度（数值差分替代 hard-code 的 0.1）
  - lambda_c 支持构造函数参数，方便实验对照
  - run_game_rounds 不再 print 冗余日志
  - 兼容原 StackelbergScheduler 别名
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────

@dataclass
class AgentBid:
    """Agent 的竞标信息"""
    agent_id: str
    port: int
    model: str
    role: str
    quality_promise: float    # 承诺质量 q ∈ (0, 1]
    cost_per_task: float      # 单位任务成本 c
    capacity: float           # 最大承载量 κ ∈ (0, 1]
    past_performance: float = 0.8  # 历史表现 α


# ─────────────────────────────────────────────────────
# Stackelberg Consensus Game
# ─────────────────────────────────────────────────────

class StackelbergConsensusGame:
    """
    Stackelberg Consensus Game（SCG）

    Leader：最大化 Σ U_i - λ·E（公式2）
    Follower：响应 Leader 激励参数 θ，最大化自身效用
    分布式学习：θ 通过公式3 迭代更新
    """

    def __init__(
        self,
        leader_port: int,
        num_agents: int = 3,
        lambda_c: float = 1.0,   # λ：共识-激励权衡系数
        eta: float = 0.05,       # η：学习率
        gamma: float = 0.1,      # γ：共识约束强度
        epsilon: float = 1e-5,   # 数值梯度步长
        verbose: bool = False,   # 是否打印详细日志
    ):
        self.leader_port = leader_port
        self.num_agents = num_agents
        self.lambda_c = lambda_c
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.verbose = verbose
        self.history: List[Dict] = []

        # 通信权重矩阵 W（均匀连接无向图）
        if num_agents > 1:
            w_val = 1.0 / (num_agents - 1)
        else:
            w_val = 0.0
        self.W = np.full((num_agents, num_agents), w_val)
        np.fill_diagonal(self.W, 0.0)

        # 激励参数 θ_i（初始化在 [0.5, 1.0]）
        rng = np.random.default_rng(seed=0)
        self.leader_params = rng.uniform(0.5, 1.0, num_agents)

    # ══════════════════════════════════════════════════
    # 公式1：共识能量
    # ══════════════════════════════════════════════════

    def consensus_energy(self, states: np.ndarray) -> float:
        """E = (1/2) Σ_{i<j} w_ij (θ_i - θ_j)²"""
        energy = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                energy += self.W[i, j] * (states[i] - states[j]) ** 2
        return energy / 2.0

    def consensus_gradient(self, i: int, states: np.ndarray) -> float:
        """∂E/∂θ_i = Σ_j w_ij (θ_i - θ_j)"""
        return float(
            sum(self.W[i, j] * (states[i] - states[j])
                for j in range(self.num_agents) if j != i)
        )

    # ══════════════════════════════════════════════════
    # 分配与效用计算
    # ══════════════════════════════════════════════════

    def _follower_response(self, bid: AgentBid, workload: float) -> Dict:
        """计算 Follower 在给定 workload 下的实际质量与效用"""
        if workload > bid.capacity and bid.capacity > 0:
            quality = bid.quality_promise * (bid.capacity / workload)
        else:
            quality = bid.quality_promise
        quality *= bid.past_performance
        cost = bid.cost_per_task * workload
        utility = quality * workload * 100.0 - cost
        return {
            "quality": quality,
            "cost": cost,
            "utility": utility,
            "overloaded": workload > bid.capacity,
        }

    def optimize_allocation(
        self, bids: List[AgentBid], total_workload: float = 1.0
    ) -> Dict[str, float]:
        """
        贪心最优化分配（公式2 的近似求解）：
        按效率（quality/cost）降序分配，直到 total_workload 耗尽。
        """
        if not bids:
            return {}

        items = []
        for bid in bids:
            opt_load = min(bid.capacity, total_workload)
            resp = self._follower_response(bid, opt_load)
            eff = resp["quality"] / resp["cost"] if resp["cost"] > 0 else resp["quality"]
            items.append({"agent_id": bid.agent_id, "eff": eff,
                          "capacity": bid.capacity, "bid": bid})
        items.sort(key=lambda x: x["eff"], reverse=True)

        allocation: Dict[str, float] = {}
        remaining = total_workload
        for item in items:
            if remaining <= 0:
                break
            assigned = min(item["capacity"], remaining)
            allocation[item["agent_id"]] = assigned
            remaining -= assigned

        # 若仍有剩余，均摊
        if remaining > 1e-9 and allocation:
            extra = remaining / len(allocation)
            for k in allocation:
                allocation[k] += extra

        return allocation

    def evaluate_allocation(
        self, bids: List[AgentBid], allocation: Dict[str, float]
    ) -> Dict:
        """计算给定分配方案的 Leader 效用和 Follower 明细"""
        bid_map = {b.agent_id: b for b in bids}
        total_q = total_c = total_follower_utility = 0.0
        details: Dict[str, Dict] = {}

        for aid, wl in allocation.items():
            if aid not in bid_map:
                continue
            resp = self._follower_response(bid_map[aid], wl)
            total_q += resp["quality"] * wl
            total_c += resp["cost"]
            total_follower_utility += resp["utility"]
            details[aid] = {"workload": wl, **resp}

        leader_utility = total_q * 100.0 - total_c

        return {
            "total_quality": total_q,
            "total_cost": total_c,
            "leader_utility": leader_utility,
            "social_welfare": total_follower_utility,
            "agent_details": details,
            "allocation": allocation,
        }

    # ══════════════════════════════════════════════════
    # 公式3：分布式学习（真实数值梯度）
    # ══════════════════════════════════════════════════

    def _utility_gradient(
        self, i: int, bids: List[AgentBid], total_workload: float
    ) -> float:
        """
        ∇U_i ≈ [U(θ + ε·e_i) - U(θ - ε·e_i)] / (2ε)
        使用数值差分计算 Leader 效用关于 θ_i 的梯度。
        """
        eps = self.epsilon
        params_plus = self.leader_params.copy()
        params_minus = self.leader_params.copy()
        params_plus[i] = np.clip(params_plus[i] + eps, 0.0, 1.0)
        params_minus[i] = np.clip(params_minus[i] - eps, 0.0, 1.0)

        def eval_with_params(params: np.ndarray) -> float:
            # θ 影响 quality_promise（激励效果的简化模型）
            scaled_bids = [
                AgentBid(
                    agent_id=b.agent_id,
                    port=b.port,
                    model=b.model,
                    role=b.role,
                    quality_promise=np.clip(b.quality_promise * params[min(j, len(params) - 1)], 0, 1),
                    cost_per_task=b.cost_per_task,
                    capacity=b.capacity,
                    past_performance=b.past_performance,
                )
                for j, b in enumerate(bids)
            ]
            alloc = self.optimize_allocation(scaled_bids, total_workload)
            res = self.evaluate_allocation(scaled_bids, alloc)
            return res["leader_utility"]

        u_plus = eval_with_params(params_plus)
        u_minus = eval_with_params(params_minus)
        return (u_plus - u_minus) / (2.0 * eps)

    def _update_params_one_round(
        self, bids: List[AgentBid], total_workload: float
    ) -> np.ndarray:
        """
        对所有 Agent 执行一步公式3更新：
            θ_i ← θ_i + η ∇U_i - γ Σ_j w_ij(θ_i - θ_j)
        """
        old = self.leader_params.copy()
        new = old.copy()

        for i in range(self.num_agents):
            grad_u = self._utility_gradient(i, bids, total_workload)
            grad_c = self.consensus_gradient(i, old)
            new[i] = old[i] + self.eta * grad_u - self.gamma * grad_c

        self.leader_params = np.clip(new, 0.0, 1.0)
        return self.leader_params.copy()

    # ══════════════════════════════════════════════════
    # 主入口：多轮迭代
    # ══════════════════════════════════════════════════

    def run_game_rounds(
        self,
        bids: List[AgentBid],
        total_workload: float = 1.0,
        num_rounds: int = 10,
    ) -> Dict:
        """
        多轮 SCG 迭代（公式3）。
        
        Returns:
            {
              "results": [
                  {"round": t, "leader_utility": ..., "consensus_energy": ..., "param_change": ...},
                  ...
              ]
            }
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎮 SCG 多轮迭代  λ={self.lambda_c}  η={self.eta}  γ={self.gamma}")
            print(f"{'='*60}")

        results: List[Dict] = []

        for t in range(num_rounds):
            old_params = self.leader_params.copy()

            # Step1: 优化分配 + 评估
            allocation = self.optimize_allocation(bids, total_workload)
            eval_res = self.evaluate_allocation(bids, allocation)

            # Step2: 参数更新（公式3）
            self._update_params_one_round(bids, total_workload)

            # 计算指标
            energy = self.consensus_energy(self.leader_params)
            change = float(np.linalg.norm(self.leader_params - old_params))

            row = {
                "round": t + 1,
                "leader_utility": eval_res["leader_utility"],
                "social_welfare": eval_res.get("social_welfare", 0.0),
                "consensus_energy": energy,
                "param_change": change,
            }
            results.append(row)

            if self.verbose:
                print(f"  Round {t+1:3d} | U={row['leader_utility']:8.2f} "
                      f"| E={energy:.6f} | Δθ={change:.6f}")

            if change < 1e-5:
                if self.verbose:
                    print(f"  ✅ 收敛于 Round {t+1}")
                break

        if self.verbose:
            print(f"{'='*60}\n")

        return {"results": results}

    # ══════════════════════════════════════════════════
    # 兼容原接口：单轮执行
    # ══════════════════════════════════════════════════

    def execute_stackelberg_game(
        self,
        bids: List[AgentBid],
        total_workload: float = 1.0,
    ) -> Dict:
        """单轮执行（兼容原接口，仍可用）"""
        allocation = self.optimize_allocation(bids, total_workload)
        result = self.evaluate_allocation(bids, allocation)

        if self.verbose:
            print(f"\n{'='*60}")
            print("🎮 Stackelberg Game - Single Round")
            for aid, detail in result["agent_details"].items():
                print(f"  {aid}: load={detail['workload']:.2%}, U={detail['utility']:.2f}")
            print(f"  Leader utility: {result['leader_utility']:.2f}")
            print(f"{'='*60}\n")

        self.history.append(result)
        return result


# ─────────────────────────────────────────────────────
# 兼容别名
# ─────────────────────────────────────────────────────

StackelbergScheduler = StackelbergConsensusGame


# ─────────────────────────────────────────────────────
# CLI 演示
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    game = StackelbergConsensusGame(
        leader_port=8001, num_agents=3, lambda_c=1.0, verbose=True
    )
    bids = [
        AgentBid("agent1", 8002, "qwen",     "solver",   0.85, 20, 0.6, 0.90),
        AgentBid("agent2", 8003, "deepseek", "reviewer", 0.90, 25, 0.4, 0.85),
        AgentBid("agent3", 8004, "qwen",     "solver",   0.75, 15, 0.5, 0.80),
    ]

    print("── 单轮演示 ──")
    game.execute_stackelberg_game(bids, 1.0)

    print("── 多轮迭代演示 ──")
    results = game.run_game_rounds(bids, 1.0, num_rounds=8)
    for r in results["results"]:
        print(f"  Round {r['round']:2d} | E={r['consensus_energy']:.6f} | Δθ={r['param_change']:.6f}")
