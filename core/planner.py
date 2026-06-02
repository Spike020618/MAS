"""
Topology Planner — 自适应拓扑规划Agent

分析问题特征，自由生成W矩阵和角色分配。
1次LLM调用 → PlanResult(W, roles, rounds, execution_order)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.llm import LLMClient


@dataclass
class PlanResult:
    n_agents: int
    W: np.ndarray
    roles: List[str]
    rounds: int
    round_mode: str
    execution_order: List[List[int]]
    reasoning: str = ""


PLANNER_SYSTEM_TEMPLATE = '''你是一个多智能体协作系统的任务规划器。给定一个是非判断题，你需要设计最优的agent协作拓扑。

核心原则：根据问题复杂度选择合适的agent数量和拓扑结构。简单题用少量agent快速共识，复杂题才需要多agent多角度分析。

你需要输出严格的JSON格式（不要加markdown代码块）：
{{
  "reasoning": "简要说明为什么选择这种拓扑（1-2句）",
  "n_agents": 3,
  "W": [[0,0,0], ...],
  "roles": ["角色1", "角色2", ...],
  "rounds": 1,
  "round_mode": "fixed"
}}

字段说明：
- n_agents: agent数量（2-{max_n}）。根据问题复杂度自由选择：
  * 简单事实/常识 → 2-3个（快速验证即可）
  * 需要推理/多角度 → 4-5个
  * 复杂争议/多学科 → 5-{max_n}个
- W: n_agents×n_agents 邻接矩阵。W[i][j]=1 表示 agent j 的结论传递给 agent i
- roles: 长度=n_agents，每个元素描述该agent的分析角色（中文，15字以内）
- rounds: 执行轮次。1=DAG单次执行，2-3=多轮迭代
- round_mode: "fixed" 或 "converge"

拓扑设计原则：
1. W对角线为0。rounds=1时W必须是DAG（无环）
2. 角色互补，避免重复
3. 不要盲目用多agent——简单题3个独立投票比5个fan-in更准
4. 综合者/裁判角色不是必须的，只有当需要融合多方观点时才设置

典型模式（按复杂度选择）：
- 简单事实验证 → n=2-3, W全0(独立投票), rounds=1
- 明确的知识题 → n=3, 链式(0→1→2), rounds=1
- 需要多角度分析 → n=4-5, fan-in(多并行→1综合), rounds=1
- 争议性/反直觉 → n=3-4, 正反方+裁判, rounds=1
- 复杂多学科 → n=5-{max_n}, 多角度+迭代, rounds=2'''

PLANNER_USER_TEMPLATE = '''请根据问题复杂度，自由选择合适的agent数量和协作拓扑：

问题: {question}

注意：简单题用2-3个agent就够了，不要过度设计。输出JSON：'''


def topological_layers(W: np.ndarray) -> List[List[int]]:
    """将有向图按层排列（Kahn算法）。同层可并行。"""
    n = W.shape[0]
    in_degree = [int((W[i] > 0).sum()) for i in range(n)]
    remaining = set(range(n))
    layers = []

    while remaining:
        layer = sorted([i for i in remaining if in_degree[i] == 0])
        if not layer:
            layers.append(sorted(remaining))
            break
        layers.append(layer)
        for i in layer:
            remaining.remove(i)
            for j in range(n):
                if W[j, i] > 0:
                    in_degree[j] -= 1

    return layers


def compute_depth_bonus(W: np.ndarray) -> List[float]:
    """计算每个节点的depth bonus。
    sink(接收但不输出): +0.5
    middle(既接收又输出): +0.2
    root(只输出或孤立): +0.0
    """
    n = W.shape[0]
    bonus = []
    for i in range(n):
        has_out = any(W[j, i] > 0 for j in range(n) if j != i)
        has_in = any(W[i, j] > 0 for j in range(n) if j != i)
        if has_in and not has_out:
            bonus.append(0.5)
        elif has_in and has_out:
            bonus.append(0.2)
        else:
            bonus.append(0.0)
    return bonus


class TopologyPlanner:
    """自适应拓扑规划器 — 1次LLM调用生成协作拓扑。"""

    def __init__(self, llm_client: LLMClient = None):
        if llm_client is None:
            llm_client = LLMClient(backend="deepseek")
        self.llm = llm_client

    def plan(self, question: str, n_agents: int = 5) -> PlanResult:
        """分析问题并生成拓扑规划。n_agents作为上限，planner自行决定实际数量。"""
        system = PLANNER_SYSTEM_TEMPLATE.format(max_n=n_agents)
        user = PLANNER_USER_TEMPLATE.format(question=question)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        raw = self.llm.chat(messages, temperature=0.4)
        return self._parse_plan(raw, n_agents, question)

    def _parse_plan(self, raw: str, n_agents: int, question: str) -> PlanResult:
        """解析LLM输出为PlanResult，含robust fallback。"""
        text = raw.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[^{}]*"W"[^}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._fallback_plan(n_agents, question)
            else:
                return self._fallback_plan(n_agents, question)

        return self._validate_and_build(data, n_agents, question)

    def _validate_and_build(self, data: Dict, max_agents: int,
                            question: str) -> PlanResult:
        """验证并构建PlanResult，修复常见问题。"""
        try:
            # Planner自行决定n_agents（2~max_agents）
            planned_n = int(data.get("n_agents", max_agents))
            n_agents = max(2, min(planned_n, max_agents))

            W_raw = data.get("W", [])
            W = np.array(W_raw, dtype=float)

            if W.shape == (n_agents, n_agents):
                pass
            elif W.size == n_agents * n_agents:
                W = W.reshape(n_agents, n_agents)
            elif W.shape[0] == W.shape[1]:
                # Planner给了不同大小的矩阵，截断或pad
                actual_n = W.shape[0]
                if actual_n < n_agents:
                    n_agents = actual_n
                else:
                    W = W[:n_agents, :n_agents]
            else:
                return self._fallback_plan(min(3, max_agents), question)

            np.fill_diagonal(W, 0.0)
            W = (W > 0).astype(float)

            roles = data.get("roles", [])
            if len(roles) < n_agents:
                roles.extend([f"分析师{i}" for i in range(len(roles), n_agents)])
            roles = roles[:n_agents]

            rounds = int(data.get("rounds", 1))
            rounds = max(1, min(rounds, 3))

            round_mode = data.get("round_mode", "fixed")
            if round_mode not in ("fixed", "converge"):
                round_mode = "fixed"

            if rounds == 1:
                execution_order = topological_layers(W)
            else:
                execution_order = [list(range(n_agents))]

            reasoning = data.get("reasoning", "")

            return PlanResult(
                n_agents=n_agents,
                W=W,
                roles=roles,
                rounds=rounds,
                round_mode=round_mode,
                execution_order=execution_order,
                reasoning=reasoning,
            )

        except (ValueError, TypeError, KeyError):
            return self._fallback_plan(min(3, max_agents), question)

    def _fallback_plan(self, n_agents: int, question: str) -> PlanResult:
        """Fallback: 独立并行投票（最稳健的默认策略）。"""
        n_agents = max(2, n_agents)
        W = np.zeros((n_agents, n_agents))

        roles = [f"独立验证者{i+1}" for i in range(n_agents)]

        return PlanResult(
            n_agents=n_agents,
            W=W,
            roles=roles,
            rounds=1,
            round_mode="fixed",
            execution_order=topological_layers(W),
            reasoning="fallback: independent parallel voting",
        )


def random_plan(n_agents: int, p: float = 0.4, seed: int = None) -> PlanResult:
    """生成随机拓扑（用于Random-W baseline对比）。"""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_agents, n_agents))

    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and rng.random() < p:
                W[i, j] = 1.0

    for i in range(n_agents):
        if W[i].sum() == 0 and W[:, i].sum() == 0:
            j = (i + 1) % n_agents
            W[j, i] = 1.0

    roles = [f"分析师{i}" for i in range(n_agents)]

    return PlanResult(
        n_agents=n_agents,
        W=W,
        roles=roles,
        rounds=1,
        round_mode="fixed",
        execution_order=topological_layers(W),
        reasoning="random topology baseline",
    )
