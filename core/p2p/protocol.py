"""
Consensus Protocol — 4种P2P编排协议

通过HTTP调用各Agent节点，实现:
  - parallel:   广播question → 收集AEIC → trust决策
  - sequential: 链式 C→A 传递
  - star:       leaf并行 → hub综合
  - debate:     多轮广播 → 收敛检测
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import requests

from core.similarity import SimilarityCalculator
from core.topology import make_topology
from core.trust import evaluate_and_decide, majority_vote, quality_weighted_vote, extract_stance


SEQUENTIAL_CONTEXT = """前一位分析师的结论是：'{prev_conclusion}'

鉴于这一先前评估，请进行你自己的独立AEIC推理。
你的A(Assumption)应批判性地评估前人结论作为前提，
你的E(Evidence)应补充新的证据（不要简单重复前人已知信息），
你的I(Inference)应推进或修正前人的推理链，
你的C(Conclusion)给出你的最终判断（可以同意或反对前人）。"""

STAR_HUB_CONTEXT = """多位专家已独立分析了以下问题。他们的结论构成你的证据基础。

专家结论（按可信度排序）：
{leaf_conclusions}

请将这些结论作为你的Evidence(E)，
在Assumption(A)中明确综合问题的前提，
在Inference(I)中权衡各方观点（注意各专家可信度不同），
给出你的最终Conclusion(C)。"""

DEBATE_CONTEXT = """这是第{round_num}轮讨论。上一轮各方的结论如下：
{other_conclusions}
你上一轮的结论是: {my_prev_conclusion}

请在考虑这些不同观点后，重新进行AEIC推理。
你可以维持或修正你的立场，但需在Inference中给出理由。"""


def _call_aeic(agent_url: str, question: str, timeout: int = 60) -> Dict:
    """向agent发送AEIC推理请求。"""
    resp = requests.post(
        f"{agent_url}/aeic",
        json={"question": question},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["aeic"]


def _call_aeic_ctx(agent_url: str, question: str, context: str, timeout: int = 60) -> Dict:
    """向agent发送带context的AEIC推理请求。"""
    resp = requests.post(
        f"{agent_url}/aeic_ctx",
        json={"question": question, "context": context},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["aeic"]


class ConsensusProtocol:
    """P2P共识协议 — 通过HTTP协调各agent完成4种模式的推理和决策。"""

    def __init__(self, agents: List[Dict], sim_calc: SimilarityCalculator):
        """
        Args:
            agents: [{agent_id, url, backend, persona}, ...] from registry
            sim_calc: SimilarityCalculator instance for trust evaluation
        """
        self.agents = agents
        self.sim_calc = sim_calc
        self.n = len(agents)

    def run(self, question: str, mode: str, use_stackelberg: bool = True,
            topology: str = None, **kwargs) -> Dict:
        topo_map = {"parallel": "complete", "sequential": "path",
                    "star": "star", "debate": "complete"}
        topo_name = topology or topo_map.get(mode, "complete")
        n = self.n if mode != "star" else self.n + 1
        W = make_topology(topo_name, self.n if mode != "star" else self.n)

        runners = {
            "parallel": self._parallel,
            "sequential": self._sequential,
            "star": self._star,
            "debate": self._debate,
        }
        return runners[mode](question, W, use_stackelberg, **kwargs)

    def _parallel(self, question: str, W: np.ndarray,
                  use_stackelberg: bool, **kwargs) -> Dict:
        """P1: 广播question → 各agent并行推理 → trust决策"""
        nodes = [None] * self.n
        urls = [a["url"] for a in self.agents]

        def _call(i):
            return i, _call_aeic(urls[i], question)

        with ThreadPoolExecutor(max_workers=self.n) as pool:
            futures = [pool.submit(_call, i) for i in range(self.n)]
            for f in as_completed(futures):
                idx, aeic = f.result()
                nodes[idx] = aeic

        result = evaluate_and_decide(nodes, W, self.sim_calc,
                                     mode="parallel", use_stackelberg=use_stackelberg)
        result["nodes"] = nodes
        result["mode"] = "parallel"
        return result

    def _sequential(self, question: str, W: np.ndarray,
                    use_stackelberg: bool, **kwargs) -> Dict:
        """P2: Agent-0推理 → 发C给Agent-1 → Agent-1推理 → ... """
        nodes = []
        urls = [a["url"] for a in self.agents]

        for i in range(self.n):
            if i == 0:
                aeic = _call_aeic(urls[i], question)
            else:
                prev_c = nodes[i - 1].get("C", "")
                context = SEQUENTIAL_CONTEXT.format(prev_conclusion=prev_c)
                aeic = _call_aeic_ctx(urls[i], question, context)
            nodes.append(aeic)

        result = evaluate_and_decide(nodes, W, self.sim_calc,
                                     mode="sequential", use_stackelberg=use_stackelberg)
        result["nodes"] = nodes
        result["mode"] = "sequential"
        return result

    def _star(self, question: str, W: np.ndarray,
              use_stackelberg: bool, **kwargs) -> Dict:
        """P3: Leaf并行推理 → 计算trust → Hub收到各C+trust → 综合推理"""
        urls = [a["url"] for a in self.agents]
        hub_idx = 0
        leaf_indices = list(range(1, self.n))

        # Phase 1: Leaf agents parallel
        leaf_nodes = [None] * len(leaf_indices)

        def _call_leaf(li, agent_idx):
            return li, _call_aeic(urls[agent_idx], question)

        with ThreadPoolExecutor(max_workers=len(leaf_indices)) as pool:
            futures = [pool.submit(_call_leaf, li, idx)
                       for li, idx in enumerate(leaf_indices)]
            for f in as_completed(futures):
                li, aeic = f.result()
                leaf_nodes[li] = aeic

        # Compute leaf trust
        leaf_W = W[1:, 1:] if W.shape[0] > len(leaf_indices) else np.ones((len(leaf_indices), len(leaf_indices)))
        leaf_result = evaluate_and_decide(leaf_nodes, leaf_W, self.sim_calc,
                                          mode="parallel", use_stackelberg=False)
        leaf_trust = leaf_result["trust"]

        # Phase 2: Hub receives leaf conclusions + trust
        sorted_leaves = sorted(range(len(leaf_indices)),
                               key=lambda j: leaf_trust[j], reverse=True)
        lines = []
        for j in sorted_leaves:
            c = leaf_nodes[j].get("C", "")
            t = leaf_trust[j]
            lines.append(f"- 专家{j+1} (可信度{t:.0%}): {c}")
        context = STAR_HUB_CONTEXT.format(leaf_conclusions="\n".join(lines))
        hub_aeic = _call_aeic_ctx(urls[hub_idx], question, context)

        # Combine: hub + leaves
        nodes = [hub_aeic] + leaf_nodes
        result = evaluate_and_decide(nodes, W, self.sim_calc,
                                     mode="parallel", use_stackelberg=use_stackelberg)
        result["nodes"] = nodes
        result["mode"] = "star"
        result["hub_index"] = 0
        return result

    def _debate(self, question: str, W: np.ndarray,
                use_stackelberg: bool, max_rounds: int = 3, **kwargs) -> Dict:
        """P4: 多轮辩论 — 拓扑约束可见性，每轮只能看到W邻居的结论"""
        max_rounds = kwargs.get("max_rounds", max_rounds)
        urls = [a["url"] for a in self.agents]
        all_rounds = []

        # Round 1: parallel (独立推理，无上游信息)
        nodes = [None] * self.n

        def _call_r1(i):
            return i, _call_aeic(urls[i], question)

        with ThreadPoolExecutor(max_workers=self.n) as pool:
            futures = [pool.submit(_call_r1, i) for i in range(self.n)]
            for f in as_completed(futures):
                idx, aeic = f.result()
                nodes[idx] = aeic
        all_rounds.append(nodes)

        # Rounds 2+: 拓扑约束 — agent i 只能看到 W[i,j]>0 的邻居结论
        for t in range(2, max_rounds + 1):
            prev_nodes = all_rounds[-1]
            new_nodes = [None] * self.n

            def _call_debate(i, _prev=prev_nodes, _t=t, _W=W):
                other_lines = []
                for j in range(self.n):
                    if j != i and _W[i, j] > 0:
                        c = _prev[j].get("C", "")
                        other_lines.append(f"- 分析师{j+1}: {c}")
                if not other_lines:
                    other_lines.append("（无可见邻居结论）")
                context = DEBATE_CONTEXT.format(
                    round_num=_t,
                    other_conclusions="\n".join(other_lines),
                    my_prev_conclusion=_prev[i].get("C", ""),
                )
                return i, _call_aeic_ctx(urls[i], question, context)

            with ThreadPoolExecutor(max_workers=self.n) as pool:
                futures = [pool.submit(_call_debate, i) for i in range(self.n)]
                for f in as_completed(futures):
                    idx, aeic = f.result()
                    new_nodes[idx] = aeic

            all_rounds.append(new_nodes)

            # Convergence check
            prev_stances = [extract_stance(nd.get("C", "")) for nd in prev_nodes]
            cur_stances = [extract_stance(nd.get("C", "")) for nd in new_nodes]
            if prev_stances == cur_stances:
                break

        final_nodes = all_rounds[-1]
        result = evaluate_and_decide(final_nodes, W, self.sim_calc,
                                     mode="debate", use_stackelberg=use_stackelberg)
        result["nodes"] = final_nodes
        result["mode"] = "debate"
        result["num_rounds"] = len(all_rounds)
        return result
