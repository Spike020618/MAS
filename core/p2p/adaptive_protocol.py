"""
Adaptive Protocol (P2P) — 通过HTTP调用agent节点执行自适应拓扑

与 AdaptiveExecutor 逻辑相同，但通过HTTP调用独立agent服务。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import requests

from core.planner import PlanResult, compute_depth_bonus
from core.similarity import SimilarityCalculator
from core.stackelberg import AgentBid, StackelbergGame
from core.trust import (
    compute_ic_consistency,
    compute_ec_relevance,
    extract_stance,
)


ADAPTIVE_ROLE_CONTEXT = """你的角色: {role}

{upstream_section}
请基于你的角色定位，对以下问题进行AEIC推理：

问题: {question}"""

UPSTREAM_SECTION = """以下是其他分析师已完成的结论供你参考：
{conclusions}

请批判性地评估这些信息，结合你自己的独立分析给出判断。"""

ITERATIVE_CONTEXT = """你的角色: {role}

这是第{round_num}轮讨论。上一轮你能看到的分析师结论如下：
{visible_conclusions}
你上一轮的结论是: {my_prev}

请在参考这些观点后，重新进行AEIC推理。你可以维持或修正立场。

问题: {question}"""


def _call_aeic(agent_url: str, question: str, timeout: int = 60) -> Dict:
    resp = requests.post(
        f"{agent_url}/aeic",
        json={"question": question},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["aeic"]


def _call_aeic_ctx(agent_url: str, question: str, context: str,
                   timeout: int = 60) -> Dict:
    resp = requests.post(
        f"{agent_url}/aeic_ctx",
        json={"question": question, "context": context},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["aeic"]


class AdaptiveProtocol:
    """P2P自适应拓扑协议 — 通过HTTP调用各agent节点。"""

    def __init__(self, agents: List[Dict], sim_calc: SimilarityCalculator):
        self.agents = agents
        self.sim_calc = sim_calc
        self.n = len(agents)

    def run(self, question: str, plan: PlanResult,
            use_stackelberg: bool = True) -> Dict:
        """执行自适应协议。"""
        if plan.n_agents > self.n:
            plan.n_agents = self.n
            plan.W = plan.W[:self.n, :self.n]
            plan.roles = plan.roles[:self.n]

        if plan.rounds == 1:
            nodes = self._execute_dag(question, plan)
        else:
            nodes = self._execute_iterative(question, plan)

        result = self._consensus_decision(nodes, plan, use_stackelberg)
        result["nodes"] = nodes
        result["plan_reasoning"] = plan.reasoning
        result["mode"] = "adaptive"
        return result

    def _execute_dag(self, question: str, plan: PlanResult) -> List[Dict]:
        n = min(plan.n_agents, self.n)
        nodes = [None] * n
        W = plan.W
        urls = [a["url"] for a in self.agents[:n]]

        for layer in plan.execution_order:
            layer = [i for i in layer if i < n]
            if not layer:
                continue

            if len(layer) == 1:
                i = layer[0]
                ctx = self._build_context(i, question, plan, nodes)
                if ctx:
                    nodes[i] = _call_aeic_ctx(urls[i], question, ctx)
                else:
                    nodes[i] = _call_aeic(urls[i], question)
            else:
                def _call(idx, _nodes=nodes):
                    ctx = self._build_context(idx, question, plan, _nodes)
                    if ctx:
                        return idx, _call_aeic_ctx(urls[idx], question, ctx)
                    else:
                        return idx, _call_aeic(urls[idx], question)

                with ThreadPoolExecutor(max_workers=len(layer)) as pool:
                    futures = [pool.submit(_call, i) for i in layer]
                    for f in as_completed(futures):
                        idx, aeic = f.result()
                        nodes[idx] = aeic

        return nodes

    def _execute_iterative(self, question: str,
                           plan: PlanResult) -> List[Dict]:
        n = min(plan.n_agents, self.n)
        W = plan.W
        urls = [a["url"] for a in self.agents[:n]]
        all_rounds = []

        nodes = [None] * n

        def _r1(i):
            role = plan.roles[i] if i < len(plan.roles) else f"分析师{i}"
            ctx = ADAPTIVE_ROLE_CONTEXT.format(
                role=role, upstream_section="", question=question)
            return i, _call_aeic_ctx(urls[i], question, ctx)

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_r1, i) for i in range(n)]
            for f in as_completed(futures):
                idx, aeic = f.result()
                nodes[idx] = aeic
        all_rounds.append(list(nodes))

        for t in range(2, plan.rounds + 1):
            prev_nodes = all_rounds[-1]
            new_nodes = [None] * n

            def _iter(i, _prev=prev_nodes, _t=t, _W=W):
                role = plan.roles[i] if i < len(plan.roles) else f"分析师{i}"
                visible = []
                for j in range(n):
                    if j != i and _W[i, j] > 0 and _prev[j]:
                        j_role = plan.roles[j] if j < len(plan.roles) else ""
                        visible.append(
                            f"- 分析师{j}({j_role}): {_prev[j].get('C', '')}")
                vis_text = "\n".join(visible) if visible else "（无可见邻居）"
                my_prev = _prev[i].get("C", "") if _prev[i] else ""
                ctx = ITERATIVE_CONTEXT.format(
                    role=role, round_num=_t,
                    visible_conclusions=vis_text,
                    my_prev=my_prev, question=question)
                return i, _call_aeic_ctx(urls[i], question, ctx)

            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(_iter, i) for i in range(n)]
                for f in as_completed(futures):
                    idx, aeic = f.result()
                    new_nodes[idx] = aeic

            all_rounds.append(new_nodes)

            if plan.round_mode == "converge":
                prev_stances = [extract_stance(nd.get("C", ""))
                                for nd in prev_nodes if nd]
                cur_stances = [extract_stance(nd.get("C", ""))
                               for nd in new_nodes if nd]
                if prev_stances == cur_stances:
                    break

        return all_rounds[-1]

    def _build_context(self, agent_idx: int, question: str,
                       plan: PlanResult, nodes: List[Dict]) -> str:
        W = plan.W
        n = min(plan.n_agents, self.n)
        role = plan.roles[agent_idx] if agent_idx < len(plan.roles) else f"分析师{agent_idx}"

        upstream = [j for j in range(n)
                    if W[agent_idx, j] > 0 and nodes[j] is not None]

        if not upstream:
            return ADAPTIVE_ROLE_CONTEXT.format(
                role=role, upstream_section="", question=question)

        conclusions = []
        for j in upstream:
            j_role = plan.roles[j] if j < len(plan.roles) else f"分析师{j}"
            c = nodes[j].get("C", "")
            conclusions.append(f"- {j_role}: {c}")

        upstream_text = UPSTREAM_SECTION.format(
            conclusions="\n".join(conclusions))
        return ADAPTIVE_ROLE_CONTEXT.format(
            role=role, upstream_section=upstream_text, question=question)

    def _consensus_decision(self, nodes: List[Dict], plan: PlanResult,
                            use_stackelberg: bool) -> Dict:
        """Option B consensus with depth bonus."""
        valid_nodes = [nd for nd in nodes if nd is not None]
        n = len(valid_nodes)
        if n == 0:
            return {"decision": "neutral", "mechanism": "empty",
                    "trust": [], "ic": [], "ec": []}

        node_indices = [i for i, nd in enumerate(nodes) if nd is not None]
        W = plan.W

        ic = [compute_ic_consistency(nd.get("I", ""), nd.get("C", ""),
                                     self.sim_calc) for nd in valid_nodes]
        ec = [compute_ec_relevance(nd.get("E", []), nd.get("C", ""),
                                   self.sim_calc) for nd in valid_nodes]

        full_bonus = compute_depth_bonus(W)
        depth_bonus = [full_bonus[i] for i in node_indices if i < len(full_bonus)]
        while len(depth_bonus) < n:
            depth_bonus.append(0.0)

        theta_norm = np.ones(n) * 0.5
        if use_stackelberg and n >= 2:
            W_sub = W[np.ix_(node_indices, node_indices)]
            bids = [
                AgentBid(
                    agent_id=f"agent-{node_indices[i]}",
                    model="adaptive",
                    alignment_willingness=max(0.05, ic[i]),
                    revision_cost=max(5.0, 25.0 * (1.0 - 0.5)),
                    max_revision_depth=float(np.clip(ec[i], 0.1, 0.95)),
                    historical_alignment=0.5 * ic[i] + 0.5 * ec[i],
                )
                for i in range(n)
            ]
            game = StackelbergGame(num_agents=n, W=W_sub)
            history = game.run(bids)
            theta = np.array(history[-1]["theta"])
            if theta.max() > theta.min():
                theta_norm = (theta - theta.min()) / (theta.max() - theta.min())

        weights = [
            ic[i] * ec[i] * (1.0 + depth_bonus[i]) * (0.5 + 0.5 * theta_norm[i])
            for i in range(n)
        ]

        stances = [extract_stance(nd.get("C", "")) for nd in valid_nodes]
        vote_scores = {}
        for i, s in enumerate(stances):
            vote_scores[s] = vote_scores.get(s, 0.0) + weights[i]

        if not vote_scores:
            decision, mechanism = "neutral", "empty"
        elif len(set(stances)) == 1:
            decision, mechanism = stances[0], "unanimous"
        else:
            decision = max(vote_scores, key=vote_scores.get)
            mechanism = "depth_weighted_vote"

        return {
            "decision": decision,
            "mechanism": mechanism,
            "trust": weights,
            "ic": ic,
            "ec": ec,
            "depth_bonus": depth_bonus,
            "theta": list(theta_norm),
            "stances": stances,
        }
