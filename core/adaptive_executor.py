"""
Adaptive Executor — 根据Planner生成的拓扑执行多agent推理

支持两种执行模式:
  - DAG (rounds=1): 拓扑排序 → 逐层并行执行
  - Iterative (rounds>1): 多轮迭代，每轮按W可见性传递信息

共识: Option B — 全节点 depth-weighted + IC/EC + Stackelberg
"""
from __future__ import annotations

import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

from core.llm import LLMClient, MultiLLMPool, AGENT_PERSONAS
from core.planner import PlanResult, compute_depth_bonus, topological_layers
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

重要：上游结论仅供参考，可能包含错误。你必须独立验证其推理链是否成立，而非默认接受。如果你发现上游的证据不足或逻辑有缺陷，应明确指出并给出你自己的独立判断。"""

ITERATIVE_CONTEXT = """你的角色: {role}

这是第{round_num}轮讨论。上一轮你能看到的分析师结论如下：
{visible_conclusions}
你上一轮的结论是: {my_prev}

请独立评估这些观点的证据质量和逻辑链。如果他人观点有充分依据，可以修正立场；如果你认为自己的证据更可靠，应坚持原判断。

问题: {question}"""


class AdaptiveExecutor:
    """自适应拓扑执行器。"""

    def __init__(self, llm_pool, sim_calc: SimilarityCalculator,
                 n_agents: int = 5):
        self.llm = llm_pool
        self.sim_calc = sim_calc
        self.n_agents = n_agents
        self._multi = isinstance(llm_pool, MultiLLMPool)

    def _generate_aeic(self, question: str, persona_id: int, agent_id: int):
        if self._multi:
            return self.llm.generate_aeic(question, persona_id=persona_id,
                                          agent_id=agent_id)
        return self.llm.generate_aeic(question, persona_id=persona_id)

    def _generate_aeic_ctx(self, question: str, context: str,
                           persona_id: int, agent_id: int):
        if self._multi:
            return self.llm.generate_aeic_with_context(
                question, context, persona_id=persona_id, agent_id=agent_id)
        return self.llm.generate_aeic_with_context(
            question, context, persona_id=persona_id)

    def execute(self, question: str, plan: PlanResult,
                use_stackelberg: bool = True) -> Dict:
        """执行完整流程: 按拓扑执行 → 共识决策。"""
        if plan.rounds == 1:
            nodes, agent_times = self._execute_dag(question, plan)
        else:
            nodes, agent_times = self._execute_iterative(question, plan)

        _tc0 = _time.time()
        result = self._consensus_decision(nodes, plan, use_stackelberg)
        result["consensus_time_s"] = round(_time.time() - _tc0, 4)
        result["nodes"] = nodes
        result["agent_times"] = agent_times
        result["plan_reasoning"] = plan.reasoning
        result["execution_order"] = plan.execution_order
        result["roles"] = plan.roles
        return result

    def _execute_dag(self, question: str, plan: PlanResult):
        """DAG模式: 按拓扑层次逐层执行，层内并行。返回 (nodes, agent_times)。"""
        n = plan.n_agents
        nodes = [None] * n
        agent_times = [0.0] * n
        W = plan.W

        for layer in plan.execution_order:
            if len(layer) == 1:
                i = layer[0]
                ctx = self._build_dag_context(i, question, plan, nodes)
                persona_id = i % len(AGENT_PERSONAS)
                t0 = _time.time()
                if ctx:
                    nodes[i] = self._generate_aeic_ctx(
                        question, ctx, persona_id=persona_id, agent_id=i)
                else:
                    nodes[i] = self._generate_aeic(
                        question, persona_id=persona_id, agent_id=i)
                agent_times[i] = _time.time() - t0
            else:
                def _call(idx, _q=question, _plan=plan, _nodes=nodes):
                    ctx = self._build_dag_context(idx, _q, _plan, _nodes)
                    pid = idx % len(AGENT_PERSONAS)
                    t0 = _time.time()
                    if ctx:
                        aeic = self._generate_aeic_ctx(
                            _q, ctx, persona_id=pid, agent_id=idx)
                    else:
                        aeic = self._generate_aeic(
                            _q, persona_id=pid, agent_id=idx)
                    return idx, aeic, _time.time() - t0

                with ThreadPoolExecutor(max_workers=len(layer)) as pool:
                    futures = [pool.submit(_call, i) for i in layer]
                    for f in as_completed(futures):
                        idx, aeic, elapsed = f.result()
                        nodes[idx] = aeic
                        agent_times[idx] = elapsed

        return nodes, agent_times

    def _execute_iterative(self, question: str,
                           plan: PlanResult):
        """迭代模式: 多轮执行，每轮按W可见性传递。返回 (nodes, agent_times)。"""
        n = plan.n_agents
        W = plan.W
        all_rounds = []
        all_round_times = []

        nodes = [None] * n
        round_times = [0.0] * n

        def _call_r1(i):
            pid = i % len(AGENT_PERSONAS)
            role = plan.roles[i] if i < len(plan.roles) else f"分析师{i}"
            ctx = ADAPTIVE_ROLE_CONTEXT.format(
                role=role, upstream_section="", question=question)
            t0 = _time.time()
            aeic = self._generate_aeic_ctx(
                question, ctx, persona_id=pid, agent_id=i)
            return i, aeic, _time.time() - t0

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_call_r1, i) for i in range(n)]
            for f in as_completed(futures):
                idx, aeic, elapsed = f.result()
                nodes[idx] = aeic
                round_times[idx] = elapsed
        all_rounds.append(list(nodes))
        all_round_times.append(round_times)

        for t in range(2, plan.rounds + 1):
            prev_nodes = all_rounds[-1]
            new_nodes = [None] * n
            rt = [0.0] * n

            def _call_iter(i, _prev=prev_nodes, _t=t, _W=W):
                pid = i % len(AGENT_PERSONAS)
                role = plan.roles[i] if i < len(plan.roles) else f"分析师{i}"
                visible = []
                for j in range(n):
                    if j != i and _W[i, j] > 0:
                        c = _prev[j].get("C", "") if _prev[j] else ""
                        visible.append(f"- 分析师{j}({plan.roles[j] if j < len(plan.roles) else ''}): {c}")
                vis_text = "\n".join(visible) if visible else "（无可见邻居）"
                my_prev = _prev[i].get("C", "") if _prev[i] else ""
                ctx = ITERATIVE_CONTEXT.format(
                    role=role, round_num=_t,
                    visible_conclusions=vis_text,
                    my_prev=my_prev, question=question)
                t0 = _time.time()
                aeic = self._generate_aeic_ctx(
                    question, ctx, persona_id=pid, agent_id=i)
                return i, aeic, _time.time() - t0

            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(_call_iter, i) for i in range(n)]
                for f in as_completed(futures):
                    idx, aeic, elapsed = f.result()
                    new_nodes[idx] = aeic
                    rt[idx] = elapsed

            all_rounds.append(new_nodes)
            all_round_times.append(rt)

            if plan.round_mode == "converge":
                prev_stances = [extract_stance(nd.get("C", ""))
                                for nd in prev_nodes if nd]
                cur_stances = [extract_stance(nd.get("C", ""))
                               for nd in new_nodes if nd]
                if prev_stances == cur_stances:
                    break

        # Sum agent times across rounds
        agent_times = [0.0] * n
        for rt in all_round_times:
            for i in range(n):
                agent_times[i] += rt[i]

        return all_rounds[-1], agent_times

    def _build_dag_context(self, agent_idx: int, question: str,
                           plan: PlanResult,
                           nodes: List[Dict]) -> str:
        """构建DAG模式下agent的context（包含role + 上游结论）。"""
        W = plan.W
        role = plan.roles[agent_idx] if agent_idx < len(plan.roles) else f"分析师{agent_idx}"

        upstream_indices = [j for j in range(plan.n_agents)
                           if W[agent_idx, j] > 0 and nodes[j] is not None]

        if not upstream_indices:
            ctx = ADAPTIVE_ROLE_CONTEXT.format(
                role=role, upstream_section="", question=question)
            return ctx

        conclusions = []
        for j in upstream_indices:
            j_role = plan.roles[j] if j < len(plan.roles) else f"分析师{j}"
            c = nodes[j].get("C", "")
            conclusions.append(f"- {j_role}: {c}")

        upstream_text = UPSTREAM_SECTION.format(
            conclusions="\n".join(conclusions))
        ctx = ADAPTIVE_ROLE_CONTEXT.format(
            role=role, upstream_section=upstream_text, question=question)
        return ctx

    def _consensus_decision(self, nodes: List[Dict], plan: PlanResult,
                            use_stackelberg: bool) -> Dict:
        """Option B: 全节点 depth-weighted + IC/EC + Stackelberg。"""
        valid_nodes = [nd for nd in nodes if nd is not None]
        n = len(valid_nodes)
        if n == 0:
            return {"decision": "neutral", "mechanism": "empty",
                    "trust": [], "ic": [], "ec": []}

        node_indices = [i for i, nd in enumerate(nodes) if nd is not None]
        W = plan.W

        ic = [compute_ic_consistency(nd.get("I", ""), nd.get("C", ""),
                                     self.sim_calc)
              for nd in valid_nodes]
        ec = [compute_ec_relevance(nd.get("E", []), nd.get("C", ""),
                                   self.sim_calc)
              for nd in valid_nodes]

        full_bonus = compute_depth_bonus(W)
        depth_bonus = [full_bonus[i] for i in node_indices]

        theta = [0.5] * n
        W_sub = W[np.ix_(node_indices, node_indices)]
        w_has_edges = W_sub.any()
        if use_stackelberg and n >= 2 and w_has_edges:
            stances_for_agree = [extract_stance(nd.get("C", "")) for nd in valid_nodes]
            agreement = []
            for i in range(n):
                neighbors = [j for j in range(n) if j != i and W_sub[i, j] > 0]
                if not neighbors:
                    agreement.append(0.5)
                else:
                    agree_count = sum(1 for j in neighbors
                                      if stances_for_agree[j] == stances_for_agree[i])
                    agreement.append(agree_count / len(neighbors))
            bids = []
            for i in range(n):
                bids.append(AgentBid(
                    agent_id=f"agent-{node_indices[i]}",
                    model="adaptive",
                    alignment_willingness=max(0.05, ic[i]),
                    revision_cost=max(5.0, 30.0 * (1.0 - agreement[i])),
                    max_revision_depth=float(np.clip(ec[i], 0.1, 0.95)),
                    historical_alignment=0.5 * ic[i] + 0.5 * ec[i],
                ))
            game = StackelbergGame(num_agents=n, W=W_sub)
            history = game.run(bids)
            theta = list(history[-1]["theta"])

        theta_arr = np.array(theta)
        if theta_arr.max() > theta_arr.min():
            theta_norm = (theta_arr - theta_arr.min()) / \
                         (theta_arr.max() - theta_arr.min())
        else:
            theta_norm = np.ones(n) * 0.5

        weights = [
            ic[i] * ec[i] * (1.0 + depth_bonus[i]) * (0.5 + 0.5 * theta_norm[i])
            for i in range(n)
        ]

        stances = [extract_stance(nd.get("C", "")) for nd in valid_nodes]
        vote_scores = {}
        for i, s in enumerate(stances):
            if s != "neutral":
                vote_scores[s] = vote_scores.get(s, 0.0) + weights[i]

        if not vote_scores:
            decision = "neutral"
            mechanism = "no_decisive_vote"
        elif len(vote_scores) == 1:
            decision = list(vote_scores.keys())[0]
            mechanism = "unanimous"
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
