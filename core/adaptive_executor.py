"""
Adaptive Executor — 根据Planner生成的拓扑执行多agent推理

支持两种执行模式:
  - DAG (rounds=1): 拓扑排序 → 逐层并行执行
  - Iterative (rounds>1): 多轮迭代，每轮按W可见性传递信息

共识: 纯Stackelberg θ加权投票（委托 trust.evaluate_and_decide）
"""
from __future__ import annotations

import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

from core.llm import LLMClient, MultiLLMPool, AGENT_PERSONAS
from core.planner import PlanResult, topological_layers
from core.similarity import SimilarityCalculator
from core.trust import evaluate_and_decide, extract_stance


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

    def _pool_id(self, plan_idx: int) -> int:
        """Map plan index to pool agent ID."""
        if self._selected is not None:
            return self._selected[plan_idx]
        return plan_idx

    def execute(self, question: str, plan: PlanResult,
                use_stackelberg: bool = True,
                selected_agents: List[int] = None) -> Dict:
        """执行完整流程: 按拓扑执行 → 共识决策。
        selected_agents: 可选，plan index → pool agent ID 的映射列表。
        """
        self._selected = selected_agents
        if plan.rounds == 1:
            nodes, agent_times = self._execute_dag(question, plan)
        else:
            nodes, agent_times = self._execute_iterative(question, plan)

        _tc0 = _time.time()
        result = self._consensus_decision(nodes, plan, use_stackelberg)
        result["consensus_time_s"] = round(_time.time() - _tc0, 4)
        result["nodes"] = nodes
        result["selected_agents"] = selected_agents
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
                pool_id = self._pool_id(i)
                ctx = self._build_dag_context(i, question, plan, nodes)
                persona_id = pool_id % len(AGENT_PERSONAS)
                t0 = _time.time()
                try:
                    if ctx:
                        nodes[i] = self._generate_aeic_ctx(
                            question, ctx, persona_id=persona_id, agent_id=pool_id)
                    else:
                        nodes[i] = self._generate_aeic(
                            question, persona_id=persona_id, agent_id=pool_id)
                except Exception:
                    nodes[i] = None
                agent_times[i] = _time.time() - t0
            else:
                def _call(idx, _q=question, _plan=plan, _nodes=nodes,
                          _pool_id_fn=self._pool_id):
                    pid_pool = _pool_id_fn(idx)
                    ctx = self._build_dag_context(idx, _q, _plan, _nodes)
                    pid = pid_pool % len(AGENT_PERSONAS)
                    t0 = _time.time()
                    try:
                        if ctx:
                            aeic = self._generate_aeic_ctx(
                                _q, ctx, persona_id=pid, agent_id=pid_pool)
                        else:
                            aeic = self._generate_aeic(
                                _q, persona_id=pid, agent_id=pid_pool)
                    except Exception:
                        aeic = None
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

        def _call_r1(i, _pool_id_fn=self._pool_id):
            pool_id = _pool_id_fn(i)
            pid = pool_id % len(AGENT_PERSONAS)
            role = plan.roles[i] if i < len(plan.roles) else f"分析师{i}"
            ctx = ADAPTIVE_ROLE_CONTEXT.format(
                role=role, upstream_section="", question=question)
            t0 = _time.time()
            try:
                aeic = self._generate_aeic_ctx(
                    question, ctx, persona_id=pid, agent_id=pool_id)
            except Exception:
                aeic = None
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

            def _call_iter(i, _prev=prev_nodes, _t=t, _W=W,
                           _pool_id_fn=self._pool_id):
                pool_id = _pool_id_fn(i)
                pid = pool_id % len(AGENT_PERSONAS)
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
                    question, ctx, persona_id=pid, agent_id=pool_id)
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
        """纯Stackelberg共识 — 委托给统一入口 evaluate_and_decide。"""
        valid_nodes = [nd for nd in nodes if nd is not None]
        if not valid_nodes:
            return {"decision": "neutral", "mechanism": "empty",
                    "trust": [], "ic": [], "ec": [], "stances": []}

        node_indices = [i for i, nd in enumerate(nodes) if nd is not None]
        W_sub = plan.W[np.ix_(node_indices, node_indices)]

        return evaluate_and_decide(valid_nodes, W_sub, self.sim_calc,
                                   mode="parallel",
                                   use_stackelberg=use_stackelberg)
