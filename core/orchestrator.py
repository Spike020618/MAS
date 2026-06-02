"""
编排控制器 — 统一4种AEIC交互模式

模式：
  - parallel: 独立推理 + 质量驱动共识
  - sequential: 链式传递 (C→A) + 层间验证
  - star: 层级综合 (C→E) + 双层质量评估
  - debate: 多轮收敛 (C→A跨轮) + 跨轮质量追踪
"""
from __future__ import annotations

import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

from core.llm import LLMClient, MultiLLMPool, AGENT_PERSONAS
from core.trust import evaluate_and_decide, majority_vote, quality_weighted_vote


# ── Prompt Templates for σ-injection ────────────────────────────

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


class Orchestrator:
    """统一编排控制器，封装4种AEIC交互模式的完整执行流。"""

    def __init__(self, llm_client, sim_calc, n_agents: int = 5):
        """
        Args:
            llm_client: LLMClient (single backend) or MultiLLMPool (multi-backend diversity)
            sim_calc: SimilarityCalculator instance
            n_agents: number of agents per question
        """
        self.llm = llm_client
        self.sim_calc = sim_calc
        self.n_agents = n_agents
        self._multi = isinstance(llm_client, MultiLLMPool)

    def _generate_aeic(self, question: str, persona_id: int, agent_id: int):
        if self._multi:
            return self.llm.generate_aeic(question, persona_id=persona_id, agent_id=agent_id)
        return self.llm.generate_aeic(question, persona_id=persona_id)

    def _generate_aeic_ctx(self, question: str, context: str, persona_id: int, agent_id: int):
        if self._multi:
            return self.llm.generate_aeic_with_context(question, context, persona_id=persona_id, agent_id=agent_id)
        return self.llm.generate_aeic_with_context(question, context, persona_id=persona_id)

    def run(self, question: str, mode: str, topology_W: np.ndarray,
            use_stackelberg: bool = True, **kwargs) -> Dict:
        """统一接口。

        Returns:
            {
                "nodes": List[Dict],  # AEIC outputs per agent
                "decision": str,
                "mechanism": str,
                "trust": List[float],
                ...
            }
        """
        runners = {
            "parallel": self._run_parallel,
            "sequential": self._run_sequential,
            "star": self._run_star,
            "debate": self._run_debate,
        }
        if mode not in runners:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(runners.keys())}")
        return runners[mode](question, topology_W, use_stackelberg, **kwargs)

    def _run_parallel(self, question: str, W: np.ndarray,
                      use_stackelberg: bool, **kwargs) -> Dict:
        """P1: 独立推理 + 质量驱动共识"""
        nodes = [None] * self.n_agents
        agent_times = [0.0] * self.n_agents

        def _gen(i):
            persona_id = i % len(AGENT_PERSONAS)
            t0 = _time.time()
            aeic = self._generate_aeic(question, persona_id=persona_id, agent_id=i)
            return i, aeic, _time.time() - t0

        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            futures = [pool.submit(_gen, i) for i in range(self.n_agents)]
            for f in as_completed(futures):
                idx, aeic, elapsed = f.result()
                nodes[idx] = aeic
                agent_times[idx] = elapsed

        result = evaluate_and_decide(nodes, W, self.sim_calc,
                                     mode="parallel",
                                     use_stackelberg=use_stackelberg)
        result["nodes"] = nodes
        result["mode"] = "parallel"
        result["agent_times"] = agent_times
        return result

    def _run_sequential(self, question: str, W: np.ndarray,
                        use_stackelberg: bool, **kwargs) -> Dict:
        """P2: 链式传递 (σ: C_{i-1} → A_i)"""
        nodes = []
        agent_times = []
        for i in range(self.n_agents):
            persona_id = i % len(AGENT_PERSONAS)
            t0 = _time.time()
            if i == 0:
                aeic = self._generate_aeic(question, persona_id=persona_id, agent_id=i)
            else:
                prev_c = nodes[i - 1].get("C", "")
                context = SEQUENTIAL_CONTEXT.format(prev_conclusion=prev_c)
                aeic = self._generate_aeic_ctx(question, context, persona_id=persona_id, agent_id=i)
            agent_times.append(_time.time() - t0)
            nodes.append(aeic)

        result = evaluate_and_decide(nodes, W, self.sim_calc,
                                     mode="sequential",
                                     use_stackelberg=use_stackelberg)
        result["nodes"] = nodes
        result["mode"] = "sequential"
        result["agent_times"] = agent_times
        return result

    def _run_star(self, question: str, W: np.ndarray,
                  use_stackelberg: bool, **kwargs) -> Dict:
        """P3: Leaf并行 → Hub综合 (σ: {C_leaf} → E_hub)"""
        n_leaves = self.n_agents - 1
        agent_times = [0.0] * (n_leaves + 1)  # hub=0, leaves=1..N-1

        # Phase 1: Leaf agents produce AEIC independently (parallel)
        leaf_nodes = [None] * n_leaves

        def _gen_leaf(i):
            persona_id = (i + 1) % len(AGENT_PERSONAS)
            t0 = _time.time()
            aeic = self._generate_aeic(question, persona_id=persona_id, agent_id=i + 1)
            return i, aeic, _time.time() - t0

        with ThreadPoolExecutor(max_workers=n_leaves) as pool:
            futures = [pool.submit(_gen_leaf, i) for i in range(n_leaves)]
            for f in as_completed(futures):
                idx, aeic, elapsed = f.result()
                leaf_nodes[idx] = aeic
                agent_times[idx + 1] = elapsed

        # Compute leaf trust to inform hub
        leaf_W = W[1:, 1:]
        leaf_result = evaluate_and_decide(leaf_nodes, leaf_W, self.sim_calc,
                                          mode="parallel",
                                          use_stackelberg=False)
        leaf_trust = leaf_result["trust"]

        # Phase 2: Hub receives leaf conclusions + trust scores
        sorted_leaves = sorted(range(n_leaves),
                               key=lambda j: leaf_trust[j], reverse=True)
        lines = []
        for rank, j in enumerate(sorted_leaves):
            c = leaf_nodes[j].get("C", "")
            t = leaf_trust[j]
            lines.append(f"- 专家{j+1} (可信度{t:.0%}): {c}")
        leaf_conclusions_text = "\n".join(lines)

        context = STAR_HUB_CONTEXT.format(leaf_conclusions=leaf_conclusions_text)
        t0 = _time.time()
        hub_aeic = self._generate_aeic_ctx(question, context, persona_id=0, agent_id=0)
        agent_times[0] = _time.time() - t0

        # Combine: hub is node 0, leaves are nodes 1..N-1
        nodes = [hub_aeic] + leaf_nodes
        result = evaluate_and_decide(nodes, W, self.sim_calc,
                                     mode="parallel",
                                     use_stackelberg=use_stackelberg)
        result["nodes"] = nodes
        result["mode"] = "star"
        result["hub_index"] = 0
        result["hub_decision"] = leaf_result.get("decision", "")
        result["agent_times"] = agent_times
        return result

    def _run_debate(self, question: str, W: np.ndarray,
                    use_stackelberg: bool,
                    max_rounds: int = 3, **kwargs) -> Dict:
        """P4: 多轮辩论 (σ: {C_j^(t-1)} → A_i^(t))"""
        max_rounds = kwargs.get("max_rounds", max_rounds)
        all_rounds = []
        all_agent_times = []

        # Round 1: Independent (parallel)
        nodes = [None] * self.n_agents
        round_times = [0.0] * self.n_agents

        def _gen_r1(i):
            persona_id = i % len(AGENT_PERSONAS)
            t0 = _time.time()
            aeic = self._generate_aeic(question, persona_id=persona_id, agent_id=i)
            return i, aeic, _time.time() - t0

        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            futures = [pool.submit(_gen_r1, i) for i in range(self.n_agents)]
            for f in as_completed(futures):
                idx, aeic, elapsed = f.result()
                nodes[idx] = aeic
                round_times[idx] = elapsed
        all_rounds.append(nodes)
        all_agent_times.append(round_times)

        # Rounds 2+: 拓扑约束
        for t in range(2, max_rounds + 1):
            prev_nodes = all_rounds[-1]
            new_nodes = [None] * self.n_agents
            rt = [0.0] * self.n_agents

            def _gen_debate(i, _prev=prev_nodes, _t=t, _W=W):
                persona_id = i % len(AGENT_PERSONAS)
                t0 = _time.time()
                other_lines = []
                for j in range(self.n_agents):
                    if j != i and _W[i, j] > 0:
                        c = _prev[j].get("C", "")
                        other_lines.append(f"- 分析师{j+1}: {c}")
                if not other_lines:
                    other_lines.append("（无可见邻居结论）")
                other_text = "\n".join(other_lines)
                my_prev_c = _prev[i].get("C", "")
                context = DEBATE_CONTEXT.format(
                    round_num=_t,
                    other_conclusions=other_text,
                    my_prev_conclusion=my_prev_c,
                )
                aeic = self._generate_aeic_ctx(question, context, persona_id=persona_id, agent_id=i)
                return i, aeic, _time.time() - t0

            with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
                futures = [pool.submit(_gen_debate, i) for i in range(self.n_agents)]
                for f in as_completed(futures):
                    idx, aeic, elapsed = f.result()
                    new_nodes[idx] = aeic
                    rt[idx] = elapsed

            all_rounds.append(new_nodes)
            all_agent_times.append(rt)

            from core.trust import extract_stance
            prev_stances = [extract_stance(nd.get("C", "")) for nd in prev_nodes]
            cur_stances = [extract_stance(nd.get("C", "")) for nd in new_nodes]
            if prev_stances == cur_stances:
                break

        # Aggregate agent_times: sum across rounds per agent
        agent_times = [0.0] * self.n_agents
        for rt in all_agent_times:
            for i in range(self.n_agents):
                agent_times[i] += rt[i]

        final_nodes = all_rounds[-1]
        result = evaluate_and_decide(final_nodes, W, self.sim_calc,
                                     mode="debate",
                                     use_stackelberg=use_stackelberg)
        result["nodes"] = final_nodes
        result["mode"] = "debate"
        result["num_rounds"] = len(all_rounds)
        result["all_rounds"] = all_rounds
        result["agent_times"] = agent_times
        return result
