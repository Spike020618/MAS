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

# ── Self-Correction (Paper 2 baseline) ─────────────────────────
SELFCORRECT_CONTEXT = """这是第{round_num}轮自我修正。你上一轮的分析如下：

A: {prev_A}
E: {prev_E}
I: {prev_I}
C: {prev_C}

请仔细检查你之前的推理过程：
1. 假设(A)是否完整覆盖了关键前提？
2. 证据(E)是否有遗漏或错误？
3. 推理(I)的逻辑链是否严密？
4. 结论(C)是否与证据一致？

如果发现问题，请修正后重新给出完整的AEIC分析。如果确认无误，可以维持原结论。"""

# ── AgentVerse Solver-Reviewer (Paper 1 baseline) ──────────────
AGENTVERSE_REVIEW_CONTEXT = """一位分析师对以下问题给出了结论：'{solver_conclusion}'

作为审查者，请独立进行AEIC推理，批判性地评估上述结论：
- 如果你认为该分析有误，指出具体问题并给出你的独立判断
- 如果你认同，请提供额外的支持证据"""

AGENTVERSE_REVISE_CONTEXT = """你之前的结论是: {my_prev_conclusion}

{n_reviewers}位审查者给出了以下反馈：
{reviewer_conclusions}

请基于这些反馈，修正或确认你的分析。重新给出完整的AEIC推理。"""


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
            "selfcorrect": self._run_selfcorrect,
            "agentverse": self._run_agentverse,
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

    def _run_selfcorrect(self, question: str, W: np.ndarray,
                         use_stackelberg: bool,
                         max_rounds: int = 3, **kwargs) -> Dict:
        """Baseline (Paper 2): 每个agent独立自我修正R轮，无peer通信。"""
        max_rounds = kwargs.get("max_rounds", max_rounds)
        all_rounds = []
        all_agent_times = []

        # Round 1: Independent
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
        all_rounds.append(list(nodes))
        all_agent_times.append(round_times)

        # Rounds 2+: each agent revises OWN previous output only
        for t in range(2, max_rounds + 1):
            prev_nodes = all_rounds[-1]
            new_nodes = [None] * self.n_agents
            rt = [0.0] * self.n_agents

            def _gen_sc(i, _prev=prev_nodes, _t=t):
                persona_id = i % len(AGENT_PERSONAS)
                t0 = _time.time()
                prev = _prev[i] if _prev[i] else {}
                e_str = "; ".join(str(x) for x in prev.get("E", []))
                context = SELFCORRECT_CONTEXT.format(
                    round_num=_t,
                    prev_A=prev.get("A", ""),
                    prev_E=e_str,
                    prev_I=prev.get("I", ""),
                    prev_C=prev.get("C", ""),
                )
                aeic = self._generate_aeic_ctx(
                    question, context, persona_id=persona_id, agent_id=i)
                return i, aeic, _time.time() - t0

            with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
                futures = [pool.submit(_gen_sc, i) for i in range(self.n_agents)]
                for f in as_completed(futures):
                    idx, aeic, elapsed = f.result()
                    new_nodes[idx] = aeic
                    rt[idx] = elapsed

            all_rounds.append(new_nodes)
            all_agent_times.append(rt)

        agent_times = [0.0] * self.n_agents
        for rt in all_agent_times:
            for i in range(self.n_agents):
                agent_times[i] += rt[i]

        final_nodes = all_rounds[-1]
        result = evaluate_and_decide(final_nodes, W, self.sim_calc,
                                     mode="parallel",
                                     use_stackelberg=use_stackelberg)
        result["nodes"] = final_nodes
        result["mode"] = "selfcorrect"
        result["num_rounds"] = len(all_rounds)
        result["agent_times"] = agent_times
        return result

    def _run_agentverse(self, question: str, W: np.ndarray,
                        use_stackelberg: bool, **kwargs) -> Dict:
        """Baseline (Paper 1): Solver提出 → Reviewers批判 → Solver修正。"""
        n_reviewers = self.n_agents - 1
        agent_times = [0.0] * self.n_agents

        # Phase 1: Solver (agent 0) generates initial AEIC
        t0 = _time.time()
        solver_aeic = self._generate_aeic(question, persona_id=0, agent_id=0)
        agent_times[0] = _time.time() - t0

        solver_c = solver_aeic.get("C", "")

        # Phase 2: Reviewers see solver's conclusion, independently produce AEIC
        reviewer_nodes = [None] * n_reviewers

        def _gen_review(i):
            persona_id = (i + 1) % len(AGENT_PERSONAS)
            t0 = _time.time()
            context = AGENTVERSE_REVIEW_CONTEXT.format(
                solver_conclusion=solver_c)
            aeic = self._generate_aeic_ctx(
                question, context, persona_id=persona_id, agent_id=i + 1)
            return i, aeic, _time.time() - t0

        with ThreadPoolExecutor(max_workers=n_reviewers) as pool:
            futures = [pool.submit(_gen_review, i) for i in range(n_reviewers)]
            for f in as_completed(futures):
                idx, aeic, elapsed = f.result()
                reviewer_nodes[idx] = aeic
                agent_times[idx + 1] = elapsed

        # Phase 3: Solver revises based on reviewer feedback
        reviewer_lines = []
        for j, rn in enumerate(reviewer_nodes):
            if rn:
                reviewer_lines.append(f"- 审查者{j+1}: {rn.get('C', '')}")
        reviewer_text = "\n".join(reviewer_lines) if reviewer_lines else "（无反馈）"

        context = AGENTVERSE_REVISE_CONTEXT.format(
            my_prev_conclusion=solver_c,
            n_reviewers=n_reviewers,
            reviewer_conclusions=reviewer_text,
        )
        t0 = _time.time()
        solver_revised = self._generate_aeic_ctx(
            question, context, persona_id=0, agent_id=0)
        agent_times[0] += _time.time() - t0

        # All nodes: revised solver + reviewers
        nodes = [solver_revised] + reviewer_nodes

        result = evaluate_and_decide(nodes, W, self.sim_calc,
                                     mode="parallel",
                                     use_stackelberg=use_stackelberg)
        result["nodes"] = nodes
        result["mode"] = "agentverse"
        result["agent_times"] = agent_times
        return result
