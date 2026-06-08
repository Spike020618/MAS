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

DEBATE_CONTEXT_ONLINE = """这是第{round_num}轮讨论。上一轮各方的结论如下：
{other_conclusions}
你上一轮的结论是: {my_prev_conclusion}

机制评估反馈（上轮表现）：
{quality_feedback}
高质量推理（证据具体、逻辑严密）将获得更高的共识权重。

请在考虑这些不同观点后，重新进行AEIC推理。
你可以维持或修正你的立场，但需在Inference中给出理由。"""

HIERARCHICAL_SUBGROUP_ONLINE = """你所在的讨论小组中，其他成员的初步结论如下：
{group_conclusions}

机制评估反馈（独立阶段表现）：
{quality_feedback}
高质量推理将获得更高权重，请确保证据具体且推理链完整。

请在考虑小组成员的观点后，重新进行AEIC推理。
注意综合你的独立判断和小组讨论的启发，给出修正后的结论。"""

HIERARCHICAL_SYNTHESIS_ONLINE = """各讨论小组已完成分组讨论，以下是各小组的代表性结论：
{subgroup_summaries}

机制评估反馈（分组讨论表现）：
{quality_feedback}
在最终综合中，推理质量更高的分析将获得更大影响力。

请综合所有小组的观点，进行最终的AEIC推理。
在Inference中权衡不同小组的分歧与共识，给出你的最终Conclusion。"""

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

# ── Hierarchical 总分总 ────────────────────────────────────────
HIERARCHICAL_SUBGROUP_CONTEXT = """你所在的讨论小组中，其他成员的初步结论如下：
{group_conclusions}

请在考虑小组成员的观点后，重新进行AEIC推理。
注意综合你的独立判断和小组讨论的启发，给出修正后的结论。"""

HIERARCHICAL_SYNTHESIS_CONTEXT = """各讨论小组已完成分组讨论，以下是各小组的代表性结论：
{subgroup_summaries}

请综合所有小组的观点，进行最终的AEIC推理。
在Inference中权衡不同小组的分歧与共识，给出你的最终Conclusion。"""

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

    def _compute_online_feedback(self, nodes, W, config=None):
        """Compute interim AEIC signals and format quality feedback for injection."""
        result = evaluate_and_decide(nodes, W, self.sim_calc,
                                      mode="parallel",
                                      use_stackelberg=True, config=config)
        lines = []
        for i in range(len(nodes)):
            ic_i = result["ic"][i]
            ec_i = result["ec"][i]
            theta_i = result["theta"][i]
            lines.append(
                f"- 分析师{i+1}: 推理一致性={ic_i:.0%}, "
                f"证据支持度={ec_i:.0%}, 当前权重={theta_i:.0%}"
            )
        return "\n".join(lines), result

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
            "hierarchical": self._run_hierarchical,
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

        # Compute leaf trust to inform hub (online Stackelberg)
        leaf_W = W[1:, 1:]
        leaf_result = evaluate_and_decide(leaf_nodes, leaf_W, self.sim_calc,
                                          mode="parallel",
                                          use_stackelberg=use_stackelberg)
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
        """P4: 多轮辩论 (σ: {C_j^(t-1)} → A_i^(t))

        When online=True, computes interim AEIC signals between rounds and
        injects quality feedback into agent prompts (online Stackelberg).
        """
        max_rounds = kwargs.get("max_rounds", max_rounds)
        online = kwargs.get("online", False)
        all_rounds = []
        all_agent_times = []
        online_feedback_history = []

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

        # Rounds 2+: 拓扑约束 (with optional online feedback)
        for t in range(2, max_rounds + 1):
            prev_nodes = all_rounds[-1]

            # Online Stackelberg: compute interim signals and quality feedback
            quality_feedback_text = ""
            if online:
                quality_feedback_text, interim_result = \
                    self._compute_online_feedback(prev_nodes, W)
                online_feedback_history.append(interim_result)

            new_nodes = [None] * self.n_agents
            rt = [0.0] * self.n_agents

            def _gen_debate(i, _prev=prev_nodes, _t=t, _W=W,
                            _online=online, _fb=quality_feedback_text):
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

                if _online and _fb:
                    context = DEBATE_CONTEXT_ONLINE.format(
                        round_num=_t,
                        other_conclusions=other_text,
                        my_prev_conclusion=my_prev_c,
                        quality_feedback=_fb,
                    )
                else:
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
        if online:
            result["online_feedback_history"] = online_feedback_history
        return result

    def _run_hierarchical(self, question: str, W: np.ndarray,
                          use_stackelberg: bool, **kwargs) -> Dict:
        """P5: 总分总 — 独立→分组讨论→全局综合

        When online=True, computes interim signals between phases and injects
        quality feedback into agent prompts (online Stackelberg).
        """
        online = kwargs.get("online", False)
        all_agent_times = [[0.0] * self.n_agents for _ in range(3)]
        online_feedback_history = []

        # Phase 1 (总): All agents independently generate AEIC
        phase1_nodes = [None] * self.n_agents

        def _gen_p1(i):
            persona_id = i % len(AGENT_PERSONAS)
            t0 = _time.time()
            aeic = self._generate_aeic(question, persona_id=persona_id, agent_id=i)
            return i, aeic, _time.time() - t0

        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            futures = [pool.submit(_gen_p1, i) for i in range(self.n_agents)]
            for f in as_completed(futures):
                idx, aeic, elapsed = f.result()
                phase1_nodes[idx] = aeic
                all_agent_times[0][idx] = elapsed

        # Online feedback after Phase 1
        p1_feedback_text = ""
        if online:
            p1_feedback_text, p1_interim = \
                self._compute_online_feedback(phase1_nodes, W)
            online_feedback_history.append(p1_interim)

        # Phase 2 (分): Split into subgroups for focused discussion
        mid = self.n_agents // 2
        groups = [list(range(mid)), list(range(mid, self.n_agents))]
        phase2_nodes = [None] * self.n_agents

        def _gen_subgroup(i, group_members, _online=online, _fb=p1_feedback_text):
            persona_id = i % len(AGENT_PERSONAS)
            t0 = _time.time()
            other_lines = []
            for j in group_members:
                if j != i:
                    c = phase1_nodes[j].get("C", "")
                    other_lines.append(f"- 成员{j+1}: {c}")

            if _online and _fb:
                context = HIERARCHICAL_SUBGROUP_ONLINE.format(
                    group_conclusions="\n".join(other_lines),
                    quality_feedback=_fb)
            else:
                context = HIERARCHICAL_SUBGROUP_CONTEXT.format(
                    group_conclusions="\n".join(other_lines))
            aeic = self._generate_aeic_ctx(question, context, persona_id=persona_id, agent_id=i)
            return i, aeic, _time.time() - t0

        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            futures = []
            for group in groups:
                for i in group:
                    futures.append(pool.submit(_gen_subgroup, i, group))
            for f in as_completed(futures):
                idx, aeic, elapsed = f.result()
                phase2_nodes[idx] = aeic
                all_agent_times[1][idx] = elapsed

        # Online feedback after Phase 2
        p2_feedback_text = ""
        if online:
            p2_feedback_text, p2_interim = \
                self._compute_online_feedback(phase2_nodes, W)
            online_feedback_history.append(p2_interim)

        # Phase 3 (总): All agents see subgroup summaries and synthesize
        subgroup_lines = []
        for g_idx, group in enumerate(groups):
            for j in group:
                c = phase2_nodes[j].get("C", "")
                subgroup_lines.append(f"- 小组{g_idx+1}-成员{j+1}: {c}")
        subgroup_text = "\n".join(subgroup_lines)

        phase3_nodes = [None] * self.n_agents

        def _gen_synthesis(i, _online=online, _fb=p2_feedback_text):
            persona_id = i % len(AGENT_PERSONAS)
            t0 = _time.time()
            if _online and _fb:
                context = HIERARCHICAL_SYNTHESIS_ONLINE.format(
                    subgroup_summaries=subgroup_text,
                    quality_feedback=_fb)
            else:
                context = HIERARCHICAL_SYNTHESIS_CONTEXT.format(
                    subgroup_summaries=subgroup_text)
            aeic = self._generate_aeic_ctx(question, context, persona_id=persona_id, agent_id=i)
            return i, aeic, _time.time() - t0

        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            futures = [pool.submit(_gen_synthesis, i) for i in range(self.n_agents)]
            for f in as_completed(futures):
                idx, aeic, elapsed = f.result()
                phase3_nodes[idx] = aeic
                all_agent_times[2][idx] = elapsed

        agent_times = [sum(all_agent_times[p][i] for p in range(3))
                       for i in range(self.n_agents)]

        result = evaluate_and_decide(phase3_nodes, W, self.sim_calc,
                                     mode="parallel",
                                     use_stackelberg=use_stackelberg)
        result["nodes"] = phase3_nodes
        result["mode"] = "hierarchical"
        result["phases"] = [phase1_nodes, phase2_nodes, phase3_nodes]
        result["agent_times"] = agent_times
        if online:
            result["online_feedback_history"] = online_feedback_history
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
