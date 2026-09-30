"""Executable workflow: retrieve -> deliberate -> consensus -> auditable trace."""
from __future__ import annotations

import uuid
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from .agents import CandidateAgent, ConstraintVerifierAgent, PreferenceAgent, RetrievalAgent
from .catalog import InMemoryCatalog
from .consensus import ConsensusEngine
from .domain import SearchRequest, SearchResponse, SearchTrace, TraceEvent
from .planner import ConsensusPlanner
from .deepseek import DeepSeekClient, DeepSeekConfig, DeepSeekError
from .deepseek_agents import DeepSeekReviewAgent
from .movie_data import JsonlMovieCatalog
from .game_agents import ContractualDeepSeekReviewAgent

# DeepSeek 输出上限 8192 tokens，而合同行动的 token 预算（700-3200）装不下
# 50 个候选的批量评审（必截断为非法 JSON）。LLM 评审聚焦头部候选，其余
# 候选由确定性 Agent 覆盖；可用 DEEPSEEK_REVIEW_LIMIT 调整。
LLM_REVIEW_LIMIT = int(os.environ.get("DEEPSEEK_REVIEW_LIMIT", "10"))


class ConsensusWorkflow:
    def __init__(
        self,
        catalog: Optional[InMemoryCatalog] = None,
        agents: Optional[Iterable[CandidateAgent]] = None,
        engine: Optional[ConsensusEngine] = None,
        planner: Optional[ConsensusPlanner] = None,
    ):
        self.catalog = catalog or InMemoryCatalog()
        self.agents = list(agents) if agents is not None else [
            RetrievalAgent(self.catalog), PreferenceAgent(), ConstraintVerifierAgent()
        ]
        self.engine = engine or ConsensusEngine()
        self.planner = planner or ConsensusPlanner()

    @classmethod
    def from_environment(cls) -> "ConsensusWorkflow":
        """Use one DeepSeek key when present; remain runnable without it."""
        configured_catalog = os.environ.get("MOVIE_CATALOG_PATH", "").strip()
        default_catalog = Path(__file__).resolve().parents[2] / "experiments/data/processed/movie_catalog.jsonl"
        catalog = JsonlMovieCatalog(configured_catalog or default_catalog) \
            if configured_catalog or default_catalog.is_file() else InMemoryCatalog()
        config = DeepSeekConfig.from_environment()
        if config is None:
            return cls(catalog=catalog)
        client = DeepSeekClient(config)
        agents = [
            RetrievalAgent(catalog),
            PreferenceAgent(),
            ConstraintVerifierAgent(),
            ContractualDeepSeekReviewAgent(client, "preference", "deepseek-preference-1"),
            ContractualDeepSeekReviewAgent(client, "diversity", "deepseek-diversity-1"),
            ContractualDeepSeekReviewAgent(client, "critic", "deepseek-critic-1"),
        ]
        return cls(catalog=catalog, agents=agents)

    def run(self, request: SearchRequest) -> SearchResponse:
        errors = request.validate()
        if errors:
            raise ValueError("; ".join(errors))
        request_id = uuid.uuid4().hex
        plan = self.planner.plan(request)
        trace = SearchTrace(
            request_id=request_id,
            mechanism="public_static_aeic_then_evidence_reputation_v1",
            topology=plan.topology,
            plan=plan,
            reputations_before=self.engine.snapshot(),
        )
        trace.events.append(TraceEvent("planning", "plan_selected", {
            "risk_level": plan.risk_level,
            "complexity_score": plan.complexity_score,
            "agent_roles": plan.agent_roles,
            "verification_budget": plan.verification_budget,
        }))
        trace.events.append(TraceEvent("mechanism", "public_contract_committed", {
            "mechanism": trace.mechanism,
            "timing": "contract_before_follower_action",
        }))
        candidates = self.catalog.search(request)
        trace.events.append(TraceEvent("retrieval", "candidates_retrieved", {
            "count": len(candidates), "candidate_ids": [item.movie_id for item in candidates]
        }))

        reports = []
        selected_roles = set(plan.agent_roles)
        if any(agent.role in {"diversity", "critic"} for agent in self.agents):
            selected_roles.update({"diversity", "critic"})
        selected_agents = [agent for agent in self.agents if agent.role in selected_roles]
        for agent in selected_agents:
            try:
                review_candidates = candidates[:LLM_REVIEW_LIMIT] \
                    if isinstance(agent, DeepSeekReviewAgent) else candidates
                agent_reports = agent.review(request, review_candidates)
            except DeepSeekError as error:
                trace.events.append(TraceEvent("deliberation", "agent_degraded", {
                    "agent_id": agent.agent_id, "role": agent.role,
                    "reason": str(error), "fallback": "deterministic_agents_continue",
                }))
                continue
            for report in agent_reports:
                report_errors = report.validate()
                if report_errors:
                    raise ValueError(f"invalid report from {agent.agent_id}: {'; '.join(report_errors)}")
            reports.extend(agent_reports)
            action_counts = Counter(report.action for report in agent_reports)
            trace.events.append(TraceEvent("deliberation", "agent_completed", {
                "agent_id": agent.agent_id, "role": agent.role, "reports": len(agent_reports),
                "actions": dict(sorted(action_counts.items())),
                "tokens": max((report.token_cost for report in agent_reports), default=0),
                "latency_ms": max((report.latency_ms for report in agent_reports), default=0),
            }))
        trace.reports = reports

        decisions = self.engine.decide(candidates, reports, request.top_k)
        trace.reputations_after = self.engine.snapshot()
        trace.events.append(TraceEvent("consensus", "ranking_completed", {
            "selected": [item.candidate.movie_id for item in decisions],
            "scores": {item.candidate.movie_id: item.score for item in decisions},
        }))
        return SearchResponse(request_id, decisions, trace)
