"""Executable workflow: retrieve -> deliberate -> consensus -> auditable trace."""
from __future__ import annotations

import uuid
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from .agents import CandidateAgent, ConstraintVerifierAgent, PreferenceAgent, RetrievalAgent
from .catalog import InMemoryCatalog
from .consensus import ConsensusEngine
from .domain import SearchRequest, SearchResponse, SearchTrace, TraceEvent
from .planner import ConsensusPlanner
from .deepseek import DeepSeekClient, DeepSeekConfig, DeepSeekError
from .movie_data import JsonlMovieCatalog
from .game_agents import ContractualDeepSeekReviewAgent

# DeepSeek 输出上限不适合一次评审全部召回池。先用可复现的检索分数
# 生成短名单，再让所有启用的 Agent 评审同一批候选，避免不完整评审者
# 和完整评审者进入同一排名。
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
    def from_environment(cls, catalog=None) -> "ConsensusWorkflow":
        """Use one DeepSeek key when present; remain runnable without it."""
        configured_catalog = os.environ.get("MOVIE_CATALOG_PATH", "").strip()
        default_catalog = Path(__file__).resolve().parents[2] / "experiments/data/processed/movie_catalog.jsonl"
        catalog = catalog or (JsonlMovieCatalog(configured_catalog or default_catalog)
                              if configured_catalog or default_catalog.is_file()
                              else InMemoryCatalog())
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
        retrieved_candidates = self.catalog.search(request)
        excluded_movie_ids = set(map(str, request.excluded_movie_ids))
        if excluded_movie_ids:
            retrieved_candidates = [
                candidate for candidate in retrieved_candidates
                if candidate.movie_id not in excluded_movie_ids
            ]
        trace.events.append(TraceEvent("retrieval", "candidates_retrieved", {
            "count": len(retrieved_candidates),
            "candidate_ids": [item.movie_id for item in retrieved_candidates],
            "excluded_movie_ids": sorted(excluded_movie_ids),
        }))

        # Fairness invariant: every candidate entering consensus receives the same
        # enabled-agent coverage.  The catalog already applies hard request filters.
        shortlist_limit = max(request.top_k, LLM_REVIEW_LIMIT)
        candidates = retrieved_candidates[:shortlist_limit]
        trace.events.append(TraceEvent("retrieval", "shortlist_frozen", {
            "count": len(candidates), "limit": shortlist_limit,
            "candidate_ids": [item.movie_id for item in candidates],
            "policy": "retrieval_rank_before_agent_review",
        }))

        reports = []
        selected_roles = set(plan.agent_roles)
        if any(agent.role in {"diversity", "critic"} for agent in self.agents):
            selected_roles.update({"diversity", "critic"})
        selected_agents = [agent for agent in self.agents if agent.role in selected_roles]
        completed = {}
        with ThreadPoolExecutor(max_workers=max(1, len(selected_agents))) as executor:
            futures = {executor.submit(agent.review, request, candidates): agent
                       for agent in selected_agents}
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    completed[agent.agent_id] = future.result()
                except DeepSeekError as error:
                    trace.events.append(TraceEvent("deliberation", "agent_degraded", {
                        "agent_id": agent.agent_id, "role": agent.role,
                        "reason": str(error), "fallback": "deterministic_agents_continue",
                    }))
        # Restore configured agent order so traces and aggregation remain reproducible.
        for agent in selected_agents:
            if agent.agent_id not in completed:
                continue
            try:
                agent_reports = completed[agent.agent_id]
            except KeyError:
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

        coverage = {
            candidate.movie_id: sorted({report.agent_id for report in reports
                                        if report.candidate_id == candidate.movie_id})
            for candidate in candidates
        }
        trace.events.append(TraceEvent("deliberation", "coverage_checked", {
            "required_agents": sorted(agent.agent_id for agent in selected_agents),
            "coverage": coverage,
            "uniform": len({tuple(value) for value in coverage.values()}) <= 1,
            "complete": all(
                set(value) == {agent.agent_id for agent in selected_agents}
                for value in coverage.values()
            ),
        }))

        decisions = self.engine.decide(candidates, reports, request.top_k)
        trace.reputations_after = self.engine.snapshot()
        trace.events.append(TraceEvent("consensus", "ranking_completed", {
            "selected": [item.candidate.movie_id for item in decisions],
            "scores": {item.candidate.movie_id: item.score for item in decisions},
        }))
        return SearchResponse(request_id, decisions, trace)
