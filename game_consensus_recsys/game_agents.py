"""Public-contract follower wrapper for movie-domain DeepSeek agents."""
from __future__ import annotations

import json
from typing import Iterable, List

from core.mechanism.contracts import CONTRACTS, parse_action, render_contract, scale_rewards

from .deepseek import DeepSeekClient, DeepSeekError
from .deepseek_agents import DeepSeekReviewAgent
from .domain import CandidateReport, MovieCandidate, SearchRequest


class ContractualDeepSeekReviewAgent(DeepSeekReviewAgent):
    """Choose an auditable action before producing the candidate reports."""

    def __init__(self, client: DeepSeekClient, role: str, agent_id: str,
                 mechanism: str = "static_aeic", reward_scale: float = 0.2):
        super().__init__(client, role, agent_id)
        self.spec = scale_rewards(CONTRACTS[mechanism], reward_scale)

    def review(self, request: SearchRequest,
               candidates: Iterable[MovieCandidate]) -> List[CandidateReport]:
        candidate_list = list(candidates)
        choice = self.client.complete_json(
            "你是资源受限且自主决策的影视推荐 Agent。先根据公开规则选择行动，不要先求解推荐。输出 JSON。",
            render_contract(self.spec) + "\n" + json.dumps({
                "role": self.role, "query": request.query,
                "candidate_count": len(candidate_list), "latency_budget_ms": request.latency_budget_ms,
                "output_schema": {"ACTION": "low_effort|aeic_effort|verify|defer_imitate"},
            }, ensure_ascii=False),
            temperature=0.1,
        )
        try:
            action = parse_action(json.dumps(choice, ensure_ascii=False))
        except ValueError as error:
            raise DeepSeekError(f"invalid follower action: {error}") from error
        reports = super().review(request, candidate_list)
        for report in reports:
            report.action = action.value
        return reports
