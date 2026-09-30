"""Public-contract follower wrapper for movie-domain DeepSeek agents."""
from __future__ import annotations

import json
from typing import Iterable, List

from experiments.core.mechanism.contracts import CONTRACTS, parse_action, render_contract, scale_rewards

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
        token_budget = {
            "low_effort": 700,
            "aeic_effort": 2400,
            "verify": 3200,
            "defer_imitate": 900,
        }[action.value]
        reports = super().review(
            request, candidate_list, action=action.value, max_tokens=token_budget,
        )
        if action.value == "verify" and reports:
            audit = self.client.complete_json(
                "你是独立核验器。只能依据给定请求、候选与报告检查约束冲突和无证据推断。输出 JSON：{\"checks\":[{\"candidate_id\":\"...\",\"passed\":true,\"issue\":\"...\"}]}。",
                json.dumps({
                    "request": request.__dict__,
                    "candidates": [{"movie_id": x.movie_id, "title": x.title,
                                    "year": x.year, "genres": x.genres} for x in candidate_list],
                    "reports": [{"candidate_id": x.candidate_id,
                                 "inference": x.inference,
                                 "verdict": x.verdict.value} for x in reports],
                }, ensure_ascii=False),
                temperature=0.0, max_tokens=900,
            )
            usage = audit.get("_usage") or {}
            extra_tokens = int(usage.get("total_tokens") or 0)
            checks = {str(x.get("candidate_id")): x for x in audit.get("checks", [])
                      if isinstance(x, dict)}
            for report in reports:
                check = checks.get(report.candidate_id, {})
                passed = bool(check.get("passed", False))
                issue = str(check.get("issue", "核验器未返回明确说明"))
                report.inference += f" 核验结果：{'通过' if passed else '保留'}；{issue}"
                report.token_cost += extra_tokens
                if not passed:
                    report.confidence *= 0.75
        return reports
