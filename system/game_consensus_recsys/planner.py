"""Request-time-only planner for progressive consensus organization."""
from __future__ import annotations

from .domain import ExecutionPlan, SearchRequest


class ConsensusPlanner:
    """Selects a plan without ground truth, candidate labels, or post-hoc utility."""

    def plan(self, request: SearchRequest) -> ExecutionPlan:
        rationale = []
        score = 0.0
        hard_constraints = sum(value is not None for value in (request.min_year, request.max_year))
        hard_constraints += len(request.required_genres) + len(request.excluded_genres)
        if hard_constraints:
            score += min(0.35, 0.12 * hard_constraints)
            rationale.append(f"{hard_constraints} hard constraints require verification")
        if request.preferred_genres:
            score += min(0.25, 0.08 * len(request.preferred_genres))
            rationale.append("personal preference evidence is available")
        if len(request.query) >= 18:
            score += 0.25
            rationale.append("long query may contain compositional intent")
        if request.top_k >= 8:
            score += 0.15
            rationale.append("large result set increases coverage and diversity risk")
        if request.latency_budget_ms < 800:
            score = max(0.0, score - 0.15)
            rationale.append("tight latency budget limits deliberation")
        score = round(min(1.0, score), 3)

        if score < 0.2:
            return ExecutionPlan(
                "low", score, ["retrieval", "verifier"],
                "targeted_review", 1, rationale or ["simple request with no extra context"],
            )
        if score < 0.6:
            return ExecutionPlan(
                "medium", score, ["retrieval", "preference", "verifier"],
                "parallel_review_then_consensus", 2, rationale,
            )
        return ExecutionPlan(
            "high", score, ["retrieval", "preference", "verifier"],
            "parallel_review_with_escalation", 3, rationale,
        )
