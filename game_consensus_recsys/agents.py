"""Deterministic domain agents for the first end-to-end system slice."""
from __future__ import annotations

from typing import Iterable, List, Protocol

from .catalog import InMemoryCatalog
from .domain import CandidateReport, Evidence, MovieCandidate, SearchRequest, Verdict


class CandidateAgent(Protocol):
    agent_id: str
    role: str

    def review(self, request: SearchRequest, candidates: Iterable[MovieCandidate]) -> List[CandidateReport]: ...


class RetrievalAgent:
    agent_id = "retrieval-1"
    role = "retrieval"

    def __init__(self, catalog: InMemoryCatalog):
        self.catalog = catalog

    def review(self, request: SearchRequest, candidates: Iterable[MovieCandidate]) -> List[CandidateReport]:
        reports = []
        for candidate in candidates:
            movie = self.catalog.get(candidate.movie_id)
            assert movie is not None
            reference = candidate.metadata.get("evidence_refs", {}).get(
                "metadata", f"movie:{candidate.movie_id}"
            )
            evidence = [
                Evidence(
                    claim=f"目录记录：{candidate.title}（{candidate.year}），类型为{'/'.join(candidate.genres)}",
                    source="catalog",
                    reference=reference,
                    verified=True,
                    strength=1.0,
                )
            ]
            reports.append(CandidateReport(
                self.agent_id, self.role, candidate.movie_id,
                "目录元数据是实体与属性判断的事实基础", evidence,
                "候选存在于目录中，且检索得分反映查询匹配度。", Verdict.SUPPORT,
                candidate.retrieval_score, 0.92, token_cost=0, latency_ms=2,
            ))
        return reports


class PreferenceAgent:
    agent_id = "preference-1"
    role = "preference"

    def review(self, request: SearchRequest, candidates: Iterable[MovieCandidate]) -> List[CandidateReport]:
        preferred = set(request.preferred_genres)
        reports = []
        for candidate in candidates:
            overlap = preferred.intersection(candidate.genres)
            if preferred:
                utility = len(overlap) / len(preferred)
                verdict = Verdict.SUPPORT if overlap else Verdict.OPPOSE
                claim = f"候选匹配用户偏好类型：{','.join(sorted(overlap))}" if overlap else "候选未匹配显式偏好类型"
            else:
                utility, verdict, claim = 0.5, Verdict.ABSTAIN, "请求没有提供长期类型偏好"
            reports.append(CandidateReport(
                self.agent_id, self.role, candidate.movie_id,
                "显式偏好优先于推测的隐式偏好",
                [Evidence(claim, "request_profile", "request:preferred_genres", True, 0.9)],
                "根据候选类型与用户偏好的交集估计个性化效用。", verdict,
                utility, 0.85 if preferred else 0.55, token_cost=0, latency_ms=1,
            ))
        return reports


class ConstraintVerifierAgent:
    agent_id = "verifier-1"
    role = "verifier"

    def review(self, request: SearchRequest, candidates: Iterable[MovieCandidate]) -> List[CandidateReport]:
        reports = []
        for candidate in candidates:
            violations = []
            if request.min_year and candidate.year < request.min_year:
                violations.append("早于最小年份")
            if request.max_year and candidate.year > request.max_year:
                violations.append("晚于最大年份")
            if set(request.excluded_genres).intersection(candidate.genres):
                violations.append("命中排除类型")
            valid = not violations
            reports.append(CandidateReport(
                self.agent_id, self.role, candidate.movie_id,
                "硬约束不应被偏好分数补偿",
                [Evidence(
                    "候选满足全部硬约束" if valid else "；".join(violations),
                    "constraint_checker", f"request:{candidate.movie_id}", True, 1.0,
                )],
                "逐项检查年份和排除类型。", Verdict.SUPPORT if valid else Verdict.OPPOSE,
                1.0 if valid else 0.0, 0.99, action="verify", token_cost=0, latency_ms=1,
            ))
        return reports
