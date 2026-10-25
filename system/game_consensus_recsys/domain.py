"""Typed domain records shared by retrieval, agents, consensus, and tracing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


SCHEMA_VERSION = "game-consensus-recsys.v1"

GENRE_ALIASES = {
    "科幻": "sci-fi", "剧情": "drama", "动作": "action", "灾难": "disaster",
    "动画": "animation", "爱情": "romance", "冒险": "adventure", "悬疑": "mystery",
    "喜剧": "comedy", "历史": "history", "惊悚": "thriller", "犯罪": "crime",
    "纪录片": "documentary", "儿童": "children", "奇幻": "fantasy", "战争": "war",
}


def normalize_genres(values: Iterable[str]) -> set[str]:
    return {GENRE_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
            for value in values if str(value).strip()}


class CandidateSource(str, Enum):
    LEXICAL = "lexical"
    GENERATIVE = "generative"
    PREFERENCE = "preference"


class Verdict(str, Enum):
    SUPPORT = "support"
    OPPOSE = "oppose"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Evidence:
    claim: str
    source: str
    reference: str
    verified: bool = False
    strength: float = 0.5

    def validate(self) -> List[str]:
        errors = []
        if not self.claim.strip():
            errors.append("evidence claim is required")
        if not self.source.strip() or not self.reference.strip():
            errors.append("evidence source and reference are required")
        if not 0 <= self.strength <= 1:
            errors.append("evidence strength must be in [0, 1]")
        return errors


@dataclass
class MovieCandidate:
    movie_id: str
    title: str
    year: int
    genres: List[str]
    sources: List[CandidateSource]
    retrieval_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors = []
        if not self.movie_id or not self.title:
            errors.append("candidate movie_id and title are required")
        if not self.sources:
            errors.append("candidate requires at least one source")
        if not 0 <= self.retrieval_score <= 1:
            errors.append("retrieval_score must be in [0, 1]")
        return errors


@dataclass
class CandidateReport:
    agent_id: str
    role: str
    candidate_id: str
    assumption: str
    evidence: List[Evidence]
    inference: str
    verdict: Verdict
    utility: float
    confidence: float
    action: str = "aeic_effort"
    token_cost: int = 0
    latency_ms: int = 0

    def validate(self) -> List[str]:
        errors = []
        for name in ("agent_id", "role", "candidate_id", "inference"):
            if not str(getattr(self, name)).strip():
                errors.append(f"report {name} is required")
        if not 0 <= self.utility <= 1 or not 0 <= self.confidence <= 1:
            errors.append("report utility and confidence must be in [0, 1]")
        for index, item in enumerate(self.evidence):
            errors.extend(f"evidence[{index}]: {error}" for error in item.validate())
        if self.token_cost < 0 or self.latency_ms < 0:
            errors.append("report costs cannot be negative")
        return errors


@dataclass(frozen=True)
class SearchRequest:
    query: str
    top_k: int = 5
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    preferred_genres: List[str] = field(default_factory=list)
    required_genres: List[str] = field(default_factory=list)
    excluded_genres: List[str] = field(default_factory=list)
    excluded_movie_ids: List[str] = field(default_factory=list)
    liked_movie_ids: List[str] = field(default_factory=list)
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    latency_budget_ms: int = 2000

    def validate(self) -> List[str]:
        errors = []
        if not self.query.strip():
            errors.append("query is required")
        if not 1 <= self.top_k <= 20:
            errors.append("top_k must be between 1 and 20")
        if self.latency_budget_ms <= 0:
            errors.append("latency_budget_ms must be positive")
        if self.min_year and self.max_year and self.min_year > self.max_year:
            errors.append("min_year cannot exceed max_year")
        return errors


@dataclass(frozen=True)
class ExecutionPlan:
    risk_level: str
    complexity_score: float
    agent_roles: List[str]
    topology: str
    verification_budget: int
    rationale: List[str]


@dataclass
class ConsensusDecision:
    candidate: MovieCandidate
    score: float
    decision_consensus: float
    evidence_consensus: float
    residual_dissent: List[str]
    accepted_evidence: List[Evidence]
    contributing_agents: List[str]
    score_components: Dict[str, float] = field(default_factory=dict)
    agent_coverage: Dict[str, str] = field(default_factory=dict)


@dataclass
class TraceEvent:
    stage: str
    event: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchTrace:
    request_id: str
    mechanism: str
    topology: str
    plan: Optional[ExecutionPlan] = None
    events: List[TraceEvent] = field(default_factory=list)
    reports: List[CandidateReport] = field(default_factory=list)
    reputations_before: Dict[str, float] = field(default_factory=dict)
    reputations_after: Dict[str, float] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass
class SearchResponse:
    request_id: str
    decisions: List[ConsensusDecision]
    trace: SearchTrace

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
