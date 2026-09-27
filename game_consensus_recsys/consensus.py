"""Evidence-aware consensus ranking with longitudinal agent reputation."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .domain import CandidateReport, ConsensusDecision, MovieCandidate, Verdict


class ConsensusEngine:
    """Transparent first mechanism; later policies can implement the same API."""

    def __init__(self, reputation_decay: float = 0.9):
        self.reputation: Dict[str, float] = defaultdict(float)
        self.decay = reputation_decay

    def snapshot(self) -> Dict[str, float]:
        return {key: round(value, 6) for key, value in sorted(self.reputation.items())}

    def decide(
        self,
        candidates: Iterable[MovieCandidate],
        reports: Iterable[CandidateReport],
        top_k: int,
    ) -> List[ConsensusDecision]:
        grouped: Dict[str, List[CandidateReport]] = defaultdict(list)
        for report in reports:
            grouped[report.candidate_id].append(report)

        decisions = []
        for candidate in candidates:
            items = grouped[candidate.movie_id]
            support, oppose, total = 0.0, 0.0, 0.0
            evidence = []
            dissent = []
            agents = []
            evidence_weight = 0.0
            verified_weight = 0.0
            for report in items:
                reputation_weight = max(0.25, 1.0 + self.reputation[report.agent_id] / 10.0)
                weight = report.confidence * reputation_weight
                total += weight
                agents.append(report.agent_id)
                if report.verdict == Verdict.SUPPORT:
                    support += weight * report.utility
                elif report.verdict == Verdict.OPPOSE:
                    oppose += weight * max(0.25, 1.0 - report.utility)
                    dissent.append(f"{report.agent_id}: {report.inference}")
                for item in report.evidence:
                    evidence.append(item)
                    evidence_weight += item.strength
                    verified_weight += item.strength * float(item.verified)

                verified = bool(report.evidence) and all(item.verified for item in report.evidence)
                contribution = (1.0 if report.verdict == Verdict.SUPPORT else -0.5 if report.verdict == Verdict.OPPOSE else 0.0)
                contribution *= float(verified) * report.confidence
                self.reputation[report.agent_id] = self.decay * self.reputation[report.agent_id] + contribution

            decision_consensus = support / max(total, 1e-9)
            evidence_consensus = verified_weight / max(evidence_weight, 1e-9)
            score = 0.35 * candidate.retrieval_score + 0.45 * decision_consensus + 0.20 * evidence_consensus - 0.20 * (oppose / max(total, 1e-9))
            decisions.append(ConsensusDecision(
                candidate=candidate,
                score=round(max(0.0, min(1.0, score)), 6),
                decision_consensus=round(decision_consensus, 6),
                evidence_consensus=round(evidence_consensus, 6),
                residual_dissent=dissent,
                accepted_evidence=[item for item in evidence if item.verified],
                contributing_agents=sorted(set(agents)),
            ))
        return sorted(decisions, key=lambda item: (-item.score, item.candidate.movie_id))[:top_k]

