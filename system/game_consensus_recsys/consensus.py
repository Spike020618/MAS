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
        candidate_list = list(candidates)
        grouped: Dict[str, List[CandidateReport]] = defaultdict(list)
        for report in reports:
            grouped[report.candidate_id].append(report)

        # All candidates in one request use the same reputation snapshot.  Updates
        # are settled only after every candidate has been scored, so input order
        # cannot change the result.
        reputation_before = dict(self.reputation)
        retrieval_values = [candidate.retrieval_score for candidate in candidate_list]
        retrieval_min = min(retrieval_values, default=0.0)
        retrieval_max = max(retrieval_values, default=0.0)

        def normalized_retrieval(value: float) -> float:
            spread = retrieval_max - retrieval_min
            return 0.5 if spread <= 1e-12 else (value - retrieval_min) / spread

        decisions = []
        reputation_contributions: Dict[str, List[float]] = defaultdict(list)
        for candidate in candidate_list:
            items = grouped[candidate.movie_id]
            verifier_reports = [item for item in items if item.role == "verifier"]
            if any(item.verdict == Verdict.OPPOSE for item in verifier_reports):
                continue

            support, oppose, total = 0.0, 0.0, 0.0
            evidence = []
            dissent = []
            agents = []
            evidence_weight = 0.0
            verified_weight = 0.0
            for report in items:
                reputation_weight = max(0.25, 1.0 + reputation_before.get(report.agent_id, 0.0) / 10.0)
                weight = report.confidence * reputation_weight
                agents.append(report.agent_id)
                # Retrieval is represented by the normalized retrieval component;
                # verifier is a gate.  Counting either again would double reward
                # catalogue rank or give every valid candidate the same bonus.
                if report.role not in {"retrieval", "verifier"}:
                    total += weight
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
                reputation_contributions[report.agent_id].append(contribution)

            # With no semantic report, remain neutral rather than treating missing
            # review as unanimous support.
            decision_consensus = (support - oppose) / total if total else 0.0
            decision_consensus = max(-1.0, min(1.0, decision_consensus))
            decision_quality = (decision_consensus + 1.0) / 2.0
            evidence_consensus = verified_weight / max(evidence_weight, 1e-9)
            retrieval_component = normalized_retrieval(candidate.retrieval_score)
            oppose_ratio = oppose / total if total else 0.0
            score = (0.45 * retrieval_component
                     + 0.40 * decision_quality
                     + 0.15 * evidence_consensus
                     - 0.10 * oppose_ratio)
            coverage = {
                report.agent_id: report.verdict.value for report in items
            }
            decisions.append(ConsensusDecision(
                candidate=candidate,
                score=round(max(0.0, min(1.0, score)), 6),
                decision_consensus=round(decision_consensus, 6),
                evidence_consensus=round(evidence_consensus, 6),
                residual_dissent=dissent,
                accepted_evidence=[item for item in evidence if item.verified],
                contributing_agents=sorted(set(agents)),
                score_components={
                    "retrieval": round(retrieval_component, 6),
                    "agent_utility": round(decision_quality, 6),
                    "evidence_quality": round(evidence_consensus, 6),
                    "opposition_penalty": round(oppose_ratio, 6),
                },
                agent_coverage=dict(sorted(coverage.items())),
            ))

        for agent_id, values in reputation_contributions.items():
            mean_contribution = sum(values) / len(values) if values else 0.0
            self.reputation[agent_id] = (
                self.decay * reputation_before.get(agent_id, 0.0) + mean_contribution
            )
        return sorted(decisions, key=lambda item: (-item.score, item.candidate.movie_id))[:top_k]
