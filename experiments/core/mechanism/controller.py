"""Reward settlement and longitudinal reputation for AEIC-MD."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Callable, Dict, Iterable, Optional

from experiments.core.trust import extract_stance

from .behavioral_metrics import evidence_novelty, unsupported_flip
from .schema import AEICReport, AgentAction, MechanismOffer
from .contracts import ContractSpec


def aeic_completeness(report: AEICReport) -> float:
    present = [bool(report.A.strip()), bool(report.E), bool(report.I.strip()), bool(report.C.strip())]
    return sum(present) / 4.0


class MechanismController:
    def __init__(self, n_agents: int, reputation_decay: float = 0.9,
                 similarity_fn: Optional[Callable[[str, str], float]] = None,
                 flip_threshold: float = 0.25):
        self.reputation = defaultdict(float, {i: 0.0 for i in range(n_agents)})
        self.decay = reputation_decay
        self.similarity_fn = similarity_fn
        self.flip_threshold = flip_threshold

    def novelty(self, report: AEICReport, baseline: Iterable[str]) -> float:
        prior = [str(x) for x in baseline if str(x).strip()]
        if not report.E:
            return 0.0
        if not prior:
            return 1.0
        if self.similarity_fn is None:
            return evidence_novelty(report, prior)
        scores = []
        for evidence in report.E:
            similarity = max(self.similarity_fn(evidence, old) for old in prior)
            scores.append(1.0 - max(0.0, min(1.0, similarity)))
        return sum(scores) / len(scores)

    def reputation_snapshot(self) -> Dict[str, float]:
        return {str(k): round(v, 6) for k, v in self.reputation.items()}

    def adaptive_spec(self, base: ContractSpec, agent_id: int) -> ContractSpec:
        """Publish a history-dependent contract before the follower acts.

        Low-reputation agents receive a verification subsidy and a stronger
        unsupported-flip penalty. High-reputation agents receive a larger reward
        for novel evidence. All changes are visible in the rendered contract.
        """
        if base.name != "adaptive_stackelberg":
            return base
        reputation = self.reputation[agent_id]
        low_trust = max(0.0, min(1.0, -reputation / 5.0))
        high_trust = max(0.0, min(1.0, reputation / 5.0))
        rewards = dict(base.rewards)
        costs = dict(base.costs)
        rewards["novel_evidence"] *= 1.0 + 0.25 * high_trust
        rewards["unsupported_flip"] *= 1.0 + 0.50 * low_trust
        rewards["verified_evidence"] *= 1.0 + 0.25 * low_trust
        costs["verify"] = max(0.5, costs["verify"] * (1.0 - 0.50 * low_trust))
        return replace(base, rewards=rewards, costs=costs)

    def settle(
        self,
        agent_id: int,
        offer: MechanismOffer,
        action: AgentAction,
        report: AEICReport,
        ground_truth: Optional[str],
        visible_evidence: Iterable[str] = (),
        previous_report: Optional[AEICReport] = None,
        previous_stance: Optional[str] = None,
        new_external_evidence: Iterable[str] = (),
    ) -> Dict[str, float]:
        stance = extract_stance(report.C)
        cost = float(offer.visible_cost.get(action.value, 0.0))
        credit = 0.0
        components: Dict[str, float] = {}

        if offer.credit_rule == "final_outcome":
            components["correct"] = float(bool(ground_truth and stance == ground_truth))
            credit += offer.visible_reward.get("correct", 0.0) * components["correct"]
        elif offer.credit_rule == "fixed_process_quality":
            components["complete_aeic"] = aeic_completeness(report)
            components["consistency"] = float(stance != "neutral")
            credit += offer.visible_reward.get("complete_aeic", 0.0) * components["complete_aeic"]
            credit += offer.visible_reward.get("consistency", 0.0) * components["consistency"]
        elif offer.credit_rule == "fixed_process_quality_components":
            components.update({
                "field_A": float(bool(report.A.strip())),
                "field_E": float(bool(report.E)),
                "field_I": float(bool(report.I.strip())),
                "field_C": float(bool(report.C.strip())),
                "consistency": float(stance != "neutral"),
            })
            for name, value in components.items():
                credit += offer.visible_reward.get(name, 0.0) * value
        elif offer.credit_rule in (
            "novel_verifiable_marginal_contribution", "history_adaptive_contract"
        ):
            novelty_baseline = list(visible_evidence)
            if previous_report is not None:
                novelty_baseline.extend(previous_report.E)
            components["novel_evidence"] = self.novelty(report, novelty_baseline)
            # Ground truth is used only by the settlement environment, never shown as a feature.
            components["verified_evidence"] = float(bool(ground_truth and stance == ground_truth))
            components["useful_correction"] = float(
                bool(previous_stance and stance != previous_stance and stance == ground_truth)
            )
            components["unsupported_flip"] = float(
                bool(previous_report and unsupported_flip(
                    previous_report, report, previous_stance, stance,
                    list(new_external_evidence), self.flip_threshold,
                ))
            )
            for name, value in components.items():
                credit += offer.visible_reward.get(name, 0.0) * value
            if offer.credit_rule == "history_adaptive_contract":
                credit *= 1.0 + max(-0.25, min(0.25, self.reputation[agent_id] / 20.0))

        utility = credit - cost
        self.reputation[agent_id] = self.decay * self.reputation[agent_id] + utility
        return {
            "credit": round(credit, 6),
            "cost": round(cost, 6),
            "utility": round(utility, 6),
            "reputation_after": round(self.reputation[agent_id], 6),
            **{f"component_{k}": round(v, 6) for k, v in components.items()},
        }
