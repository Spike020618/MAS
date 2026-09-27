"""Visible mechanism contracts used before a follower chooses an action."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict

from .schema import AgentAction, MechanismOffer


ACTION_DESCRIPTIONS = {
    AgentAction.LOW_EFFORT: "Give a short answer with minimal supporting work.",
    AgentAction.AEIC_EFFORT: "Produce a complete AEIC report from independent reasoning.",
    AgentAction.VERIFY: "Perform an additional verification or counterfactual check, then report AEIC.",
    AgentAction.DEFER_IMITATE: "Use visible peer information before reporting; add little independent work.",
}


@dataclass(frozen=True)
class ContractSpec:
    name: str
    credit_rule: str
    rewards: Dict[str, float]
    costs: Dict[str, float]
    information_policy: str
    verification_budget: int = 0

    def config_hash(self) -> str:
        raw = json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_offer(self, prompt: str) -> MechanismOffer:
        return MechanismOffer(
            mechanism=self.name,
            credit_rule=self.credit_rule,
            visible_reward=dict(self.rewards),
            visible_cost=dict(self.costs),
            information_policy=self.information_policy,
            verification_budget=self.verification_budget,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            config_hash=self.config_hash(),
        )


CONTRACTS = {
    "no_mechanism": ContractSpec(
        "no_mechanism", "none", {},
        {"low_effort": 0.0, "aeic_effort": 1.0, "verify": 2.0, "defer_imitate": 0.2},
        "condition_dependent",
    ),
    "outcome_only": ContractSpec(
        "outcome_only", "final_outcome", {"correct": 3.0},
        {"low_effort": 0.0, "aeic_effort": 1.0, "verify": 2.0, "defer_imitate": 0.2},
        "condition_dependent",
    ),
    "static_aeic": ContractSpec(
        "static_aeic", "fixed_process_quality", {"complete_aeic": 1.0, "consistency": 1.0},
        {"low_effort": 0.0, "aeic_effort": 1.0, "verify": 2.0, "defer_imitate": 0.2},
        "condition_dependent",
    ),
    "aeic_contract": ContractSpec(
        "aeic_contract", "novel_verifiable_marginal_contribution",
        {"novel_evidence": 1.5, "verified_evidence": 2.0, "useful_correction": 2.0, "unsupported_flip": -1.5},
        {"low_effort": 0.0, "aeic_effort": 1.0, "verify": 2.0, "defer_imitate": 0.2},
        "condition_dependent", verification_budget=1,
    ),
    "adaptive_stackelberg": ContractSpec(
        "adaptive_stackelberg", "history_adaptive_contract",
        {"novel_evidence": 1.5, "verified_evidence": 2.0, "useful_correction": 2.0, "unsupported_flip": -1.5},
        {"low_effort": 0.0, "aeic_effort": 1.0, "verify": 2.0, "defer_imitate": 0.2},
        "adaptive", verification_budget=1,
    ),
}


def render_contract(spec: ContractSpec) -> str:
    action_lines = []
    for action in AgentAction:
        cost = spec.costs.get(action.value, 0.0)
        action_lines.append(f"- {action.value}: {ACTION_DESCRIPTIONS[action]} Cost={cost:.2f}")
    reward_lines = [f"- {key}: {value:+.2f}" for key, value in sorted(spec.rewards.items())]
    rewards = "\n".join(reward_lines) if reward_lines else "- No explicit credit is offered."
    public_name = "aeic_contract" if spec.name == "adaptive_stackelberg" else spec.name
    public_rule = ("novel_verifiable_marginal_contribution"
                   if spec.credit_rule == "history_adaptive_contract"
                   else spec.credit_rule)
    return (
        "[VISIBLE MECHANISM CONTRACT]\n"
        f"Mechanism: {public_name}\nCredit rule: {public_rule}\n"
        "Choose exactly one action before solving:\n"
        + "\n".join(action_lines)
        + "\nPotential credit:\n" + rewards
        + '\nReturn one JSON object whose first field is "ACTION" with the chosen action name.'
    )


def scale_rewards(spec: ContractSpec, scale: float) -> ContractSpec:
    if scale <= 0:
        raise ValueError("reward scale must be positive")
    return ContractSpec(
        name=spec.name,
        credit_rule=spec.credit_rule,
        rewards={key: value * scale for key, value in spec.rewards.items()},
        costs=dict(spec.costs),
        information_policy=spec.information_policy,
        verification_budget=spec.verification_budget,
    )


def scale_costs(spec: ContractSpec, scale: float) -> ContractSpec:
    """Return a contract with only follower action costs rescaled."""
    if scale <= 0:
        raise ValueError("cost scale must be positive")
    return ContractSpec(
        name=spec.name,
        credit_rule=spec.credit_rule,
        rewards=dict(spec.rewards),
        costs={key: value * scale for key, value in spec.costs.items()},
        information_policy=spec.information_policy,
        verification_budget=spec.verification_budget,
    )


def apply_signal_mask(spec: ContractSpec, mask: str) -> ContractSpec:
    """Expose a component-wise StaticAEIC contract for local-signal ablations."""
    normalized = "".join(x for x in "AEIC" if x in set((mask or "").upper()))
    if not normalized:
        raise ValueError("signal mask must contain at least one of A, E, I, C")
    if spec.credit_rule != "fixed_process_quality":
        raise ValueError("signal masks are defined only for StaticAEIC")
    rewards = {f"field_{name}": (0.25 if name in normalized else 0.0)
               for name in "AEIC"}
    rewards["consistency"] = spec.rewards.get("consistency", 0.0)
    return ContractSpec(
        name=spec.name,
        credit_rule="fixed_process_quality_components",
        rewards=rewards,
        costs=dict(spec.costs),
        information_policy=spec.information_policy,
        verification_budget=spec.verification_budget,
    )


def parse_action(text: str) -> AgentAction:
    first = (text or "").strip().splitlines()[0].lower()
    for action in AgentAction:
        if action.value in first:
            return action
    try:
        start = text.index("{")
        data, _end = json.JSONDecoder().raw_decode(text[start:])
        value = str(data.get("ACTION", "")).lower()
        return AgentAction(value)
    except (ValueError, json.JSONDecodeError, KeyError):
        pass
    raise ValueError("response does not declare a valid ACTION on the first line")
