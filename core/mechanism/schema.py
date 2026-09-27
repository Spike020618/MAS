"""Versioned records for causal follower-response experiments.

The old online pipeline retained final nodes but discarded the action offered to
the follower and most intermediate reports.  These records make those fields
mandatory before a record can be marked complete.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "aeic-md.v1"


class AgentAction(str, Enum):
    LOW_EFFORT = "low_effort"
    AEIC_EFFORT = "aeic_effort"
    VERIFY = "verify"
    DEFER_IMITATE = "defer_imitate"


@dataclass(frozen=True)
class AEICReport:
    A: str
    E: List[str]
    I: str
    C: str

    def validate(self) -> List[str]:
        errors: List[str] = []
        for name in ("A", "I", "C"):
            if not isinstance(getattr(self, name), str):
                errors.append(f"AEIC.{name} must be a string")
        if not isinstance(self.E, list) or not all(isinstance(x, str) for x in self.E):
            errors.append("AEIC.E must be a list of strings")
        return errors


@dataclass(frozen=True)
class MechanismOffer:
    mechanism: str
    credit_rule: str
    visible_reward: Dict[str, float]
    visible_cost: Dict[str, float]
    information_policy: str
    verification_budget: int = 0
    prompt_hash: str = ""
    config_hash: str = ""

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.mechanism:
            errors.append("mechanism is required")
        if not self.credit_rule:
            errors.append("credit_rule is required")
        if not self.prompt_hash:
            errors.append("prompt_hash is required")
        if not self.config_hash:
            errors.append("config_hash is required")
        if self.verification_budget < 0:
            errors.append("verification_budget cannot be negative")
        return errors


@dataclass
class AgentRound:
    round_id: int
    agent_id: int
    offered: MechanismOffer
    chosen_action: AgentAction
    report: AEICReport
    visible_peer_ids: List[int] = field(default_factory=list)
    visible_peer_reports: List[Dict[str, Any]] = field(default_factory=list)
    stance: Optional[str] = None
    confidence: Optional[float] = None
    reward: Optional[float] = None
    realized_cost: Optional[float] = None
    reputation_before: Optional[float] = None
    reputation_after: Optional[float] = None
    reward_components: Dict[str, float] = field(default_factory=dict)
    token_input: Optional[int] = None
    token_output: Optional[int] = None
    latency_s: Optional[float] = None
    raw_response: Optional[str] = None
    action_trace: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> List[str]:
        errors = self.offered.validate() + self.report.validate()
        if self.round_id < 1:
            errors.append("round_id must be >= 1")
        if self.agent_id < 0:
            errors.append("agent_id must be >= 0")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            errors.append("confidence must be in [0, 1]")
        for name in ("token_input", "token_output"):
            value = getattr(self, name)
            if value is not None and value < 0:
                errors.append(f"{name} cannot be negative")
        return errors


@dataclass
class ExperimentRecord:
    experiment_id: str
    dataset: str
    question_id: str
    condition: str
    seed: int
    model_assignments: Dict[str, str]
    rounds: List[AgentRound]
    ground_truth: Optional[str] = None
    final_decision: Optional[str] = None
    completed: bool = False
    error: Optional[str] = None
    attempt_count: int = 1
    prior_errors: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def validate(self, require_complete: bool = False) -> List[str]:
        errors: List[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(f"unsupported schema_version: {self.schema_version}")
        if self.attempt_count < 1:
            errors.append("attempt_count must be >= 1")
        for name in ("experiment_id", "dataset", "question_id", "condition"):
            if not getattr(self, name):
                errors.append(f"{name} is required")
        if not self.rounds and not self.error:
            errors.append("rounds cannot be empty for a non-error record")
        for i, item in enumerate(self.rounds):
            errors.extend(f"rounds[{i}]: {e}" for e in item.validate())
        if require_complete or self.completed:
            if self.error:
                errors.append("completed record cannot contain error")
            if not self.final_decision:
                errors.append("completed record requires final_decision")
            for i, item in enumerate(self.rounds):
                if item.token_input is None or item.token_output is None:
                    errors.append(f"rounds[{i}] requires token usage")
                if item.latency_s is None:
                    errors.append(f"rounds[{i}] requires latency_s")
                if item.reward is None:
                    errors.append(f"rounds[{i}] requires reward")
                if item.realized_cost is None:
                    errors.append(f"rounds[{i}] requires realized_cost")
                if item.reputation_before is None or item.reputation_after is None:
                    errors.append(f"rounds[{i}] requires reputation snapshots")
                if item.raw_response is None:
                    errors.append(f"rounds[{i}] requires raw_response")
                if not item.action_trace:
                    errors.append(f"rounds[{i}] requires action_trace")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def record_from_dict(item: Dict[str, Any]) -> ExperimentRecord:
    """Rehydrate a persisted record so frozen inputs receive schema validation."""
    rounds = []
    for raw in item.get("rounds") or []:
        offered = MechanismOffer(**raw["offered"])
        report = AEICReport(**raw["report"])
        values = dict(raw)
        values["offered"] = offered
        values["report"] = report
        values["chosen_action"] = AgentAction(raw["chosen_action"])
        rounds.append(AgentRound(**values))
    values = dict(item)
    values["rounds"] = rounds
    return ExperimentRecord(**values)
