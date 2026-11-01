"""ReDial execution contract and post-hoc semantic process evaluation."""
import copy
from collections import Counter


VERDICTS = {"supported", "unsupported", "unclear"}


class EvidenceContractChecker:
    """Validate executable updates without pretending to judge semantic truth."""

    def __init__(self, case_id, available_source_ids):
        self.case_id = case_id
        self.available_source_ids = set(available_source_ids)

    def __call__(self, agent, old_report, proposal, observation, state):
        del old_report, observation, state
        updates = proposal.get("updates") if isinstance(proposal, dict) else None
        if not isinstance(updates, list) or len(updates) != 1:
            raise ValueError("proposal must contain exactly one scoped update")
        update = updates[0]
        required = {"target_claim_id", "decision", "reason", "source_ids"}
        if not isinstance(update, dict) or set(update) != required:
            raise ValueError("invalid update schema")
        target = update["target_claim_id"]
        if not isinstance(target, str) or not target.startswith(agent + ":"):
            raise ValueError("agent may only execute an update to its own report")
        if update["decision"] not in {"accept", "reject", "unresolved"}:
            raise ValueError("invalid update decision")
        if not isinstance(update["reason"], str) or not update["reason"].strip():
            raise ValueError("update reason is required")
        refs = update["source_ids"]
        if not isinstance(refs, list) or not refs or not set(refs) <= self.available_source_ids:
            raise ValueError("update cites unavailable evidence")
        return {
            "decision": update["decision"],
            "target_claim_ids": [target],
            "evidence_refs": list(dict.fromkeys(refs)),
            "rationale": "structural evidence contract passed for " + self.case_id,
        }


class AnnotatedProcessEvaluator:
    """Evaluate completed events against blind annotations after execution."""

    def __init__(self, available_source_ids, annotations):
        self.available_source_ids = set(available_source_ids)
        self.annotations = copy.deepcopy(annotations)
        self._validate_annotations()

    def _validate_annotations(self):
        if not isinstance(self.annotations, dict):
            raise ValueError("annotations must be keyed by qualified claim ID")
        for target, row in self.annotations.items():
            if not isinstance(target, str) or ":" not in target or not isinstance(row, dict):
                raise ValueError("invalid annotation identity")
            if set(row) != {"verdict", "source_ids", "annotator_ids", "adjudication"}:
                raise ValueError("incomplete annotation record")
            if row["verdict"] not in VERDICTS:
                raise ValueError("invalid annotation verdict")
            if not isinstance(row["source_ids"], list) or not set(row["source_ids"]) <= self.available_source_ids:
                raise ValueError("annotation cites unavailable evidence")
            if not isinstance(row["annotator_ids"], list) or len(set(row["annotator_ids"])) < 2:
                raise ValueError("at least two independent annotators are required")
            if not isinstance(row["adjudication"], str) or not row["adjudication"].strip():
                raise ValueError("annotation adjudication is required")

    def evaluate(self, events):
        counts = Counter()
        rows = []
        for event in events:
            check = event.get("check")
            if not isinstance(check, dict) or len(check.get("target_claim_ids", [])) != 1:
                raise ValueError("event has no auditable scoped check")
            target = check["target_claim_ids"][0]
            if target not in self.annotations:
                raise ValueError("event target has no blind annotation")
            verdict = self.annotations[target]["verdict"]
            decision = event["decision"]
            if verdict == "unsupported" and decision == "reject":
                outcome = "incorrect_claim_corrected"
            elif verdict == "supported" and decision == "accept":
                outcome = "correct_claim_preserved"
            elif verdict == "supported" and decision == "reject":
                outcome = "correct_claim_incorrectly_changed"
            elif verdict == "unsupported" and decision == "accept":
                outcome = "incorrect_claim_preserved"
            else:
                outcome = "unclear_claim_left_" + decision
            counts[outcome] += 1
            rows.append({"target_claim_id": target, "verdict": verdict,
                         "decision": decision, "outcome": outcome})
        return {"counts": dict(counts), "events": rows,
                "annotations_used_during_execution": False}
