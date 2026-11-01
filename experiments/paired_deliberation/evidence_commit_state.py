"""Offline-tested candidate: commit observations independently of repair proposals.

Not wired into frozen atomic-v3 experiments. This is execution-layer behavior;
a single controller can use the same interface. It is not a multi-agent gain claim.
"""
import copy

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.contracts_v6 import check_patch, check_report
from experiments.paired_deliberation.pipeline_v5 import source_map
from experiments.paired_deliberation.state_v6 import StateV6, FACT_TYPES, ReplacementPreconditionError
from experiments.paired_deliberation.verified_state import EvidenceVerifier


class EvidenceCommitState(StateV6):
    def _target(self, reviewer, target, expected_hash):
        if reviewer not in self.owners and reviewer != "central": raise ValueError("unknown reviewer")
        if target not in self.claims: raise ValueError("unknown target")
        row = self.claims[target]
        if expected_hash != fingerprint(row): raise ValueError("stale claim revision")
        return row

    def _check(self, claim, record):
        if not isinstance(record, dict) or record.get("verdict") not in {"supported", "refuted", "insufficient"}:
            raise ValueError("invalid observation record")
        if "original_claim" in record and fingerprint(record["original_claim"]) != fingerprint(claim):
            raise ValueError("observation is for a different claim")
        if record.get("kind") == "exact_catalog_fact":
            exact = EvidenceVerifier(source_map(self.evidence)).verify(claim)
            if (exact["kind"] != "exact_catalog_fact" or exact["verdict"] != record["verdict"]
                    or exact["input_hash"] != record.get("input_hash")):
                raise ValueError("catalog observation provenance mismatch")
        if record.get("verification_failed"):
            return "insufficient", "invalid_proxy"
        if record["verdict"] == "insufficient":
            return "insufficient", "unresolved"
        if record.get("kind") not in {"exact_catalog_fact", "atomic_source_proxy"}:
            raise ValueError("decisive observation needs a registered verifier kind")
        if record["kind"] == "atomic_source_proxy" and "original_claim" not in record:
            raise ValueError("atomic observation must bind its original claim")
        return record["verdict"], "catalog_exact" if record["kind"] == "exact_catalog_fact" else "model_proxy"

    def observe(self, reviewer, target, expected_hash, verifier):
        current = self._target(reviewer, target, expected_hash)
        if current["claim"]["claim_type"] not in FACT_TYPES:
            raise ValueError("hypotheses cannot receive factual certification")
        record = verifier.verify(copy.deepcopy(current["claim"]))
        verdict, trust = self._check(current["claim"], record)
        context_hash, claim_hash = fingerprint(self.evidence), fingerprint(current["claim"])
        if (current.get("observed_context_hash") == context_hash and current.get("observed_claim_hash") == claim_hash
                and current["verification"] == record):
            return {"action": "observation_reused", "target": target}
        before, updated = copy.deepcopy(current), copy.deepcopy(self.claims)
        row = updated[target]
        row.update(own_status={"supported": "supported", "refuted": "retracted", "insufficient": "unresolved"}[verdict],
                   verification=copy.deepcopy(record), evidence_trust=trust, observed_verdict=verdict,
                   observed_context_hash=context_hash, observed_claim_hash=claim_hash, revision=row["revision"] + 1)
        self._refresh(updated)
        self.claims = updated
        event = {"action": "observe_" + {"supported": "keep", "refuted": "retract", "insufficient": "defer"}[verdict],
                 "reviewer": reviewer, "target": target, "before": before, "after": copy.deepcopy(updated[target]),
                 "verification": copy.deepcopy(record), "applied_verdict": verdict}
        self.events.append(event)
        return copy.deepcopy(event)

    def propose(self, reviewer, target, proposal, verifier):
        if not isinstance(proposal, dict): raise ValueError("patch must be an object")
        current = self._target(reviewer, target, proposal.get("expected_hash"))
        check_patch(proposal, target, fingerprint(current))
        if (current.get("observed_context_hash") != fingerprint(self.evidence)
                or current.get("observed_claim_hash") != fingerprint(current["claim"])):
            raise ValueError("commit a matching observation before proposing repairs")
        replacement = proposal["replacement"]
        if replacement is None:
            event = {"action": "acknowledge", "target": target, "reviewer": reviewer,
                     "observed_verdict": current["observed_verdict"], "state_changed": False}
            self.events.append(event)
            return copy.deepcopy(event)
        if current["own_status"] != "retracted" or current["observed_verdict"] != "refuted":
            raise ReplacementPreconditionError("replacement requires committed old refutation")
        check_report({"ranking": [r["movie_id"] for r in self.evidence["candidates_in_shared_retrieval_order"][:5]],
                      "claims": [replacement], "updates": []}, self.evidence)
        for key in ("claim_id", "candidate_ids", "claim_type", "depends_on"):
            if replacement[key] != current["claim"][key]: raise ValueError("replacement changes scope: " + key)
        record = verifier.verify(copy.deepcopy(replacement))
        verdict, trust = self._check(replacement, record)
        if verdict != "supported":
            raise ReplacementPreconditionError("replacement requires independently supported new claim")
        # A rejected repair never rolls back the preceding observation.
        before, updated = copy.deepcopy(current), copy.deepcopy(self.claims)
        row = updated[target]
        row.update(claim=copy.deepcopy(replacement), own_status="supported", verification=copy.deepcopy(record),
                   evidence_trust=trust, observed_verdict="supported", observed_claim_hash=fingerprint(replacement),
                   revision=row["revision"] + 1)
        self._refresh(updated)
        self.claims = updated
        event = {"action": "replace", "target": target, "reviewer": reviewer,
                 "before": before, "after": copy.deepcopy(updated[target])}
        self.events.append(event)
        return copy.deepcopy(event)

    def apply(self, *args, **kwargs):
        raise ValueError("legacy coupled transition disabled; use observe then propose")

    def replace_owner(self, *args, **kwargs):
        raise ValueError("unconstrained report rewriting is a control condition, not a committed-evidence operation")
