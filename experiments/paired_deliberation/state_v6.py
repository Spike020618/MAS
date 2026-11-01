"""Typed, dependency-aware state with transactional patches and raw report swaps."""
import copy

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.contracts_v6 import check_patch, check_report

FACT_TYPES = {"item_fact", "explicit_preference"}


class ReplacementPreconditionError(ValueError):
    """A well-formed proposal was correctly blocked by the semantic policy."""


class StateV6:
    def __init__(self, reports, evidence):
        self.evidence = copy.deepcopy(evidence)
        self.claims, self.events = {}, []
        self.owners = set(reports)
        for owner, report in reports.items():
            check_report(report, evidence)
            for claim in report["claims"]:
                self.claims[owner + ":" + claim["claim_id"]] = self._row(owner, claim)
        self._check_graph(self.claims)
        self._refresh(self.claims)

    @staticmethod
    def _row(owner, claim, revision=0):
        claim = StateV6._qualified_claim(owner, claim)
        own = "unverified" if claim["claim_type"] in FACT_TYPES else "hypothesis"
        return {"owner": owner, "claim": copy.deepcopy(claim), "own_status": own,
                "status": own, "revision": revision, "verification": None, "blocked_by": []}

    @staticmethod
    def _qualified_claim(owner, claim):
        claim = copy.deepcopy(claim)
        claim["depends_on"] = [dep if ":" in dep else owner + ":" + dep for dep in claim["depends_on"]]
        if len(set(claim["depends_on"])) != len(claim["depends_on"]):
            raise ValueError("duplicate resolved dependency")
        return claim

    @staticmethod
    def _check_graph(rows, allow_missing=False):
        active, done = set(), set()
        def visit(target):
            if target in active:
                raise ValueError("cyclic claim dependencies")
            if target in done:
                return
            if target not in rows:
                if allow_missing:
                    return
                raise ValueError("unknown claim dependency")
            active.add(target)
            for dep in rows[target]["claim"]["depends_on"]:
                visit(dep)
            active.remove(target)
            done.add(target)
        for target in rows:
            visit(target)

    @staticmethod
    def _refresh(rows):
        done = set()
        def update(target):
            if target in done:
                return
            row = rows[target]
            deps = row["claim"]["depends_on"]
            for dep in deps:
                if dep in rows:
                    update(dep)
            blocked = [dep for dep in deps if dep not in rows or rows[dep]["status"] != "supported"]
            row["blocked_by"] = blocked
            row["status"] = ("retracted" if row["own_status"] == "retracted" else
                             "invalidated" if blocked else row["own_status"])
            done.add(target)
        for target in rows:
            update(target)

    def apply(self, reviewer, target, proposal, verifier):
        if reviewer not in self.owners and reviewer != "central":
            raise ValueError("unknown reviewer")
        row = self.claims[target]
        if row["own_status"] == "retracted":
            raise ValueError("cannot mutate retracted claim")
        check_patch(proposal, target, fingerprint(row))
        replacement = proposal["replacement"]
        if row["claim"]["claim_type"] not in FACT_TYPES:
            if replacement is not None:
                raise ValueError("hypotheses cannot receive factual replacement certification")
            check = {"verdict": "not_evaluated", "kind": "hypothesis", "usage": {}}
            action, status, new_check = "defer", "hypothesis", None
        else:
            if replacement is not None:
                check_report({"ranking": [m["movie_id"] for m in self.evidence[
                    "candidates_in_shared_retrieval_order"][:5]], "claims": [replacement], "updates": []}, self.evidence)
                for field in ("claim_id", "candidate_ids", "claim_type", "depends_on"):
                    if replacement[field] != row["claim"][field]:
                        raise ValueError("replacement changes scope: " + field)
            # No state mutation until all checks, including replacement, finish.
            check = verifier.verify(row["claim"])
            new_check = None
            if check["verdict"] not in {"supported", "refuted", "insufficient"}:
                raise ValueError("unknown verifier verdict")
            if replacement is not None:
                if check["verdict"] != "refuted":
                    raise ReplacementPreconditionError("replacement requires old refutation")
                new_check = verifier.verify(replacement)
                if new_check["verdict"] != "supported":
                    raise ReplacementPreconditionError("replacement requires new support")
                action, status = "replace", "supported"
            else:
                action, status = {"supported": ("keep", "supported"), "refuted": ("retract", "retracted"),
                                  "insufficient": ("defer", "unresolved")}[check["verdict"]]
        before = copy.deepcopy(self.claims)
        updated = copy.deepcopy(self.claims)
        changed = updated[target]
        if replacement is not None:
            changed["claim"] = copy.deepcopy(replacement)
        changed.update(own_status=status, revision=changed["revision"] + 1, verification=new_check or check)
        self._refresh(updated)
        self.claims = updated
        event = {"reviewer": reviewer, "target": target, "action": action,
                 "before": before[target], "after": copy.deepcopy(updated[target]),
                 "old_verification": check, "replacement_verification": new_check,
                 "dependent_changes": {key: {"before": before[key], "after": copy.deepcopy(value)}
                     for key, value in updated.items() if key != target and value != before[key]}}
        self.events.append(event)
        return copy.deepcopy(event)

    def replace_owner(self, owner, report):
        """Ordinary/central unconstrained revision; no hidden evidence gating."""
        check_report(report, self.evidence)
        before = copy.deepcopy(self.claims)
        updated = {key: copy.deepcopy(row) for key, row in before.items() if row["owner"] != owner}
        for claim in report["claims"]:
            claim = self._qualified_claim(owner, claim)
            target = owner + ":" + claim["claim_id"]
            revision = before.get(target, {}).get("revision", 0) + 1
            if target in before and claim == before[target]["claim"]:
                updated[target] = copy.deepcopy(before[target])
                updated[target]["revision"] = revision
            else:
                updated[target] = self._row(owner, claim, revision)
        # Removed premises leave dependents invalidated, never silently supported.
        introduced = {key for key in updated if key not in before or updated[key]["claim"] != before[key]["claim"]}
        for key in introduced:
            if any(dep not in updated for dep in updated[key]["claim"]["depends_on"]):
                raise ValueError("new report has missing dependency")
        self._check_graph(updated, allow_missing=True)
        self._refresh(updated)
        self.claims = updated
        self.events.append({"action": "whole_report", "owner": owner, "before": before,
                            "after": copy.deepcopy(updated)})

    def payload(self):
        return {"supported_claims": {key: copy.deepcopy(row) for key, row in self.claims.items()
                                     if row["status"] == "supported"},
                "uncertain_claims": {key: copy.deepcopy(row) for key, row in self.claims.items()
                                     if row["status"] not in {"supported", "retracted"}},
                "retracted_ids": sorted(key for key, row in self.claims.items() if row["status"] == "retracted"),
                "instruction": "Uncertain includes unverified, unresolved, hypothetical and invalidated premises. "
                               "None are certified facts. Retracted IDs supply no support."}
