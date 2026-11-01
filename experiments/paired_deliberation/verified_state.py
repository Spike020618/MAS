"""V5 claim-scoped verification and transactional state transitions.

No evaluation labels. Catalog entailment is deterministic for the exact bounded
grammar below. Arbitrary natural language requires an injected evidence judge;
absence of a judge or conflicting judgements results in abstention.
"""
import copy
import re

from experiments.consensus_recommendation.common import fingerprint


VERDICTS = {"supported", "refuted", "insufficient"}
FACT = re.compile(r"Movie ([^\s]+) has catalog (genre|year) (.+)\.")


class VerificationError(ValueError):
    """Untrusted semantic judge output; never an entailment verdict."""


def canonical(value):
    import json
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class EvidenceVerifier:
    def __init__(self, sources, judge=None):
        if not isinstance(sources, dict):
            raise ValueError("sources must be an ID-to-source mapping")
        self.sources = copy.deepcopy(sources)
        self.judge = judge

    def verify(self, claim):
        refs = claim.get("source_ids")
        if (not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs)
                or len(refs) != len(set(refs)) or not set(refs) <= set(self.sources)):
            raise ValueError("invalid claim evidence references")
        text = claim.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty claim")
        payload = {"claim": copy.deepcopy(claim),
                   "evidence": {ref: copy.deepcopy(self.sources[ref]) for ref in refs}}
        match = FACT.fullmatch(text)
        if match:
            movie_id, field, value = match.groups()
            source = self.sources.get("movie:" + movie_id)
            if source is not None and "movie:" + movie_id in refs:
                observed = source.get("genres") if field == "genre" else source.get("year")
                if observed is not None:
                    supported = value in observed if field == "genre" and isinstance(observed, list) else str(observed) == value
                    return self._record("supported" if supported else "refuted", refs,
                                        "exact_catalog_fact", "Exact bounded claim compared with frozen catalog.", payload)
        if self.judge is None:
            return self._record("insufficient", refs, "abstention", "No semantic judge for this claim scope.", payload)
        result = self.judge(copy.deepcopy(payload))
        if not isinstance(result, dict) or set(result) != {"verdict", "reason", "evidence_spans", "usage"}:
            raise VerificationError("invalid semantic judge schema")
        if result["verdict"] not in VERDICTS or not isinstance(result["reason"], str) or not result["reason"].strip():
            raise VerificationError("invalid semantic judgement")
        if not isinstance(result["usage"], dict):
            raise VerificationError("semantic judge usage must be explicit")
        spans = result["evidence_spans"]
        if not isinstance(spans, list):
            raise VerificationError("semantic evidence spans must be a list")
        if result["verdict"] != "insufficient" and not spans:
            raise VerificationError("decisive semantic judgement needs evidence spans")
        for span in spans:
            if not isinstance(span, dict) or set(span) != {"source_id", "quote"}:
                raise VerificationError("invalid evidence span")
            if span["source_id"] not in refs or not isinstance(span["quote"], str) or not span["quote"]:
                raise VerificationError("invalid evidence span source")
            source = self.sources[span["source_id"]]
            source_text = source.get("text", canonical(source)) if isinstance(source, dict) else str(source)
            # Structured lists have no contiguous prose substring in JSON.
            # Permit only this deterministic, lossless field projection, never
            # arbitrary paraphrases, reordered elements, or invented values.
            genres = source.get("genres") if isinstance(source, dict) else None
            projected_genres = (", ".join(genres) if isinstance(genres, list)
                                and genres and all(isinstance(x, str) for x in genres) else None)
            if span["quote"] not in source_text and span["quote"] != projected_genres:
                raise VerificationError("semantic judge fabricated a quoted span")
        record = self._record(result["verdict"], refs, "model_proxy", result["reason"], payload)
        record["evidence_spans"] = copy.deepcopy(spans)
        record["usage"] = copy.deepcopy(result["usage"])
        return record

    @staticmethod
    def _record(verdict, refs, kind, reason, payload):
        return {"verdict": verdict, "source_ids": list(refs), "kind": kind,
                "reason": reason, "input_hash": fingerprint(payload), "usage": {}}


class ClaimState:
    def __init__(self, reports, verifier, invalid_verification="raise"):
        if invalid_verification not in {"raise", "defer"}:
            raise ValueError("unknown invalid-verification policy")
        self.invalid_verification = invalid_verification
        self.verifier = verifier
        self.reports = copy.deepcopy(reports)
        self.claims = {}
        self.history = []
        for owner, report in reports.items():
            for claim in report["claims"]:
                local = claim["claim_id"]
                if not isinstance(local, str) or not local or ":" in local:
                    raise ValueError("initial claim IDs must be nonempty local IDs")
                target = owner + ":" + local
                if target in self.claims:
                    raise ValueError("duplicate qualified claim ID")
                self.claims[target] = {"owner": owner, "claim": copy.deepcopy(claim),
                                       "status": "unverified", "revision": 0, "verification": None}

    def _verify(self, claim):
        try:
            return self.verifier.verify(claim)
        except VerificationError as exc:
            if self.invalid_verification == "raise":
                raise
            return {"verdict": "insufficient", "kind": "invalid_proxy",
                    "reason": str(exc), "source_ids": claim["source_ids"],
                    "usage": {}, "verification_failed": True}

    def apply(self, reviewer, proposal):
        if reviewer not in self.reports:
            raise ValueError("unknown reviewer")
        required = {"target_claim_id", "expected_hash", "replacement"}
        if not isinstance(proposal, dict) or set(proposal) != required:
            raise ValueError("proposal is one scoped patch, never a report or ranking")
        target = proposal["target_claim_id"]
        if target not in self.claims:
            raise ValueError("unknown update target")
        current = self.claims[target]
        if current["status"] == "retracted":
            raise ValueError("cannot mutate a retracted claim")
        if proposal["expected_hash"] != fingerprint(current):
            raise ValueError("stale claim revision")
        before = copy.deepcopy(current)
        old_check = self._verify(current["claim"])
        replacement = proposal["replacement"]
        new_check = None
        if replacement is not None:
            required_claim = {"claim_id", "text", "source_ids", "candidate_ids", "stance"}
            if not isinstance(replacement, dict) or set(replacement) != required_claim:
                raise ValueError("replacement claim schema")
            if replacement["claim_id"] != current["claim"]["claim_id"]:
                raise ValueError("replacement cannot rename a claim")
            if replacement["candidate_ids"] != current["claim"]["candidate_ids"]:
                raise ValueError("replacement cannot change candidate binding")
            if replacement["stance"] not in {"support", "oppose", "unknown"}:
                raise ValueError("replacement stance")
            if old_check["verdict"] != "refuted":
                raise ValueError("replacement requires refutation of the old claim")
            new_check = self._verify(replacement)
            if new_check["verdict"] != "supported":
                raise ValueError("replacement requires independent support")
            action, status = "replace", "supported"
            current["claim"] = copy.deepcopy(replacement)
        elif old_check["verdict"] == "refuted":
            action, status = "retract", "retracted"
        elif old_check["verdict"] == "supported":
            action, status = "keep", "supported"
        else:
            action, status = "defer", "unresolved"
        current["status"] = status
        current["revision"] += 1
        current["verification"] = new_check or old_check
        event = {"reviewer": reviewer, "owner": current["owner"], "target": target,
                 "action": action, "before": before, "after": copy.deepcopy(current),
                 "old_verification": old_check, "replacement_verification": new_check}
        self.history.append(event)
        return copy.deepcopy(event)

    def aggregation_payload(self):
        return {
            "supported_claims": {key: copy.deepcopy(value) for key, value in self.claims.items()
                                 if value["status"] == "supported"},
            "uncertain_claims": {key: copy.deepcopy(value) for key, value in self.claims.items()
                                 if value["status"] in {"unresolved", "unverified"}},
            "retracted_ids": [key for key, value in self.claims.items() if value["status"] == "retracted"],
            "instruction": "Use supported claims as evidence; uncertain claims are hypotheses. Retracted IDs provide no support.",
        }
