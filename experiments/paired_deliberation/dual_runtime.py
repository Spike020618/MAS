"""Adapt the frozen D1 engine to the dual contract without modifying its sources.

The underlying client reserves and persists the actual dual request/response.
Only the in-memory response is projected to the engine's legacy judge interface;
both evidence sides and their provenance remain in conversion records.
"""
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.conflict_verifier import DualEvidenceVerifier, request as dual_request


class DualRuntimeClient:
    def __init__(self, client, verifier_class=DualEvidenceVerifier, request_builder=dual_request):
        self.client = client
        self.verifier_class, self.request_builder = verifier_class, request_builder
        self.conversions = []

    @property
    def limits(self):
        return self.client.limits

    def action_fits(self, *args, **kwargs):
        return self.client.action_fits(*args, **kwargs)

    def account_shared(self, **kwargs):
        return self.client.account_shared(**kwargs)

    def summary(self):
        return self.client.summary()

    def complete(self, slot, request):
        if request["kind"] != "verification":
            return self.client.complete(slot, request)
        old_payload = json.loads(request["messages"][1]["content"])
        if set(old_payload) != {"claim", "evidence", "schema"}:
            raise ValueError("unexpected verification envelope")
        payload = {key: old_payload[key] for key in ("claim", "evidence")}
        actual_request = self.request_builder(payload)
        record = self.client.complete(slot, actual_request)
        conversion = {"slot": slot, "actual_request_hash": fingerprint(actual_request),
                      "actual_report_hash": fingerprint(record["report"]), "status": "received"}
        self.conversions.append(conversion)
        def judge(_payload):
            return {**copy.deepcopy(record["report"]), "usage": copy.deepcopy(record["usage"])}
        try:
            verified = self.verifier_class(payload["evidence"], judge).verify(payload["claim"])
        except Exception:
            conversion["status"] = "invalid_dual_response"
            raise
        conversion.update(status="derived", verification=copy.deepcopy(verified))
        projected = copy.deepcopy(record)
        projected["report"] = {"verdict": verified["verdict"], "reason": verified["reason"],
                               "evidence_spans": verified.get("evidence_spans", [])}
        return projected


def _run_suite(evidence, client_factory, shared=None, max_rounds=3,
               verifier_class=DualEvidenceVerifier, request_builder=dual_request):
    from experiments.paired_deliberation.runner_v6 import run_suite as run_frozen_suite
    clients = {}
    def factory(method):
        clients[method] = DualRuntimeClient(client_factory(method), verifier_class, request_builder)
        return clients[method]
    results = run_frozen_suite(evidence, factory, shared, max_rounds)
    for method, result in results.items():
        result["dual_evidence_conversions"] = copy.deepcopy(clients[method].conversions)
    return results


def check_release(validation_root):
    root = Path(validation_root)
    score_path = root / "release_score.json"
    if not score_path.exists():
        raise RuntimeError("passing scoped semantic release required before integration")
    score = json.loads(score_path.read_text())
    if (not score.get("pass") or not all(score.get("gates", {}).values())
            or score.get("extra_regression", {}).get("errors")):
        raise RuntimeError("scoped semantic gate failed; no integration calls allowed")
    from experiments.paired_deliberation.validate_dual_evidence import verify_plan
    plan = verify_plan(root)
    if (score["plan_hash"] != plan["plan_hash"] or plan.get("contract_version") != "scoped_dual_v2"
            or score.get("contract_version") != "scoped_dual_v2"):
        raise RuntimeError("semantic release does not match the scoped contract")
    return plan, score


def run_suite(evidence, client_factory, validation_root, shared=None, max_rounds=3):
    """Public runtime: only a matching passing scope-aware release can dispatch."""
    check_release(validation_root)
    from experiments.paired_deliberation.scoped_verifier import ScopedEvidenceVerifier, request
    return _run_suite(evidence, client_factory, shared, max_rounds, ScopedEvidenceVerifier, request)
