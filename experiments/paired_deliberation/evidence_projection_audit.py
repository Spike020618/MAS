"""Offline fixed-schedule evidence projection; no new model/ranking predictions.

Ignores revision proposals and consumes the recorded verification results via a
null patch. It diagnoses observation/proposal coupling, not an online v4 effect.
"""
import argparse
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint, sha
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.controlled_effect import verify_plan


def project(evidence, initial, schedule, checks):
    state = StateV6(initial, evidence)
    missing = []
    for step in schedule:
        target = step["target"]
        row = state.claims[target]
        verification = checks.get(fingerprint(row["claim"]))
        if verification is None:
            missing.append(target)
            continue
        class Recorded:
            def verify(self, claim): return copy.deepcopy(verification)
        # This only consumes an existing judgement. It does not infer truth or
        # execute the original model's replacement under a rewritten hash.
        state.apply(row["owner"], target, {"target_claim_id": target,
                    "expected_hash": fingerprint(row), "replacement": None}, Recorded())
    return state, missing


def run(root):
    root = Path(root)
    plan = verify_plan(root)
    inputs = {r["id"]: r for r in json.loads((root / "inputs.json").read_text())}
    labels = json.loads((root / "labels.json").read_text())
    score = json.loads((root / "score.json").read_text())
    if score["plan_hash"] != plan["plan_hash"] or fingerprint(labels) != plan["labels_hash"]:
        raise ValueError("source score/labels changed")
    rows = []
    for label in labels:
        path = root / "cases" / (label["id"] + ".json")
        saved = json.loads(path.read_text())
        if saved["integrity"] != fingerprint(saved["result"]): raise ValueError("case changed")
        for method in ("verified_fixed", "verified_adaptive"):
            observed = saved["result"]["methods"][method]
            if observed["status"] != "completed": continue
            result = observed["result"]
            projection, missing = project(inputs[label["id"]]["evidence"], inputs[label["id"]]["shared"]["initial"],
                                          result["schedule"], result["verification_cache"])
            target = label["target"]
            checked = label["target_hash"] in result["verification_cache"]
            wanted = {"supported": "supported", "refuted": "retracted", "insufficient": "unresolved"}[label["expected_verdict"]]
            original = next(r for r in score["details"] if r["id"] == label["id"] and r["method"] == method)
            rows.append({"id": label["id"], "method": method, "family": label["family"], "corrupted": label["corrupted"],
                "target_checked": checked, "expected_status": wanted,
                "observed_state_outcome_correct": original["scoped_state_outcome_matches_gold_conditional"],
                "projected_target_status": projection.claims[target]["status"],
                "projected_state_outcome_correct": projection.claims[target]["status"] == wanted if checked else None,
                "missing_checks": missing, "case_sha256": sha(path)})
    summary = {}
    for method in ("verified_fixed", "verified_adaptive"):
        selected = [r for r in rows if r["method"] == method and r["target_checked"]]
        summary[method] = {"checked": len(selected), "observed_correct": sum(r["observed_state_outcome_correct"] is True for r in selected),
            "projection_correct": sum(r["projected_state_outcome_correct"] is True for r in selected),
            "missing_recorded_checks": sum(len(r["missing_checks"]) for r in selected)}
    report = {"type": "posthoc_fixed_observations_projection", "plan_hash": plan["plan_hash"],
        "source_score_sha256": sha(root / "score.json"), "report_code_sha256": sha(Path(__file__)),
        "summary": summary, "rows": rows, "new_api_calls": 0,
        "limits": ["same recorded checks and selected targets, not a new online trajectory",
                   "no replacement proposals applied, no fresh model responses, no ranking recomputation",
                   "does not demonstrate causal recommendation gain or validate a new live mechanism"]}
    create(root / "posthoc_evidence_projection.json", report)
    return {k: v for k, v in report.items() if k != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), ensure_ascii=False, indent=2))
