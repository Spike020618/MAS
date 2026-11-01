"""Offline provenance/contract replay and immutable engineering preparation."""
import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, OUTPUT, create, fingerprint, read_rows, sha
from experiments.consensus_recommendation.protocol import bundle, validate
from experiments.paired_deliberation.contracts_v6 import METHODS, initial_request, review_request
from experiments.paired_deliberation.run_experiment import load_case_ids
from experiments.paired_deliberation.state_v6 import StateV6

VERSION = "d1-v6-on-demand-engineering-r3"
LIMITS = {"max_calls": 20, "max_evidence_checks": 6, "per_call_prompt_tokens": 64000, "per_call_completion_tokens": 1280,
          "total_prompt_tokens": 640000, "total_completion_tokens": 25600}
OLD_RUN = ROOT / "experiments/results/paired_deliberation_v51_preflight_r2"


def historical_replay(root=OLD_RUN):
    """Original outputs stay invalid; improved instructions are not retroactive fixes."""
    root = Path(root)
    rows, counts = [], Counter()
    for case_path in sorted((root / "cases").glob("*.json")):
        case = json.loads(case_path.read_text())
        if fingerprint(case["result"]) != case["integrity"]:
            raise ValueError("historical case integrity mismatch")
        for method, result in case["result"].items():
            if method == "no_scope_constraint":
                continue
            for rejected in result.get("rejected_proposals", []):
                target = rejected["target"]
                matches = []
                for raw_path in sorted((root / "slots").glob(case_path.stem + "_" + method + "_review_*.raw.json")):
                    raw = json.loads(raw_path.read_text())
                    payload = json.loads(raw["request"]["messages"][1]["content"])
                    if payload["target_claim_id"] == target:
                        matches.append((raw_path, raw, payload))
                if len(matches) != 1:
                    raise ValueError("cannot identify unique original failed response")
                path, raw, payload = matches[0]
                if raw["integrity"] != fingerprint(raw["raw"]):
                    raise ValueError("historical raw integrity mismatch")
                obj = json.loads(json.loads(raw["raw"]["body"])["choices"][0]["message"]["content"])
                if obj.get("type") == "json_object":
                    obj = {k: v for k, v in obj.items() if k != "type"}
                whole = raw["request"]["kind"] == "whole_review"
                error = None
                try:
                    if whole:
                        validate(obj, payload["evidence"])
                    elif set(obj) != {"target_claim_id", "expected_hash", "requested_action", "replacement"}:
                        raise ValueError("scoped review schema")
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                category = ("claim_count" if whole and isinstance(obj.get("claims"), list) and len(obj["claims"]) > 3
                            else "wrapper" if "schema" in obj else "missing_hash" if not whole and "expected_hash" not in obj
                            else "other_contract" if error else "semantic_precondition")
                counts[category] += 1
                rows.append({"case": case_path.stem, "method": method, "target": target,
                    "raw_path": str(path.relative_to(ROOT)), "raw_sha256": sha(path),
                    "original_error": rejected["error"], "replay_contract_error": error,
                    "category": category, "original_response_unchanged": True})
    if not rows:
        raise ValueError("historical failure fixtures unavailable")
    return {"rejected": len(rows), "counts": dict(counts), "rows": rows,
            "scope": "Replay of original failures; no regeneration, semantic repair or historical score change."}


def code_hashes():
    own = Path(__file__).parent
    paths = sorted(own.glob("*_v6.py")) + [own / name for name in (
        "verified_state.py", "requests_v5.py", "client_v5.py", "durable_provider.py", "pipeline_v5.py")]
    paths += [own.parent / "consensus_recommendation" / name for name in ("protocol.py", "common.py")]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def prepare(output):
    output = Path(output)
    # No development or reserved labels are opened, including for sample selection.
    ids = load_case_ids("engineering")[:3]
    inputs = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.inputs.jsonl")}
    candidates = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.candidates.jsonl")}
    from experiments.scripts.data.build_redial_benchmark import load_catalog
    _, _, movies = load_catalog(ROOT / "experiments/data/processed/movie_catalog.jsonl")
    cases = [bundle(inputs[cid], candidates[cid], movies) for cid in ids]
    for evidence in cases:
        create(output / "inputs" / (evidence["case_id"].replace(":", "_") + ".json"), evidence)
    replay = historical_replay()
    create(output / "historical_contract_replay.json", replay)
    source_paths = [OUTPUT / "data" / name for name in
                    ("dev.inputs.jsonl", "dev.candidates.jsonl", "manifest.json", "development_retrieval_audit.json")]
    source_paths += [ROOT / "experiments/data/processed/movie_catalog.jsonl"]
    plan = {"version": VERSION, "research_point_count": 1,
        "components": ["evidence_constrained_updates", "budget_coordination"],
        "status": "engineering_preparation_not_effect_protocol", "case_ids": ids, "methods": list(METHODS),
        "level": "shared_initial_mechanism", "engineering_rounds": 3,
        "model": "deepseek-chat", "temperature": 0, "limits": LIMITS,
        "maximum_physical_attempts": len(ids) * (3 + 4 + 10 + 13 + 10 + 10),
        "attempt_bound_note": "per case: shared initial 3; C0 final1 (upper4); peer<=7 (upper10); "
                              "central<=10 (upper13); fixed/adaptive<=10 each, including final; conservative 50 per case",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "code_sha256": code_hashes(), "case_hashes": {case["case_id"]: fingerprint(case) for case in cases},
        "runtime_labels_available": False, "reserved_test_allowed": False, "pilot_allowed": False,
        "candidate_policy": "existing label-free hybrid retrieval order, unchanged 100-candidate pool; no target injection",
        "primary_future_metric": "Hit@5 against frozen eligible target set; labels not accessed by this preparation",
        "scheduling": {"fixed": "owner-round-robin, lexical claims", "adaptive": {
            "opposing_stances": 2, "uncertain_stance": 1, "ranking_disagreement": 1, "has_dependents": 1},
            "cost_assumption": "equal bounded compound review actions; actual costs separately recorded",
            "stop": "3 actions, no unchecked factual target, or budget reserved for final"},
        "evidence_acquisition": "on-demand interpretation of frozen available sources; no new web retrieval",
        "failure_policy": "no automatic retry; failed path reports common retrieval-top5 fallback separately; "
                          "unknown-billing pending intent stops dispatch; contract rejection counts as engineering failure",
        "rejection_policy": "Separate malformed contracts from legitimate replacement precondition refusals; "
                            "semantic refusals stay in process metrics and never become accepted replacements.",
        "engineering_acceptance": {"all_15_paths_completed": True, "contract_rejections_max": 0,
                                   "within_budgets": True, "no_unknown_usage": True, "one_returned_model": True},
        "scope": "Engineering only; does not validate natural-language accuracy, coordination or recommendation gain.",
        "formal_parameters_pending": ["independent evaluation sample/contamination audit", "power and repetitions",
            "controlled mechanism corpus split and semantic gates", "confirmatory multiple-comparison and budget curve"]}
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    return plan


def verify_plan(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if plan["plan_hash"] != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("plan integrity failed")
    if plan["code_sha256"] != code_hashes():
        raise ValueError("source changed; freeze a new engineering version")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("source data changed")
    for cid, expected in plan["case_hashes"].items():
        path = output / "inputs" / (cid.replace(":", "_") + ".json")
        if fingerprint(json.loads(path.read_text())) != expected:
            raise ValueError("input integrity mismatch")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({"status": result["status"], "cases": len(result["case_ids"]),
                      "methods": result["methods"], "maximum_physical_attempts": result["maximum_physical_attempts"],
                      "new_api_calls": 0}, ensure_ascii=False))
