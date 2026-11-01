"""V5.1 calibration/comparison entry; prepare-only by default."""
import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, OUTPUT, create, fingerprint, read_rows, sha
from experiments.consensus_recommendation.protocol import bundle
from experiments.paired_deliberation.calibration_v51 import export
from experiments.paired_deliberation.client_v5 import BoundedClient, RawDeepSeek, parse_raw
from experiments.paired_deliberation.comparison_v51 import generate_shared, run_suite, METHODS, ALIASES
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.run_experiment import load_case_ids
from experiments.paired_deliberation.verified_state import EvidenceVerifier, VerificationError


def manifest(output):
    calibration = export(output / "calibration")
    names = ("run_v51.py", "calibration_v51.py", "comparison_v51.py", "verified_state.py",
             "client_v5.py", "requests_v5.py", "durable_provider.py", "pipeline_v5.py")
    plan = {"version": "v51-shared-feedback-comparison", "engineering_case_ids": load_case_ids("engineering")[:3],
            "calibration": calibration, "methods": list(METHODS), "aliases": ALIASES,
            "model": "deepseek-chat", "temperature": 0,
            "path_limits": {"per_call_prompt_tokens": 64000, "per_call_completion_tokens": 1280,
                            "total_prompt_tokens": 520000, "total_completion_tokens": 80000},
            "code_sha256": {name: sha(Path(__file__).parent / name) for name in names},
            "data_sha256": {name: sha(OUTPUT / "data" / name)
                            for name in ("dev.inputs.jsonl", "dev.candidates.jsonl")},
            "catalog_sha256": sha(ROOT / "experiments/data/processed/movie_catalog.jsonl"),
            "shared_protocol_sha256": sha(Path(__file__).parents[1] / "consensus_recommendation/protocol.py"),
            "runtime_labels_available": False, "reserved_test_allowed": False,
            "pilot_allowed": False,
            "scope": "controlled calibration plus three engineering comparisons; not confirmatory performance"}
    create(output / "plan.json", plan)
    return plan


def score_calibration(output, manifest_data):
    labels = json.loads((output / "calibration/labels.json").read_text())
    confusion, correct, totals = Counter(), Counter(), Counter()
    invalid = 0
    models = set()
    for row in labels:
        response = json.loads((output / "calibration/predictions" / (row["id"] + ".json")).read_text())
        record_path = output / "slots" / (row["id"] + ".json")
        if record_path.exists():
            models.add(json.loads(record_path.read_text())["record"]["model"])
        verdict = response.get("verdict", "invalid")
        confusion[(row["label"], verdict)] += 1
        invalid += verdict == "invalid"
        for axis in ("all", "label:" + row["label"], "category:" + row["category"]):
            totals[axis] += 1
            correct[axis] += verdict == row["label"]
    rates = {axis: correct[axis] / total for axis, total in totals.items()}
    gates = manifest_data["gates"]
    false_supported = sum(value for (gold, predicted), value in confusion.items()
                          if predicted == "supported" and gold != "supported")
    passed = (rates["all"] >= gates["accuracy_min"]
              and min(value for axis, value in rates.items() if axis.startswith("label:")) >= gates["recall_each_class_min"]
              and min(value for axis, value in rates.items() if axis.startswith("category:")) >= gates["accuracy_each_category_min"]
              and false_supported <= gates["false_supported_max"] and invalid == 0)
    result = {"items": len(labels), "rates": rates, "invalid": invalid, "false_supported": false_supported,
              "pass": passed, "confusion": {gold + "->" + predicted: value for (gold, predicted), value in confusion.items()},
              "scope": "controlled evidence-relative calibration, not real-dialogue or recommendation accuracy",
              "plan_sha256": sha(output / "plan.json"), "inputs_hash": manifest_data["inputs_hash"],
              "returned_models": sorted(models),
              "labels_sha256": sha(output / "calibration/labels.json")}
    create(output / "calibration/score.json", result)
    return result


def run(output, phase="prepare", execute=False):
    output = Path(output)
    plan = manifest(output)
    if not execute:
        return {"status": "prepared", "calibration_items": 84, "engineering_cases": 3,
                "executable_methods": len(METHODS), "alias_methods": ALIASES, "new_api_calls": 0}
    if phase not in {"calibration", "comparison"}:
        raise ValueError("execution requires calibration or comparison phase")
    if phase == "comparison":
        score = json.loads((output / "calibration/score.json").read_text())
        if (not score["pass"] or score["plan_sha256"] != sha(output / "plan.json")
                or score["inputs_hash"] != plan["calibration"]["inputs_hash"]
                or len(score["returned_models"]) != 1):
            raise ValueError("expanded calibration gate failed; comparison not authorized by this protocol")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    provider = DurableProvider(output / "slots", plan, RawDeepSeek(model=plan["model"]), parse_raw, True)
    if phase == "calibration":
        inputs = json.loads((output / "calibration/inputs.json").read_text())
        limits = {**plan["path_limits"], "total_prompt_tokens": 4000000, "total_completion_tokens": 180000}
        client = BoundedClient(provider, limits)
        for row in inputs:
            prediction_path = output / "calibration/predictions" / (row["id"] + ".json")
            def judge(payload, slot=row["id"]):
                response = client.complete(slot, RequestsV5.verification(payload))
                return {**response["report"], "usage": response["usage"]}
            try:
                prediction = EvidenceVerifier(row["evidence"], judge).verify(row["claim"])
            except VerificationError as exc:
                prediction = {"verdict": "invalid", "reason": str(exc)}
            create(prediction_path, prediction)
        return score_calibration(output, plan["calibration"])
    inputs = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.inputs.jsonl")}
    candidates = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.candidates.jsonl")}
    _, _, movies = __import__("experiments.scripts.data.build_redial_benchmark", fromlist=["load_catalog"]).load_catalog(
        ROOT / "experiments/data/processed/movie_catalog.jsonl")
    for cid in plan["engineering_case_ids"]:
        evidence = bundle(inputs[cid], candidates[cid], movies)
        case_path = output / "cases" / (cid.replace(":", "_") + ".json")
        if case_path.exists():
            saved = json.loads(case_path.read_text())
            if saved["integrity"] != fingerprint(saved["result"]):
                raise ValueError("tampered case checkpoint")
            continue
        shared = generate_shared(evidence, BoundedClient(provider, plan["path_limits"]))
        observed = {call["record"]["model"] for call in shared["initial_calls"] + shared["feedback_calls"]}
        if observed != set(score["returned_models"]):
            raise ValueError("returned model changed since calibration")
        create(output / "shared" / (cid.replace(":", "_") + ".json"), shared)
        result = run_suite(evidence, shared, lambda _method: BoundedClient(provider, plan["path_limits"]))
        create(case_path, {"result": result, "integrity": fingerprint(result)})
    return {"status": "three_engineering_comparisons_collected", "pilot_allowed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "calibration", "comparison"), default="prepare")
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.phase, args.execute_authorized), ensure_ascii=False, indent=2))
