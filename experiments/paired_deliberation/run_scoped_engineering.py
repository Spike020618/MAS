"""Five-condition engineering recheck, strictly gated on scoped validation."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.dual_runtime import check_release, run_suite
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.runner_v6 import shared_initial
from experiments.paired_deliberation.run_v6 import acceptance

PRIOR = ROOT / "experiments/results/paired_deliberation_d1_v6_engineering_r3"


def prepare(output, validation):
    output, validation = Path(output), Path(validation).resolve()
    _, score = check_release(validation)
    original = json.loads((PRIOR / "plan.json").read_text())
    files = [ROOT / name for name in original["code_sha256"]]
    files += [Path(__file__), Path(__file__).with_name("dual_runtime.py"),
              Path(__file__).with_name("scoped_verifier.py"), Path(__file__).with_name("validate_scoped_evidence.py"),
              Path(__file__).with_name("validate_dual_evidence.py")]
    plan = {**original, "version": "d1-scoped-dual-engineering-r1", "validation_root": str(validation),
        "validation_score_sha256": sha(validation / "release_score.json"),
        "validation_model": score["physical"]["models"], "maximum_new_requests": 150,
        "source_sha256": original["source_sha256"],
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        "reuse_source": str(PRIOR / "slots"),
        "reuse_sha256": {p.name: sha(p) for p in sorted((PRIOR / "slots").glob("*.raw.json"))},
        "scope": "Shared-initial five-condition engineering after scope-aware verifier validation; no task labels or effect scores."}
    plan.pop("plan_hash", None)
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    for cid in plan["case_ids"]:
        path = PRIOR / "inputs" / (cid.replace(":", "_") + ".json")
        create(output / "inputs" / path.name, json.loads(path.read_text()))
    for p in sorted(set(files)):
        target = output / "source_snapshot" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != p.read_bytes():
            raise ValueError("snapshot changed")
        target.write_bytes(p.read_bytes())
    return {"status": "prepared", "cases": 3, "methods": 5, "maximum_new_requests": 150, "new_api_calls": 0}


def execute(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if plan["plan_hash"] != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("plan changed")
    check_release(plan["validation_root"])
    if sha(Path(plan["validation_root"]) / "release_score.json") != plan["validation_score_sha256"]:
        raise ValueError("release changed")
    for name, expected in plan["code_sha256"].items():
        if sha(ROOT / name) != expected or sha(output / "source_snapshot" / name) != expected:
            raise ValueError("source changed")
    for name, expected in plan["reuse_sha256"].items():
        if sha(Path(plan["reuse_source"]) / name) != expected:
            raise ValueError("reuse source changed")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("data source changed")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    raw = RawDeepSeek(model=plan["model"])
    slots = output / "slots"
    class Transport:
        def count(self):
            received = sum("reused_from" not in json.loads(p.read_text()) for p in slots.glob("*.raw.json"))
            return received + len(list(slots.glob("*.intent.json")))
        def preflight(self, request):
            raw.preflight(request)
            if self.count() >= plan["maximum_new_requests"]:
                raise RuntimeError("batch cap reached")
        def __call__(self, request):
            if self.count() > plan["maximum_new_requests"]:
                raise RuntimeError("batch cap reached")
            try:
                return raw(request)
            except OSError as exc:
                raise RuntimeError("transport failed; stop without repeating pending intent: " + str(exc)) from exc
    provider = DurableProvider(slots, plan, Transport(), parse_raw, True,
                               reuse_initial=plan["reuse_source"], reuse_all=True)
    results = []
    for cid in plan["case_ids"]:
        prefix = cid.replace(":", "_")
        path = output / "cases" / (prefix + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["integrity"] != fingerprint(saved["result"]) or saved["result"]["plan_hash"] != plan["plan_hash"]:
                raise ValueError("case integrity mismatch")
            results.append(saved["result"])
            if saved["result"].get("stop_after_case"):
                break
            continue
        evidence = json.loads((output / "inputs" / (prefix + ".json")).read_text())
        if fingerprint(evidence) != plan["case_hashes"][cid]:
            raise ValueError("input changed")
        client = AttemptClient(provider, plan["limits"])
        result = {"case_id": cid, "plan_hash": plan["plan_hash"], "methods": {}}
        try:
            shared = shared_initial(evidence, client, prefix + "_shared")
            if {call["record"]["model"] for call in shared["calls"]} != set(plan["validation_model"]):
                raise ValueError("model differs from semantic validation")
            create(output / "shared" / (prefix + ".json"), shared)
            result["methods"] = run_suite(evidence, lambda _: AttemptClient(provider, plan["limits"]),
                                           plan["validation_root"], shared, plan["engineering_rounds"])
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            result["failure"] = {"error": str(exc), "shared_usage": client.summary()}
        failed = bool(result.get("failure")) or len(result["methods"]) != 5
        for record in result["methods"].values():
            if record["status"] != "completed":
                failed = True
                continue
            row = record["result"]
            failed |= (any(r["kind"] == "contract" for r in row["rejected"])
                       or any(v.get("verification_failed") for v in row["verification_cache"].values())
                       or row["usage"]["model"] not in plan["validation_model"])
        result["stop_after_case"] = bool(failed)
        create(path, {"result": result, "integrity": fingerprint(result)})
        results.append(result)
        print(json.dumps({"case_id": cid, "paths": len(result["methods"]), "stop": bool(failed)}), flush=True)
        if failed:
            break
    report = acceptance(results, plan)
    raw_files = [json.loads(p.read_text()) for p in slots.glob("*.raw.json")]
    new = [r for r in raw_files if "reused_from" not in r]
    bodies, unknown_usage = [], 0
    for r in new:
        try:
            body = json.loads(r["raw"]["body"])
            if any(type(body.get("usage", {}).get(k)) is not int or body["usage"][k] < 0
                   for k in ("prompt_tokens", "completion_tokens")):
                raise ValueError("unknown usage")
            bodies.append(body)
        except (ValueError, KeyError, TypeError):
            unknown_usage += 1
    report.update(plan_hash=plan["plan_hash"], physical={"new_responses": len(new),
        "reused_responses": len(raw_files) - len(new), "pending_intents": len(list(slots.glob("*.intent.json"))),
        "unknown_usage_responses": unknown_usage,
        "known_prompt_tokens": sum(b["usage"]["prompt_tokens"] for b in bodies),
        "known_completion_tokens": sum(b["usage"]["completion_tokens"] for b in bodies),
        "usage_complete": unknown_usage == 0})
    report["pass"] &= (not report["physical"]["pending_intents"] and unknown_usage == 0
                        and not any(r["stop_after_case"] for r in results))
    create(output / "engineering_acceptance.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.output) if args.execute_authorized else prepare(args.output, args.validation),
                     ensure_ascii=False, indent=2))
