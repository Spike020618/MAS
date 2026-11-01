"""D1 engineering runner. No pilot, reserved data or recommendation scoring."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.contracts_v6 import METHODS
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.preflight_v6 import prepare, verify_plan
from experiments.paired_deliberation.runner_v6 import shared_initial, run_suite


def acceptance(results, plan):
    paths = [case["methods"].get(method) for case in results for method in METHODS]
    complete = len(results) == len(plan["case_ids"]) and len(paths) == 15 and all(
        row is not None and row["status"] == "completed" for row in paths)
    completed = [row["result"] for row in paths if row is not None and row["status"] == "completed"]
    rejected = sum(sum(rejection.get("kind", "contract") == "contract" for rejection in row["rejected"]) for row in completed)
    semantic_rejected = sum(sum(rejection.get("kind") == "semantic_precondition" for rejection in row["rejected"]) for row in completed)
    models = {row["usage"]["model"] for row in completed}
    unknown = sum(not row["usage"]["usage_complete"] for row in completed)
    verification_failures = sum(check.get("verification_failed", False)
                                for row in completed for check in row["verification_cache"].values())
    gates = {"all_paths_completed": complete, "zero_contract_rejection": rejected == 0,
             "no_invalid_verifier_outputs": verification_failures == 0,
             "one_returned_model": len(models) == 1, "known_usage": unknown == 0,
             "within_budgets": all(row["usage"]["attempts"] <= plan["limits"]["max_calls"]
                 and row["usage"]["reserved_prompt_upper"] <= plan["limits"]["total_prompt_tokens"]
                 and row["usage"]["reserved_output"] <= plan["limits"]["total_completion_tokens"]
                 for row in completed)}
    return {"pass": all(gates.values()), "gates": gates, "rejected_proposals": rejected,
            "semantic_precondition_refusals": semantic_rejected,
            "invalid_verifier_outputs": verification_failures, "completed_paths": len(completed),
            "models": sorted(models), "pilot_allowed": False,
            "scope": "engineering only; no semantic accuracy or recommendation gain claim"}


def run(output, phase="prepare", execute=False):
    output = Path(output)
    if phase == "prepare":
        plan = prepare(output)
        return {"status": "prepared", "plan_hash": plan["plan_hash"], "new_api_calls": 0,
                "maximum_physical_attempts": plan["maximum_physical_attempts"]}
    plan = verify_plan(output)
    offline = json.loads((output / "offline_acceptance.json").read_text())
    if not offline["pass"] or offline["plan_hash"] != plan["plan_hash"]:
        raise ValueError("matching offline acceptance required")
    if phase != "engineering" or not execute:
        raise ValueError("engineering requires explicit execute flag; no automatic network dispatch")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    raw_transport = RawDeepSeek(model=plan["model"])
    slots = output / "slots"

    class CappedTransport:
        def attempts(self):
            return len({p.name.split(".")[0] for p in slots.glob("*.raw.json")} |
                       {p.name.split(".")[0] for p in slots.glob("*.intent.json")})

        def preflight(self, request):
            raw_transport.preflight(request)
            if self.attempts() >= plan["maximum_physical_attempts"]:
                raise RuntimeError("batch attempt cap reached before dispatch")

        def __call__(self, request):
            if self.attempts() > plan["maximum_physical_attempts"]:
                raise RuntimeError("batch attempt cap reached")
            return raw_transport(request)

    previous = output.parent / "paired_deliberation_d1_v6_engineering_r2/slots"
    provider = DurableProvider(slots, plan, CappedTransport(), parse_raw, True,
                               reuse_initial=previous if previous.exists() and output.name.endswith("_r3") else None,
                               reuse_all=output.name.endswith("_r3"))
    collected = []
    for cid in plan["case_ids"]:
        prefix = cid.replace(":", "_")
        path = output / "cases" / (prefix + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["integrity"] != fingerprint(saved["result"])
                    or saved["result"]["plan_hash"] != plan["plan_hash"]):
                raise ValueError("case checkpoint integrity mismatch")
            collected.append(saved["result"])
            if any(row["status"] != "completed" for row in saved["result"]["methods"].values()):
                break
            continue
        evidence = json.loads((output / "inputs" / (prefix + ".json")).read_text())
        shared_client = AttemptClient(provider, plan["limits"])
        try:
            shared = shared_initial(evidence, shared_client, prefix + "_shared")
        except Exception as exc:
            result = {"case_id": cid, "plan_hash": plan["plan_hash"], "methods": {},
                      "shared_initial_failure": {"error": type(exc).__name__ + ": " + str(exc),
                                                 "usage": shared_client.summary()}}
            create(path, {"result": result, "integrity": fingerprint(result)})
            collected.append(result)
            break
        create(output / "shared" / (prefix + ".json"), shared)
        result = {"case_id": cid, "plan_hash": plan["plan_hash"], "methods": run_suite(evidence,
            lambda _: AttemptClient(provider, plan["limits"]), shared, plan["engineering_rounds"])}
        create(path, {"result": result, "integrity": fingerprint(result)})
        collected.append(result)
        print(json.dumps({"case_id": cid, "completed_paths": sum(row["status"] == "completed"
              for row in result["methods"].values()), "contract_rejections": sum(
              sum(r.get("kind", "contract") == "contract" for r in row["result"]["rejected"])
              for row in result["methods"].values() if row["status"] == "completed")}), flush=True)
        # A failing engineering case is sufficient to stop; do not buy more examples of the same failure.
        if (any(row["status"] != "completed" or any(r.get("kind", "contract") == "contract" for r in row["result"]["rejected"])
                or any(check.get("verification_failed") for check in row["result"]["verification_cache"].values())
                for row in result["methods"].values())):
            break
    result = acceptance(collected, plan)
    raw_files = list(slots.glob("*.raw.json"))
    reused = sum("reused_from" in json.loads(path.read_text()) for path in raw_files)
    physical = {"response_artifacts": len(raw_files), "reused_responses": reused, "new_responses": len(raw_files) - reused,
                "pending_intents": len(list(slots.glob("*.intent.json"))),
                "known_prompt_tokens": 0, "known_completion_tokens": 0, "unknown_usage_responses": 0}
    for path in raw_files:
        try:
            usage = json.loads(json.loads(path.read_text())["raw"]["body"])["usage"]
            for axis in ("prompt", "completion"):
                value = usage[axis + "_tokens"]
                if type(value) is not int or value < 0:
                    raise ValueError("unknown usage")
            physical["known_prompt_tokens"] += usage["prompt_tokens"]
            physical["known_completion_tokens"] += usage["completion_tokens"]
        except (ValueError, KeyError, TypeError):
            physical["unknown_usage_responses"] += 1
    result.update(plan_hash=plan["plan_hash"], physical=physical)
    if physical["pending_intents"] or physical["unknown_usage_responses"]:
        result["pass"] = False
    create(output / "engineering_acceptance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "engineering"), default="prepare")
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.phase, args.execute_authorized), ensure_ascii=False, indent=2))
