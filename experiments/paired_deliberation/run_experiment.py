"""Prepare or explicitly execute the frozen ReDial E1/E2 development experiment."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import OUTPUT, ROOT, create, fingerprint, read_rows
from experiments.consensus_recommendation.protocol import FOCUS, bundle
from experiments.paired_deliberation.deepseek_transport import DeepSeekTransport
from experiments.paired_deliberation.pipeline import run_e1_method, run_e2_arm
from experiments.paired_deliberation.plan import authorize, build_plan
from experiments.paired_deliberation.provider import CheckpointedProvider
from experiments.paired_deliberation.redial_adapter import EvidenceContractChecker
from experiments.paired_deliberation.redial_requests import RedialRequestBuilder


ROLES = list(FOCUS)
E1_METHODS = ("single_once", "single_serial", "same_agent_samples",
              "multi_role", "central_all_evidence")
E2_METHODS = {
    "independent": ("independent", "frozen"),
    "self_review": ("self_review", "frozen"),
    "peer_review": ("peer_review", "frozen"),
    "evidence_update": ("evidence_update", "frozen"),
    "peer_sequential": ("peer_review", "sequential"),
    "update_sequential": ("evidence_update", "sequential"),
}


def load_case_ids(stage):
    audit = json.loads((OUTPUT / "data/development_retrieval_audit.json").read_text())
    key = stage + "_case_ids"
    if key not in audit:
        raise ValueError("stage case list unavailable")
    return audit[key]


def run(stage, output, execute_authorized=False, only_case=None):
    if execute_authorized:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    case_ids = load_case_ids(stage)
    budget = {"per_call_prompt_tokens": 32000, "per_call_completion_tokens": 1280,
              "total_prompt_tokens": 128000, "total_completion_tokens": 10000,
              "total_tool_calls": 0}
    plan = build_plan(case_ids, "deepseek-chat", budget, review_rounds=3)
    plan = authorize(plan, provider_calls=execute_authorized, reserved_test=False)
    create(output / "plan.json", plan)
    if not execute_authorized:
        return {"status": "prepared", "stage": stage, "cases": len(case_ids),
                "plan_hash": plan["plan_hash"], "provider_calls": 0,
                "reserved_test_authorized": False}

    inputs = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.inputs.jsonl")}
    candidates = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.candidates.jsonl")}
    _, _, movies = __import__(
        "experiments.scripts.data.build_redial_benchmark", fromlist=["load_catalog"]
    ).load_catalog(ROOT / "experiments/data/processed/movie_catalog.jsonl")
    transport = DeepSeekTransport()
    provider = CheckpointedProvider(output / "slots", plan["plan_hash"], transport, authorized=True)
    completed = []
    pending = []
    for case_id in case_ids:
        if only_case is not None and case_id != only_case:
            continue
        result_path = output / "cases" / (case_id.replace(":", "_") + ".json")
        if result_path.exists():
            completed.append(json.loads(result_path.read_text()))
            continue
        unresolved = [str(path) for path in (output / "slots").glob(
            case_id.replace(":", "_") + "_*.intent.json"
        ) if not path.with_name(path.name.replace(".intent.json", ".json")).exists()]
        if unresolved:
            pending.append({"case_id": case_id, "unfinished_intents": unresolved})
            print(json.dumps({"case_id": case_id, "status": "pending_unknown_billing"}), flush=True)
            continue
        evidence = bundle(inputs[case_id], candidates[case_id], movies)
        builder = RedialRequestBuilder(evidence)
        e1 = {}
        for method in E1_METHODS:
            try:
                e1[method] = {"status": "valid", "result": run_e1_method(
                    case_id, method, ROLES, 3, provider, builder.e1, builder.aggregate, budget
                )}
            except (ValueError, KeyError, TypeError) as exc:
                e1[method] = {"status": "invalid", "error": str(exc)}
        initial = {}
        for role in ROLES:
            slot = case_id.replace(":", "_") + "_E2_initial_" + role
            request = builder.e1(case_id, "multi_role", "initial", role, None)
            initial[role] = provider.complete(slot, request)
        source_ids = {row["source_id"] for row in evidence["observations"]}
        source_ids |= {row["source_id"] for row in evidence["candidates_in_shared_retrieval_order"]}
        e2 = {}
        for name, (arm, visibility) in E2_METHODS.items():
            checker = EvidenceContractChecker(case_id, source_ids) if arm == "evidence_update" else None
            try:
                e2[name] = {"status": "valid", "result": run_e2_arm(
                    case_id, initial, {role: evidence for role in ROLES}, arm, ROLES, budget,
                    visibility, provider, builder.e2_review, builder.aggregate, checker, name,
                )}
            except (ValueError, KeyError, TypeError) as exc:
                e2[name] = {"status": "invalid", "error": str(exc)}
        record = {"case_id": case_id, "evidence_hash": fingerprint(evidence),
                  "E1": e1, "E2": e2, "evaluation_labels_read": False}
        envelope = {"record": record, "integrity": fingerprint(record)}
        create(result_path, envelope)
        completed.append(envelope)
        print(json.dumps({"case_id": case_id, "status": "collected", "completed": len(completed)}), flush=True)
    summary = {"status": "completed" if not pending else "incomplete_pending_billing", "stage": stage, "cases": len(completed),
               "pending": pending,
               "plan_hash": plan["plan_hash"], "evaluation_labels_read": False,
               "reserved_test_read": False}
    summary_name = ("execution_case_" + only_case.replace(":", "_") + ".json" if only_case
                    else "execution_summary.json" if not pending else "execution_pending_summary.json")
    create(output / summary_name, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("engineering", "pilot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--only-case", help="Resume a single case in the existing frozen plan")
    parser.add_argument("--protocol", choices=("v51", "v5", "legacy-v4"), default="v51",
                        help="V5.1 expanded calibration is default; older versions are explicit reproduction")
    parser.add_argument("--phase", choices=("prepare", "calibration", "comparison"), default="prepare")
    args = parser.parse_args()
    if args.protocol == "v51":
        if args.stage != "engineering" or args.only_case:
            raise ValueError("V5.1 is limited to its frozen three-case engineering protocol")
        from experiments.paired_deliberation.run_v51 import run as run_adjusted
        result = run_adjusted(args.output, args.phase, args.execute_authorized)
    elif args.protocol == "v5":
        if args.only_case:
            raise ValueError("V5 case selection is frozen by its repair plan")
        from experiments.paired_deliberation.run_v5 import run as run_repaired
        result = run_repaired(args.stage, args.output, args.execute_authorized)
    else:
        result = run(args.stage, args.output, args.execute_authorized, args.only_case)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
