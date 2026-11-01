"""Zero-network acceptance for the complete paired-deliberation implementation."""
import argparse
import json
import tempfile
from pathlib import Path

from experiments.consensus_recommendation.common import create
from experiments.paired_deliberation.plan import build_plan
from experiments.paired_deliberation.pipeline import run_e1_method, run_e2_arm
from experiments.paired_deliberation.provider import CheckpointedProvider
from experiments.paired_deliberation.redial_adapter import EvidenceContractChecker


AGENTS = ("current_request", "history_applicability", "movie_fit")


def provider_record(report, prompt=3, completion=2):
    return {"report": report,
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "tool_calls": 0, "tool_units": 0, "actual_cost": "0", "retries": 0},
            "model": "offline-fixture-model", "elapsed_ms": 1}


def initial_records():
    return {agent: provider_record({"choice": agent, "claims": [{"claim_id": "c1"}], "updates": []})
            for agent in AGENTS}


def run_acceptance(automatic_packet):
    budget = {"per_call_prompt_tokens": 100, "per_call_completion_tokens": 100,
              "total_prompt_tokens": 1000, "total_completion_tokens": 1000,
              "total_tool_calls": 0}
    plan = build_plan(["offline:fixture"], "offline-fixture-model", budget, review_rounds=3)
    observations = {agent: {"available_source_ids": ["message:1"]} for agent in AGENTS}
    arm_results = {}
    e1_results = {}

    def transport(payload):
        if payload["kind"] == "aggregate":
            return provider_record({"ranking": ["m1", "m2", "m3", "m4", "m5"]})
        request = payload.get("request")
        if request is None:
            return provider_record({"choice": "offline-e1", "claims": [], "updates": []})
        agent = request["agent"]
        report = {"choice": "reviewed-" + agent, "claims": [{"claim_id": "c1"}], "updates": []}
        if request["mode"] == "evidence_update":
            report["updates"] = [{"target_claim_id": agent + ":c1", "decision": "accept",
                                  "reason": "offline structural fixture", "source_ids": ["message:1"]}]
        return provider_record(report)

    with tempfile.TemporaryDirectory() as directory:
        for method in ("single_once", "single_serial", "same_agent_samples",
                       "multi_role", "central_all_evidence"):
            provider = CheckpointedProvider(Path(directory) / ("E1_" + method), plan["plan_hash"],
                                            transport, authorized=True)
            result = run_e1_method(
                "offline:fixture", method, list(AGENTS), 3, provider,
                lambda _case, current_method, stage, identity, prior: {
                    "kind": "review", "method": current_method, "stage": stage,
                    "identity": identity, "prior": prior,
                },
                lambda _case, current_method, reports: {
                    "kind": "aggregate", "method": current_method, "reports": reports,
                }, budget,
            )
            e1_results[method] = {"records": result["ledger"]["records"],
                                  "stages": result["ledger"]["stages"],
                                  "model": result["model"]}
        for method, spec in plan["methods"].items():
            if not method.startswith("E2_"):
                continue
            provider = CheckpointedProvider(Path(directory) / method, plan["plan_hash"],
                                            transport, authorized=True)
            checker = (EvidenceContractChecker("offline:fixture", {"message:1"})
                       if spec["arm"] == "evidence_update" else None)
            result = run_e2_arm(
                "offline:fixture", initial_records(), observations, spec["arm"], list(AGENTS),
                budget, spec["visibility"], provider,
                lambda _case, request: {"kind": "review", "request": request},
                lambda _case, _arm, state: {"kind": "aggregate", "state": state}, checker,
                method,
            )
            arm_results[method] = {"review_calls": result["review_calls"],
                                   "visibility": result["visibility"],
                                   "ledger": result["ledger"], "model": result["model"]}

    packet = json.loads(Path(automatic_packet).read_text())
    synthetic = [row for case in packet["controlled_corruption_cases"] for row in case["cases"]]
    supported = [row for row in synthetic if row["verdict"] == "supported"]
    unsupported = [row for row in synthetic if row["verdict"] == "unsupported"]
    synthetic_valid = bool(supported and unsupported) and all(
        (row["mutation"] == "none") == (row["verdict"] == "supported") for row in synthetic
    )
    required = {"E2_independent", "E2_self_review", "E2_peer_review", "E2_evidence_update",
                "E2_peer_sequential", "E2_update_sequential"}
    report = {
        "version": "paired-deliberation.offline-acceptance-v1",
        "pass": set(arm_results) == required and len(e1_results) == 5 and synthetic_valid,
        "plan_hash": plan["plan_hash"],
        "e1_methods": e1_results,
        "arms": arm_results,
        "synthetic": {"cases": len(synthetic), "supported": len(supported),
                      "unsupported": len(unsupported), "integrity_pass": synthetic_valid,
                      "scope": "controlled catalog facts only"},
        "provider_calls": 0,
        "network_used": False,
        "reserved_labels_read": False,
        "human_gold_claimed": False,
        "proxy_judging_run": False,
        "next_authorized_action": "none; wait for a separate experiment-run request",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_acceptance(args.automatic_packet)
    create(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
