"""Run V5 on actual generated fact fixtures; evaluate labels after execution."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import create, fingerprint, sha
from experiments.paired_deliberation.verified_state import ClaimState, EvidenceVerifier


def execute_fixture(row, facts):
    # Only executable input fields enter the method; synthetic labels and
    # mutation names stay in the separate evaluator.
    claim = {"claim_id": "c1", "text": row["claim_text"],
             "source_ids": list(row["source_ids"]),
             "candidate_ids": [row["source_ids"][0].split(":", 1)[1]], "stance": "support"}
    state = ClaimState({"owner": {"claims": [claim]}, "peer": {"claims": []}},
                       EvidenceVerifier({item["source_id"]: item for item in facts}))
    event = state.apply("peer", {"target_claim_id": "owner:c1",
        "expected_hash": fingerprint(state.claims["owner:c1"]), "replacement": None})
    return {"event": event, "aggregation": state.aggregation_payload()}


def run(packet_path, automatic_path):
    packet = json.loads(Path(packet_path).read_text())
    automatic = json.loads(Path(automatic_path).read_text())
    facts = {case["annotation_case_id"]: case["candidate_facts"] for case in packet["cases"]}
    counts, failures, records = Counter(), [], []
    for case in automatic["controlled_corruption_cases"]:
        for row in case["cases"]:
            result = execute_fixture({"claim_text": row["claim_text"], "source_ids": row["source_ids"]},
                                     copy.deepcopy(facts[case["annotation_case_id"]]))
            # Evaluation begins only after the state transition has completed.
            expected = "keep" if row["verdict"] == "supported" else "retract"
            action = result["event"]["action"]
            passed = action == expected
            counts[row["verdict"] + ":" + action] += 1
            if not passed:
                failures.append({"case": case["annotation_case_id"], "id": row["synthetic_id"]})
            records.append({"case": case["annotation_case_id"], "id": row["synthetic_id"],
                            "expected": expected, "action": action, "pass": passed,
                            "target_hash_after": fingerprint(result["event"]["after"]),
                            "retracted_excluded_from_support": not (
                                action == "retract" and "owner:c1" in result["aggregation"]["supported_claims"])})
    return {"version": "v5-repair-controlled-fact-acceptance", "cases": len(records),
            "pass": bool(records) and not failures, "counts": dict(counts), "failures": failures,
            "records": records, "new_api_calls": 0, "evaluation_labels_available_to_method": False,
            "scope": "deterministic catalog fact state-transition acceptance, not natural-language or recommendation effectiveness",
            "natural_language_proxy_accuracy_evaluated": False, "e3_natural_data_experiment_completed": False,
            "input_sha256": {str(packet_path): sha(packet_path), str(automatic_path): sha(automatic_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet, args.automatic)
    create(args.output, result)
    print(json.dumps({key: result[key] for key in ("pass", "cases", "counts", "new_api_calls", "scope")},
                     ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
