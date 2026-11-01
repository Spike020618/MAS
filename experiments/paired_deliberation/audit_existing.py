"""Audit whether historical R5 traces can serve as a paired-protocol replay.

This is deliberately an incompatibility detector, not a converter. It reads no
reserved labels and never rewrites historical records.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import OUTPUT, create, read_rows


def evidence_sources(inputs, candidates):
    return {
        case_id: (
            {row["source_id"] for row in inputs[case_id]["observations"]}
            | {"movie:" + movie_id for movie_id in candidates[case_id]["ranked_movie_ids"]}
        )
        for case_id in inputs
    }


def audit(stage="pilot"):
    data = OUTPUT / "data"
    inputs = {row["case_id"]: row for row in read_rows(data / "dev.inputs.jsonl")}
    candidates = {row["case_id"]: row for row in read_rows(data / "dev.candidates.jsonl")}
    sources = evidence_sources(inputs, candidates)
    root = OUTPUT / "collection_execution_v3" / stage / "cases"
    counts = Counter()
    per_case = []

    for path in sorted(root.glob("*.json")):
        case = json.loads(path.read_text())["record"]
        case_id = case["case_id"]
        target_roles = []
        invalid_sources = []
        malformed_updates = 0
        for review_index, review in enumerate(case["paths"]["R5"].get("reviews", [])):
            updates = review.get("updates", [])
            if len(updates) != 1:
                malformed_updates += 1
                continue
            update = updates[0]
            target = update.get("target_claim_id", "")
            role = target.split(":", 1)[0] if ":" in target else None
            target_roles.append(role)
            bad = sorted(set(update.get("source_ids", [])) - sources[case_id])
            if bad:
                invalid_sources.append({"review_index": review_index, "source_ids": bad})

        duplicate_target_role = len(target_roles) != len(set(target_roles))
        counts["cases"] += 1
        counts["reviews"] += len(case["paths"]["R5"].get("reviews", []))
        counts["cases_with_duplicate_target_role"] += duplicate_target_role
        counts["malformed_update_records"] += malformed_updates
        counts["cases_with_out_of_evidence_source"] += bool(invalid_sources)
        per_case.append(
            {
                "case_id": case_id,
                "target_roles": target_roles,
                "duplicate_target_role": duplicate_target_role,
                "malformed_update_records": malformed_updates,
                "out_of_evidence_sources": invalid_sources,
            }
        )

    blockers = []
    if counts["cases_with_duplicate_target_role"]:
        blockers.append(
            "Historical reviews are centralized review slots and do not identify distinct agent-owned updates."
        )
    if counts["malformed_update_records"]:
        blockers.append("Some historical review slots do not contain exactly one executable update.")
    if counts["cases_with_out_of_evidence_source"]:
        blockers.append("Some update references fall outside the frozen evidence bundle.")
    blockers.extend(
        [
            "No independent semantic-support labels exist for accept/reject decisions.",
            "Historical R5 aggregation consumes review text; it is not the new state-transition adapter.",
        ]
    )
    report = {
        "version": "paired-deliberation.historical-compatibility-v1",
        "stage": stage,
        "counts": dict(counts),
        "compatible_as_new_e2_effect_replay": False,
        "reusable_as": [
            "historical failure evidence",
            "schema and source-reference fixtures",
            "cost-distribution calibration",
        ],
        "not_reusable_as": [
            "semantic truth labels",
            "new frozen-state transition results",
            "evidence that the proposed mechanism is effective",
        ],
        "blockers": blockers,
        "reserved_labels_read": False,
        "new_model_calls": 0,
        "per_case": per_case,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("engineering", "pilot"), default="pilot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.stage)
    if args.output:
        create(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
