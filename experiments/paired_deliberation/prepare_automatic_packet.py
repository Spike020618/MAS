"""Build deterministic and controlled-corruption evaluation fixtures."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import create
from experiments.paired_deliberation.automatic_evaluation import (
    deterministic_contract_audit,
    synthetic_fact_cases,
)


def prepare(annotation_packet):
    packet = json.loads(Path(annotation_packet).read_text())
    cases = []
    deterministic = []
    for case in packet["cases"]:
        source_ids = {row["source_id"] for row in case["observations"]}
        source_ids |= {row["source_id"] for row in case["candidate_facts"]}
        candidate_ids = {row["movie_id"] for row in case["candidate_facts"]}
        for claim in case["claims"]:
            deterministic.append({
                "annotation_id": claim["annotation_id"],
                **deterministic_contract_audit(
                    {"text": claim["claim_text"], "source_ids": claim["cited_source_ids"],
                     "candidate_ids": claim["candidate_ids"]},
                    source_ids, candidate_ids,
                ),
            })
        generated = synthetic_fact_cases(case["candidate_facts"], maximum=24)
        cases.append({"annotation_case_id": case["annotation_case_id"], **generated})
    return {
        "version": "paired-deliberation.automatic-evaluation-packet-v1",
        "deterministic_contract_results": deterministic,
        "controlled_corruption_cases": cases,
        "proxy_judging": {
            "status": "not_run",
            "required_panel": "two blinded independent judges plus a third only on disagreement",
            "label_type": "model_proxy_not_human_gold",
        },
        "human_gold_claimed": False,
        "reserved_labels_read": False,
        "new_model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.annotation_packet)
    create(args.output, result)
    print(json.dumps({
        "deterministic_claims": len(result["deterministic_contract_results"]),
        "deterministic_valid": sum(row["valid"] for row in result["deterministic_contract_results"]),
        "synthetic_cases": sum(len(case["cases"]) for case in result["controlled_corruption_cases"]),
        "new_model_calls": 0,
    }))


if __name__ == "__main__":
    main()
