"""Freeze the held-out test matrix from a completed dev selection report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ["StrategyQA", "MMLU-Hard", "TruthfulQA"]


def job(dataset: str, mechanism: str, label: str, reward: float = 1.0,
        cost: float = 1.0) -> dict:
    return {
        "dataset": dataset, "mechanism": mechanism, "reward_scale": reward,
        "cost_scale": cost, "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.selection.read_text(encoding="utf-8"))
    selected = report["selections"]["balanced"]["selected"]
    reward = float(selected["reward_scale"])
    cost = float(selected["cost_scale"])
    jobs = []
    for dataset in DATASETS:
        jobs.extend([
            job(dataset, "no_mechanism", "no_mechanism"),
            job(dataset, "static_aeic", "static_center", 0.2, 1.0),
            job(dataset, "static_aeic", "selected_contract", reward, cost),
        ])
    matrix = {
        "matrix_version": "selected-contract-v13.0-test",
        "purpose": "One-shot held-out validation of the balanced dev-selected finite-grid contract",
        "selection_source": str(args.selection),
        "selection_matrix_sha256": report["matrix_sha256"],
        "selected_contract": {"reward_scale": reward, "cost_scale": cost,
                              "dev_contract_id": selected["contract_id"]},
        "defaults": {
            "split": "test", "n_samples": 150, "behavior": "independent",
            "n_agents": 5, "n_rounds": 3, "seed": 20261016,
            "max_attempts": 3, "backends": ["deepseek"],
            "resource_tiered": True, "deepseek_only_tiers": True,
            "similarity": "lexical",
            "experiment_id": "aeic-stackelberg-v13-selected-contract-test",
            "out": "experiments/results/mechanism/stackelberg/selected_contract_v13_test.jsonl",
        },
        "jobs": jobs,
        "confirmatory_contrasts": [
            "selected_contract_vs_no_mechanism",
            "selected_contract_vs_static_center",
        ],
        "claim_boundary": (
            "The selected contract maximizes development-sample utility over the "
            "observed finite grid. Follower best response is unverified; this "
            "matrix cannot certify a Stackelberg equilibrium."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
