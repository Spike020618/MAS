"""Offline threshold sensitivity from immutable AEIC-MD question-level records."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.core.mechanism.behavioral_metrics import stance_flip, unsupported_flip
from experiments.core.mechanism.schema import AEICReport


def report_from(item: dict) -> AEICReport:
    report = item.get("report") or {}
    return AEICReport(
        str(report.get("A", "")), [str(x) for x in report.get("E", [])],
        str(report.get("I", "")), str(report.get("C", "")),
    )


def threshold_counts(records: list[dict], thresholds: list[float]) -> dict:
    cells = defaultdict(lambda: {
        str(value): {"flip_opportunities": 0, "unsupported_flips": 0}
        for value in thresholds
    })
    for record in records:
        if record.get("completed") is not True or record.get("error"):
            continue
        prior = {}
        cell = f"{record.get('dataset')}::{record.get('condition')}"
        for item in sorted(record.get("rounds", []),
                           key=lambda x: (int(x.get("round_id", 0)), int(x.get("agent_id", 0)))):
            agent = int(item.get("agent_id", 0))
            current = report_from(item)
            previous = prior.get(agent)
            if previous is not None and stance_flip(previous.get("stance"), item.get("stance")):
                visible = [
                    str(e) for peer in item.get("visible_peer_reports", [])
                    for e in peer.get("E", [])
                ]
                for threshold in thresholds:
                    bucket = cells[cell][str(threshold)]
                    bucket["flip_opportunities"] += 1
                    bucket["unsupported_flips"] += int(unsupported_flip(
                        previous["report"], current, previous.get("stance"),
                        item.get("stance"), visible, threshold,
                    ))
            prior[agent] = {"report": current, "stance": item.get("stance")}
    for values in cells.values():
        for bucket in values.values():
            denominator = bucket["flip_opportunities"]
            bucket["unsupported_flip_rate"] = (
                bucket["unsupported_flips"] / denominator if denominator else 0.0
            )
    return dict(sorted(cells.items()))


def cost_weight_values(records: list[dict], weights: list[float]) -> dict:
    cells = defaultdict(lambda: {str(value): [] for value in weights})
    for record in records:
        if record.get("completed") is not True or record.get("error"):
            continue
        rounds = record.get("rounds", [])
        if not rounds:
            continue
        correct = float(record.get("final_decision") == record.get("ground_truth"))
        mean_cost = sum(float(item.get("realized_cost") or 0.0) for item in rounds) / len(rounds)
        cell = f"{record.get('dataset')}::{record.get('condition')}"
        for weight in weights:
            cells[cell][str(weight)].append(correct - weight * mean_cost)
    result = {}
    for cell, values in sorted(cells.items()):
        result[cell] = {}
        for weight, observations in values.items():
            result[cell][weight] = {
                "n_questions": len(observations),
                "mean_net_value": (sum(observations) / len(observations)
                                   if observations else 0.0),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--threshold", action="append", type=float,
                        default=None, help="Repeatable; defaults to .10/.25/.40")
    parser.add_argument("--cost-weight", action="append", type=float,
                        default=None, help="Repeatable; defaults to 0/.05/.10/.20")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    thresholds = args.threshold or [0.10, 0.25, 0.40]
    cost_weights = args.cost_weight or [0.0, 0.05, 0.10, 0.20]
    if any(not 0 <= value <= 1 for value in thresholds):
        parser.error("thresholds must be between 0 and 1")
    if any(value < 0 for value in cost_weights):
        parser.error("cost weights must be nonnegative")
    records = []
    for path in args.records:
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                       if line.strip())
    result = {
        "records": [str(path) for path in args.records],
        "thresholds": thresholds,
        "unsupported_flip": threshold_counts(records, thresholds),
        "cost_weights": cost_weights,
        "net_value": cost_weight_values(records, cost_weights),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
