"""Summarize compute-controlled report-interface records."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mechanism.statistical_tests import compare_maps, holm


def safe_div(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        if row.get("completed") is True and row.get("observation"):
            grouped[(row["dataset"], row["protocol"])].append(row)
    result = {"schema_version": "report-interface-analysis.v1", "groups": []}
    for (dataset, protocol), items in sorted(grouped.items()):
        n = len(items)
        observations = [item["observation"] for item in items]
        decisions = [str(obs.get("stance") or "").lower() for obs in observations]
        truths = [str(item.get("ground_truth") or "").lower() for item in items]
        result["groups"].append({
            "dataset": dataset, "protocol": protocol, "n": n,
            "parse_success_rate": safe_div(sum(bool(x["parse_success"]) for x in observations), n),
            "stance_extractable_rate": safe_div(sum(bool(x["stance_extractable"]) for x in observations), n),
            "evidence_boundary_rate": safe_div(sum(bool(x["evidence_boundary_explicit"]) for x in observations), n),
            "task_accuracy": safe_div(sum(a == b for a, b in zip(decisions, truths)), n),
            "mean_output_tokens": safe_div(sum(int(x.get("token_output") or 0) for x in items), n),
            "mean_latency_s": safe_div(sum(float(x.get("latency_s") or 0) for x in items), n),
        })
    unit_maps = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("completed") is not True or not row.get("observation"):
            continue
        unit = (str(row["dataset"]), str(row["question_id"]), int(row["seed"]))
        protocol = row["protocol"]
        observation = row["observation"]
        unit_maps["parse_success"][protocol][unit] = bool(observation["parse_success"])
        unit_maps["stance_extractable"][protocol][unit] = bool(observation["stance_extractable"])
        unit_maps["task_accuracy"][protocol][unit] = (
            str(observation.get("stance") or "").lower()
            == str(row.get("ground_truth") or "").lower()
        )
        unit_maps["output_tokens"][protocol][unit] = int(row.get("token_output") or 0)
    comparisons = []
    for reference in ("free_text", "generic_structure"):
        item = {"left": reference, "right": "aeic"}
        for endpoint in ("parse_success", "stance_extractable", "task_accuracy"):
            item[endpoint] = compare_maps(
                unit_maps[endpoint][reference], unit_maps[endpoint]["aeic"], "binary"
            )
        item["output_tokens"] = compare_maps(
            unit_maps["output_tokens"][reference], unit_maps["output_tokens"]["aeic"],
            "continuous",
        )
        comparisons.append(item)
    for endpoint in ("parse_success", "stance_extractable", "task_accuracy", "output_tokens"):
        adjusted = holm([item[endpoint]["pvalue"] for item in comparisons])
        for item, value in zip(comparisons, adjusted):
            item[endpoint]["pvalue_holm"] = value
    result["paired_comparisons"] = comparisons
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
