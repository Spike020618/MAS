"""Paired AEIC-MD inference separately within each dataset."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .statistical_tests import (
        compare_maps, complete_case, deduplicate, holm, load, paired_maps,
        paired_question_maps,
    )
except ImportError:  # Direct script execution.
    from statistical_tests import (
        compare_maps, complete_case, deduplicate, holm, load, paired_maps,
        paired_question_maps,
    )


ENDPOINTS = ("high_effort", "decision_quality", "tokens",
             "question_high_effort_rate", "question_mean_tokens")


def compare_dataset(records: list[dict], reference: str) -> dict:
    complete, units = complete_case(records)
    conditions_present = {item.get("condition") for item in complete}
    if reference not in conditions_present:
        return {"n_complete_case_units": len(units), "comparisons": [],
                "error": "reference condition not found"}
    action, decision, tokens = paired_maps(complete)
    question_effort, question_tokens = paired_question_maps(complete)
    comparisons = []
    for condition in sorted(set(action) - {reference}):
        comparisons.append({
            "condition": condition,
            "vs": reference,
            "high_effort": compare_maps(action[reference], action[condition], "binary"),
            "decision_quality": compare_maps(decision[reference], decision[condition], "binary"),
            "tokens": compare_maps(tokens[reference], tokens[condition], "continuous"),
            "question_high_effort_rate": compare_maps(
                question_effort[reference], question_effort[condition], "continuous"),
            "question_mean_tokens": compare_maps(
                question_tokens[reference], question_tokens[condition], "continuous"),
        })
    for endpoint in ENDPOINTS:
        adjusted = holm([item[endpoint]["pvalue"] for item in comparisons])
        for item, value in zip(comparisons, adjusted):
            item[endpoint]["pvalue_holm_within_dataset"] = value
    return {"n_complete_case_units": len(units), "comparisons": comparisons}


def build_report(records: list[dict], reference: str) -> dict:
    groups = defaultdict(list)
    for item in records:
        groups[str(item.get("dataset"))].append(item)
    return {
        "reference": reference,
        "multiplicity": "Holm within each dataset and endpoint",
        "datasets": {
            dataset: compare_dataset(items, reference)
            for dataset, items in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = deduplicate([item for path in args.records for item in load(path)])
    report = build_report(records, args.reference)
    report["records"] = [str(path) for path in args.records]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
