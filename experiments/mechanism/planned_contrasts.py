"""Execute exact, matrix-registered paired contrasts without post-hoc selection."""
from __future__ import annotations

import argparse
import json
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


def compare_pair(records: list[dict], left: str, right: str) -> dict:
    selected = [item for item in records if item.get("condition") in {left, right}]
    complete, units = complete_case(selected)
    present = {item.get("condition") for item in complete}
    if present != {left, right}:
        return {"n_complete_case_units": len(units), "error": "paired conditions incomplete"}
    action, decision, tokens = paired_maps(complete)
    effort, question_tokens = paired_question_maps(complete)
    return {
        "n_complete_case_units": len(units),
        "high_effort": compare_maps(action[left], action[right], "binary"),
        "decision_quality": compare_maps(decision[left], decision[right], "binary"),
        "tokens": compare_maps(tokens[left], tokens[right], "continuous"),
        "question_high_effort_rate": compare_maps(effort[left], effort[right], "continuous"),
        "question_mean_tokens": compare_maps(
            question_tokens[left], question_tokens[right], "continuous"),
    }


def build_report(records: list[dict], matrix: dict) -> dict:
    results = []
    for spec in matrix.get("contrasts", []):
        result = {**spec, **compare_pair(records, spec["left"], spec["right"])}
        results.append(result)
    valid = [item for item in results if "error" not in item]
    for endpoint in ENDPOINTS:
        adjusted = holm([item[endpoint]["pvalue"] for item in valid])
        for item, value in zip(valid, adjusted):
            item[endpoint]["pvalue_holm_across_planned_contrasts"] = value
    return {
        "matrix_version": matrix.get("matrix_version"),
        "multiplicity": "Holm across all registered contrasts within each endpoint",
        "n_registered_contrasts": len(results),
        "contrasts": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if not matrix.get("contrasts"):
        raise SystemExit("matrix has no registered contrasts")
    records = deduplicate([item for path in args.records for item in load(path)])
    report = build_report(records, matrix)
    report["records"] = [str(path) for path in args.records]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if any("error" in item for item in report["contrasts"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
