"""Per-domain and worst-group summaries for AEIC-MD generalization claims."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .analyze import deduplicate, load_records, summarize_condition
except ImportError:  # Direct script execution.
    from analyze import deduplicate, load_records, summarize_condition


ENDPOINTS = (
    "decision_quality", "high_effort_choice_rate", "mean_evidence_novelty",
    "unsupported_flip_rate", "mean_tokens_per_agent_round",
)


def build_report(records: list[dict], reference: str | None = None) -> dict:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in records:
        if item.get("completed") and not item.get("error"):
            groups[(str(item.get("dataset")), str(item.get("condition")))].append(item)
    cells = {
        f"{dataset}::{condition}": summarize_condition(items)
        for (dataset, condition), items in sorted(groups.items())
    }
    deltas = {}
    if reference:
        datasets = sorted({dataset for dataset, _ in groups})
        conditions = sorted({condition for _, condition in groups if condition != reference})
        for condition in conditions:
            per_dataset = {}
            for dataset in datasets:
                left = cells.get(f"{dataset}::{reference}")
                right = cells.get(f"{dataset}::{condition}")
                if not left or not right:
                    continue
                per_dataset[dataset] = {
                    endpoint: (
                        None if left.get(endpoint) is None or right.get(endpoint) is None
                        else right[endpoint] - left[endpoint]
                    ) for endpoint in ENDPOINTS
                }
            if per_dataset:
                worst = {}
                for endpoint in ENDPOINTS:
                    values = [(dataset, value[endpoint]) for dataset, value in per_dataset.items()
                              if value[endpoint] is not None]
                    if values:
                        dataset, value = min(values, key=lambda pair: pair[1])
                        worst[endpoint] = {"dataset": dataset, "delta": value}
                deltas[condition] = {"by_dataset": per_dataset, "worst_delta": worst}
    return {"reference": reference, "dataset_conditions": cells, "deltas": deltas}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--reference")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = deduplicate([
        item for path in args.records for item in load_records(path)
    ])
    report = build_report(records, args.reference)
    report["records"] = [str(path) for path in args.records]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
