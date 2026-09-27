"""Combine preregistered boundary extremes with the frozen dev center point."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from .analyze import deduplicate, load_records, summarize_condition
except ImportError:  # Direct script execution.
    from analyze import deduplicate, load_records, summarize_condition


REWARD = re.compile(r"__static_aeic(?:@([0-9.]+))?")
COST = re.compile(r"__cost@([0-9.]+)")


def parameters(condition: str) -> tuple[float, float]:
    reward_match = REWARD.search(condition)
    if not reward_match:
        raise ValueError(f"not a static AEIC condition: {condition}")
    reward = float(reward_match.group(1) or 1.0)
    cost_match = COST.search(condition)
    return reward, float(cost_match.group(1)) if cost_match else 1.0


def build_report(boundary: list[dict], center: list[dict]) -> dict:
    selected_center = [
        item for item in center
        if item.get("condition") == "independent__static_aeic@0.2"
        and item.get("completed") and not item.get("error")
    ]
    records = deduplicate(boundary + selected_center)
    groups = defaultdict(list)
    for item in records:
        condition = str(item.get("condition"))
        if "__static_aeic" not in condition or not item.get("completed") or item.get("error"):
            continue
        reward, cost = parameters(condition)
        groups[(str(item.get("dataset")), reward, cost)].append(item)
    cells = {}
    for (dataset, reward, cost), items in sorted(groups.items()):
        cells[f"{dataset}::reward={reward:g}::cost={cost:g}"] = summarize_condition(items)
    diagnostics = {}
    for dataset in sorted({key[0] for key in groups}):
        reward_rates = [(reward, summarize_condition(groups[(dataset, reward, 1.0)])["high_effort_choice_rate"])
                        for reward in (0.05, 0.2, 0.5) if (dataset, reward, 1.0) in groups]
        cost_rates = [(cost, summarize_condition(groups[(dataset, 0.2, cost)])["high_effort_choice_rate"])
                      for cost in (0.5, 1.0, 1.5) if (dataset, 0.2, cost) in groups]
        diagnostics[dataset] = {
            "reward_response": reward_rates,
            "reward_monotonic_non_decreasing": len(reward_rates) == 3 and all(
                b[1] >= a[1] for a, b in zip(reward_rates, reward_rates[1:])),
            "cost_response": cost_rates,
            "cost_monotonic_non_increasing": len(cost_rates) == 3 and all(
                b[1] <= a[1] for a, b in zip(cost_rates, cost_rates[1:])),
        }
    return {"cells": cells, "diagnostics": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boundary", type=Path, nargs="+")
    parser.add_argument("--center", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    boundary = [item for path in args.boundary for item in load_records(path)]
    center = [item for path in args.center for item in load_records(path)]
    report = build_report(boundary, center)
    report["boundary_sources"] = [str(path) for path in args.boundary]
    report["center_sources"] = [str(path) for path in args.center]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
