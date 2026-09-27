"""Paired inferential tests for AEIC-MD pilot/formal JSONL records."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.stats import binomtest, wilcoxon


HIGH_EFFORT = {"aeic_effort", "verify"}


def holm(pvalues: List[float]) -> List[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    n = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def mcnemar_exact(left: List[bool], right: List[bool]) -> dict:
    if len(left) != len(right):
        raise ValueError("paired vectors have different lengths")
    b = sum(a and not c for a, c in zip(left, right))
    c = sum((not a) and c for a, c in zip(left, right))
    discordant = b + c
    pvalue = binomtest(min(b, c), discordant, 0.5).pvalue if discordant else 1.0
    return {"left_only": b, "right_only": c, "discordant": discordant,
            "pvalue": float(pvalue)}


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def deduplicate(records: List[dict]) -> List[dict]:
    latest = {}
    for item in records:
        key = (item.get("experiment_id"), item.get("dataset"), item.get("condition"),
               str(item.get("question_id")), item.get("seed"))
        latest[key] = item
    return list(latest.values())


def complete_case(records: List[dict]):
    conditions = sorted({x.get("condition") for x in records})
    valid = {
        condition: {(x.get("dataset"), str(x.get("question_id")), int(x.get("seed", -1))) for x in records
                    if x.get("condition") == condition and x.get("completed") and not x.get("error")}
        for condition in conditions
    }
    common = set.intersection(*valid.values()) if valid else set()
    return [x for x in records
            if (x.get("dataset"), str(x.get("question_id")), int(x.get("seed", -1))) in common], common


def paired_maps(records: List[dict]):
    action = defaultdict(dict)
    decision = defaultdict(dict)
    tokens = defaultdict(dict)
    for record in records:
        if not record.get("completed") or record.get("error"):
            continue
        condition = record["condition"]
        unit = (str(record.get("dataset")), str(record["question_id"]),
                int(record.get("seed", -1)))
        decision[condition][unit] = record.get("final_decision") == record.get("ground_truth")
        for item in record.get("rounds", []):
            key = (unit, int(item["round_id"]), int(item["agent_id"]))
            action[condition][key] = item.get("chosen_action") in HIGH_EFFORT
            tokens[condition][key] = (item.get("token_input") or 0) + (item.get("token_output") or 0)
    return action, decision, tokens


def paired_question_maps(records: List[dict]):
    effort = defaultdict(dict)
    tokens = defaultdict(dict)
    for record in records:
        if not record.get("completed") or record.get("error"):
            continue
        condition = record["condition"]
        unit = (str(record.get("dataset")), str(record["question_id"]),
                int(record.get("seed", -1)))
        rounds = record.get("rounds", [])
        if not rounds:
            continue
        effort[condition][unit] = sum(
            item.get("chosen_action") in HIGH_EFFORT for item in rounds
        ) / len(rounds)
        tokens[condition][unit] = sum(
            (item.get("token_input") or 0) + (item.get("token_output") or 0)
            for item in rounds
        ) / len(rounds)
    return effort, tokens


def compare_maps(left: Dict, right: Dict, kind: str) -> dict:
    keys = sorted(set(left) & set(right))
    if kind == "binary":
        a, b = [bool(left[k]) for k in keys], [bool(right[k]) for k in keys]
        result = mcnemar_exact(a, b)
        result.update({"n_pairs": len(keys), "left_rate": sum(a) / len(a),
                       "right_rate": sum(b) / len(b),
                       "delta": (sum(b) - sum(a)) / len(a)})
        return result
    a, b = [float(left[k]) for k in keys], [float(right[k]) for k in keys]
    differences = [y - x for x, y in zip(a, b)]
    if not differences or all(x == 0 for x in differences):
        statistic, pvalue = 0.0, 1.0
    else:
        statistic, pvalue = wilcoxon(differences, alternative="two-sided")
    return {"n_pairs": len(keys), "left_mean": sum(a) / len(a),
            "right_mean": sum(b) / len(b), "mean_delta": sum(differences) / len(differences),
            "statistic": float(statistic), "pvalue": float(pvalue)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--reference", default="independent__no_mechanism")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--condition", action="append", default=[],
                        help="Include only this exact condition (repeatable)")
    args = parser.parse_args()
    raw = deduplicate([item for path in args.records for item in load(path)])
    if args.condition:
        wanted = set(args.condition)
        raw = [item for item in raw if item.get("condition") in wanted]
    records, common_units = complete_case(raw)
    if args.reference not in {item.get("condition") for item in records}:
        raise SystemExit(f"reference condition not found: {args.reference}")
    action, decision, tokens = paired_maps(records)
    question_effort, question_tokens = paired_question_maps(records)
    conditions = sorted(set(action) - {args.reference})
    comparisons = []
    for condition in conditions:
        comparisons.append({
            "condition": condition,
            "vs": args.reference,
            "high_effort": compare_maps(action[args.reference], action[condition], "binary"),
            "decision_quality": compare_maps(decision[args.reference], decision[condition], "binary"),
            "tokens": compare_maps(tokens[args.reference], tokens[condition], "continuous"),
            "question_high_effort_rate": compare_maps(
                question_effort[args.reference], question_effort[condition], "continuous"
            ),
            "question_mean_tokens": compare_maps(
                question_tokens[args.reference], question_tokens[condition], "continuous"
            ),
        })
    for endpoint in ("high_effort", "decision_quality", "tokens",
                     "question_high_effort_rate", "question_mean_tokens"):
        adjusted = holm([x[endpoint]["pvalue"] for x in comparisons])
        for item, value in zip(comparisons, adjusted):
            item[endpoint]["pvalue_holm"] = value
    result = {"records": [str(x) for x in args.records], "reference": args.reference,
              "n_complete_case_units": len(common_units),
              "comparisons": comparisons}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
