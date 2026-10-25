"""Certify necessary one-stage best-response conditions from payoff bounds.

This is a rejection certificate, not an estimator of counterfactual outcomes.
It assumes the published one-stage credit-minus-cost objective; continuation
values or unmodelled intrinsic rewards cannot be inferred from these logs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def audit(records):
    groups = defaultdict(lambda: {"n": 0, "strictly_dominated_choices": 0,
                                  "mean_regret_lower_bound": 0.0, "bounds": {}})
    for row in records:
        if not row.get("completed") or row.get("error"):
            continue
        for item in row["rounds"]:
            offer = item["offered"]
            if offer["credit_rule"] != "fixed_process_quality":
                raise ValueError("certificate supports fixed_process_quality only")
            rewards = offer["visible_reward"]
            if any(v < 0 for v in rewards.values()):
                raise ValueError("nonnegative reward coefficients required")
            costs = offer["visible_cost"]
            # Both recorded process signals lie in [0,1], for every action.
            maximum_credit = sum(rewards.values())
            low_lower = -costs["low_effort"]
            upper = {a: maximum_credit - c for a, c in costs.items()}
            action = item["chosen_action"]
            gap = max(0.0, low_lower - upper[action])
            g = groups[row["condition"]]
            g["n"] += 1
            g["strictly_dominated_choices"] += int(gap > 1e-12)
            g["mean_regret_lower_bound"] += gap
            g["bounds"] = {"low_utility_lower_bound": low_lower,
                           "action_utility_upper_bounds": upper}
    for g in groups.values():
        g["mean_regret_lower_bound"] /= g["n"]
        g["dominated_choice_rate"] = g["strictly_dominated_choices"] / g["n"]
        g["necessary_best_response_check_pass"] = not g["strictly_dominated_choices"]
    return {"scope": "one-stage published payoff; no inferred continuation value",
            "interpretation": "positive lower bound rejects exact best response; zero does not prove equilibrium",
            "contracts": dict(groups)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("records", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    records = [json.loads(line) for path in args.records
               for line in path.read_text().splitlines() if line.strip()]
    report = audit(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
