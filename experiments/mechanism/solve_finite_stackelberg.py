"""Enumerate a finite pure-commitment single-leader/single-follower game.

Input must contain counterfactual payoffs for EVERY action under EVERY contract.
Observed chosen-action logs alone are insufficient. Multiple agents may only be
represented as one joint follower when that modelling assumption is justified.
"""
import argparse
import json
import math
from pathlib import Path


def solve(game, tolerance=1e-10):
    actions = game["actions"]
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("unique nonempty actions required")
    contracts = game["contracts"]
    ids = [c["id"] for c in contracts]
    if not contracts or len(ids) != len(set(ids)):
        raise ValueError("unique nonempty contracts required")
    responses = []
    for contract in contracts:
        payoffs = contract["payoffs"]
        if set(payoffs) != set(actions):
            raise ValueError("all counterfactual actions must have payoffs")
        for payoff in payoffs.values():
            for key in ("leader", "follower"):
                if not math.isfinite(float(payoff[key])):
                    raise ValueError("finite payoffs required")
        maximum = max(float(p["follower"]) for p in payoffs.values())
        best = [a for a in actions if maximum - payoffs[a]["follower"] <= tolerance]
        optimistic = sorted(best, key=lambda a: (-payoffs[a]["leader"], a))[0]
        pessimistic = sorted(best, key=lambda a: (payoffs[a]["leader"], a))[0]
        responses.append({"contract_id": contract["id"], "best_responses": best,
                          "optimistic_action": optimistic, "pessimistic_action": pessimistic,
                          "optimistic_value": payoffs[optimistic]["leader"],
                          "pessimistic_value": payoffs[pessimistic]["leader"]})
    winners = {}
    for convention in ("optimistic", "pessimistic"):
        selected = sorted(responses, key=lambda r: (-r[convention + "_value"], r["contract_id"]))[0]
        winners[convention] = {"contract_id": selected["contract_id"],
                              "action": selected[convention + "_action"],
                              "leader_utility": selected[convention + "_value"],
                              "follower_regret": 0.0, "leader_regret": 0.0}
    return {"scope": "finite supplied payoff table; pure leader commitments; one follower",
            "payoff_source": game.get("payoff_source", "unspecified"),
            "responses": responses, "solutions": winners}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = solve(json.loads(args.game.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
