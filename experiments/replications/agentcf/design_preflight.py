"""Counterexamples and contract checks, not recommendation experiments.

All payoff numbers below are hand-constructed mathematical fixtures. No dataset,
model, API, or evidence of method effectiveness is produced by this module.
"""
import json
from itertools import product
from pathlib import Path


def pure_ne(payoffs):
    profiles = tuple(payoffs)
    return [a for a in profiles if all(
        payoffs[a][i] >= payoffs[b][i]
        for i in range(len(a)) for b in profiles
        if all(a[j] == b[j] for j in range(len(a)) if j != i))]


def difference_rewards(global_values):
    shaped = {}
    for a, value in global_values.items():
        shaped[a] = tuple(value - global_values[a[:i] + (0,) + a[i + 1:]]
                          for i in range(len(a)))
    return shaped


def run_checks():
    # A legal potential game with a poor strict local equilibrium.
    values = {(0, 0): 0.4, (1, 0): 0.3, (0, 1): 0.3, (1, 1): 0.8}
    team = {a: (v, v) for a, v in values.items()}
    diff = difference_rewards(values)
    same_gains = all(abs((diff[b][i] - diff[a][i]) - (values[b] - values[a])) < 1e-12
                     for a, b in product(values, repeat=2) for i in range(2)
                     if all(a[j] == b[j] for j in range(2) if j != i))
    # A second fixture: neither isolated information action pays for its cost,
    # while their combination is valuable. One-step greedy stops too early.
    info_value = {(0, 0): 0.0, (1, 0): -0.1, (0, 1): -0.1, (1, 1): 0.8}
    # One supported fact and one refuted fact can coexist. A repeated supporting
    # citation alone must not erase the rebuttal or count as an extra source.
    source_fixture = ["record:42", "record:42", "record:42"]
    return {
        "kind": "synthetic_mathematical_counterexamples_not_model_results",
        "model_calls": 0,
        "potential_game": {"profiles": [{"actions": list(a), "value": v} for a, v in values.items()],
                           "team_ne": [list(a) for a in pure_ne(team)],
                           "difference_ne": [list(a) for a in pure_ne(diff)],
                           "unilateral_gains_identical": same_gains,
                           "bad_ne_value": values[(0, 0)], "global_best_value": max(values.values())},
        "myopic_information_value": {"single_action_gains": [info_value[(1, 0)], info_value[(0, 1)]],
                                     "joint_gain": info_value[(1, 1)]},
        "duplicate_evidence": {"reports": len(source_fixture), "unique_sources": len(set(source_fixture))},
        "rejected_inferences": ["potential implies global optimality",
                                "difference reward removes coordination barriers",
                                "nonpositive single-step value implies no useful joint query",
                                "more role reports imply more independent evidence"]}


if __name__ == "__main__":
    result = run_checks()
    from .audit import SNAPSHOT, inference_client_probe, missing_rank_probe
    result["upstream_additional_probes"] = {
        "inference_client": inference_client_probe(SNAPSHOT / "upstream/agentcf"),
        "missing_rank": missing_rank_probe(SNAPSHOT / "upstream/agentcf")}
    result["paid_experiment_ready"] = False
    result["readiness_note"] = "Counterexample/source audit only; environment, concrete actions, scoring, budget, and new evaluation protocol not yet accepted."
    destination = Path(__file__).resolve().parents[2] / "results/replications/agentcf_2026-10-26/design_preflight.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
