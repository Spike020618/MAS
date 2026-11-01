"""Offline control acceptance: score replay plus exact-objective witness checks."""
from __future__ import annotations

import io
import json
import re
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .common import OUTPUT, PROTOCOL, sha, rows, write_once, sample_key
from .controls import CFRecommender, capture_values
from .baselines import Recommender, score_prediction
from .run_controls import dependencies
from .train_cf import CF_OUTPUT, CONTROL


def main():
    data, controls = OUTPUT / "data", OUTPUT / "controls"
    cfg, ctrl = json.loads(PROTOCOL.read_text()), json.loads(CONTROL.read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    cf_manifest = json.loads((CF_OUTPUT / "manifest.json").read_text())
    cf_summary = json.loads((controls / "cf_development/summary.json").read_text())
    exact_summary = json.loads((controls / "exact_diagnostic/summary.json").read_text())
    priors = json.loads((data / "train_priors.json").read_text())["items"]
    inputs = rows(data / "dev.inputs.jsonl")
    labels = {r["user_id"]: r for r in rows(data / "dev.labels.jsonl")}
    by_user = {r["user_id"]: r for r in inputs}
    dep = dependencies()
    checks = {"source_artifacts_intact": all(sha(data / name) == value for name, value in manifest["artifacts"].items()),
              "cf_features_intact": sha(CF_OUTPUT / "features.npz") == cf_manifest["features_sha256"],
              "trainer_frozen": sha(Path(__file__).with_name("train_cf.py")) == cf_manifest["dependencies"]["code"],
              "control_protocol_frozen": sha(CONTROL) == cf_manifest["dependencies"]["controls"],
              "cf_run_frozen": cf_summary["dependencies"] == dep,
              "exact_run_frozen": exact_summary["dependencies"] == dep,
              "exact_complete": exact_summary["complete"],
              "test_not_scored": not cf_summary["reserved_test_scored"] and not exact_summary["reserved_test_scored"]}
    cf_records, errors = {}, []
    for path in (controls / "cf_development/cases").glob("*.json"):
        record = json.loads(path.read_text())
        pred = record["prediction"]
        key = (pred["user_id"], pred["history_budget"])
        if key in cf_records or record["dependencies"] != dep:
            errors.append(f"CF key/dependency {key}")
        cf_records[key] = record
        if record["metrics"] != score_prediction(pred, labels[key[0]], priors):
            errors.append(f"CF metrics {key}")
        if set(pred["candidate_ids"]) & set(by_user[key[0]]["seen_movie_ids"]):
            errors.append(f"CF seen item {key}")
    expected = {(r["user_id"], b) for r in inputs for b in cfg["history_budgets"]}
    checks["cf_cases_complete"] = set(cf_records) == expected
    checks["cf_summary_reproduced"] = all(
        abs(cf_summary["budgets"][str(b)]["methods"][m][metric] - float(np.mean([
            cf_records[(r["user_id"], b)]["metrics"]["methods"][m][metric] for r in inputs]))) < 1e-12
        for b in cfg["history_budgets"] for m in cfg["methods"] for metric in ("recall_at_5", "hit_at_5", "ndcg_at_5"))
    models = {"genre": Recommender(priors, cfg),
              "collaborative": CFRecommender(priors, cfg, CF_OUTPUT / "features.npz")}
    sampled = sorted(inputs, key=lambda r: sample_key(r["user_id"]))[:ctrl["exact_max_users"]]
    exact_expected = {(r["user_id"], b, name) for r in sampled for b in cfg["history_budgets"] for name in models}
    exact_observed, outcomes = set(), {}
    for path in sorted((controls / "exact_diagnostic/cases").glob("*.json")):
        record = json.loads(path.read_text())
        pred, solved = record["prediction"], record["solver"]
        uid, budget, name = pred["user_id"], pred["history_budget"], record["representation"]
        key = (uid, budget, name)
        if key in exact_observed or key not in exact_expected or record["dependencies"] != dep:
            errors.append(f"exact key/dependency {key}")
        exact_observed.add(key)
        replay, matrices = capture_values(models[name], by_user[uid], budget)
        if replay["candidate_ids"] != pred["candidate_ids"]:
            errors.append(f"exact candidate replay {key}")
        original = replay["predictions"]["greedy_minimax_regret"]["movie_ids"]
        if original != pred["predictions"]["greedy_minimax_regret"]["movie_ids"]:
            errors.append(f"greedy replay {key}")
        exact_ids = [pred["candidate_ids"][j] for j in solved["chosen"]]
        if exact_ids != pred["predictions"]["exact_minimax_regret"]["movie_ids"] or len(set(exact_ids)) != cfg["top_k"]:
            errors.append(f"exact witness alignment {key}")
        matrix = matrices["values"]
        achieved = float((matrix.max(axis=1) - matrix[:, solved["chosen"]].max(axis=1)).max())
        tolerance = 2 * ctrl["solver_tolerance"]
        if abs(achieved - solved["regret"]) > tolerance or achieved > solved["initial_regret"] + tolerance:
            errors.append(f"exact objective replay {key}")
        if solved["global_optimum_certified_to_tolerance"] and solved["upper_bound"] - solved["lower_bound"] > tolerance:
            errors.append(f"certificate gap {key}")
        if record["metrics"] != score_prediction(pred, labels[uid], priors):
            errors.append(f"exact metric replay {key}")
        group = outcomes.setdefault(f"{name}_{budget}", {"objective_improved": [], "objective_unchanged": []})
        improved = solved["initial_regret"] - solved["regret"] > ctrl["solver_tolerance"]
        delta = record["metrics"]["methods"]["exact_minimax_regret"]["recall_at_5"] - record["metrics"]["methods"]["greedy_minimax_regret"]["recall_at_5"]
        group["objective_improved" if improved else "objective_unchanged"].append(delta)
        if len(exact_observed) % 100 == 0:
            print(f"verified exact witnesses {len(exact_observed)}/{len(exact_expected)}", flush=True)
    checks["exact_cases_complete"] = exact_observed == exact_expected
    checks["case_replay_valid"] = not errors
    associations = {name: {kind: {"cases": len(values), "recall_difference_mean": float(np.mean(values)) if values else None,
                                 "improved_users": sum(v > 0 for v in values), "harmed_users": sum(v < 0 for v in values)}
                            for kind, values in groups.items()} for name, groups in outcomes.items()}
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("experiments.tests.test_slate_controls")
    with patch.object(socket, "create_connection", side_effect=RuntimeError("network forbidden")), \
            patch.object(socket, "getaddrinfo", side_effect=RuntimeError("DNS forbidden")):
        test = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    checks["tests_pass"] = test.wasSuccessful() and not test.skipped
    result = {"checks": checks, "pass": all(checks.values()), "cf_cases": len(cf_records),
              "exact_cases": len(exact_observed), "errors": errors, "tests_run": test.testsRun,
              "objective_vs_external_recall": associations,
              "test_log": re.sub(r"Ran (\d+) tests in [\d.]+s", r"Ran \1 tests (duration omitted)", stream.getvalue()),
              "paid_api_calls": 0, "reserved_test_scored": False, "verifier_sha256": sha(Path(__file__))}
    write_once(controls / "acceptance.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"test_log", "objective_vs_external_recall"}}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
