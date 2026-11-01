"""Verify the independent study and recompute development scores, not test scores."""
from __future__ import annotations

import io
import json
import re
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

from .common import ROOT, OUTPUT, PROTOCOL, sha, rows, write_once, user_bucket
from .baselines import score_prediction


def main():
    data = OUTPUT / "data"
    run = OUTPUT / "development_checked"
    cfg = json.loads(PROTOCOL.read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    priors = json.loads((data / "train_priors.json").read_text())
    checks = {}
    checks["artifacts_intact"] = all(sha(data / name) == value for name, value in manifest["artifacts"].items())
    checks["protocol_frozen"] = sha(PROTOCOL) == manifest["dependencies"]["protocol"] == summary["dependencies"]["protocol"]
    checks["builder_frozen"] = sha(Path(__file__).with_name("build.py")) == manifest["dependencies"]["build_code"]
    checks["common_frozen"] = sha(Path(__file__).with_name("common.py")) == manifest["dependencies"]["common_code"] == summary["dependencies"]["common"]
    checks["method_frozen"] = sha(Path(__file__).with_name("baselines.py")) == summary["dependencies"]["baselines"]
    checks["runner_frozen"] = sha(Path(__file__).with_name("run_dev.py")) == summary["dependencies"]["runner"]
    checks["data_dependency"] = sha(data / "manifest.json") == summary["dependencies"]["data_manifest"]
    source = ROOT / "experiments/data/raw/movielens/ml-25m"
    checks["source_hashes"] = (sha(source / "ratings.csv") == manifest["dependencies"]["ratings"] and
                               sha(source / "movies.csv") == manifest["dependencies"]["movies"])
    inputs = rows(data / "dev.inputs.jsonl")
    test_inputs = rows(data / "reserved_test.inputs.jsonl")
    labels = {r["user_id"]: r for r in rows(data / "dev.labels.jsonl")}
    checks["disjoint_users"] = not ({r["user_id"] for r in inputs} & {r["user_id"] for r in test_inputs})
    checks["correct_user_partitions"] = (all(user_bucket(r["user_id"]) == 0 for r in inputs) and
                                          all(user_bucket(r["user_id"]) == 1 for r in test_inputs))
    checks["precutoff_profiles"] = all(len(r["history"]) == 40 and
                                        all(x["timestamp"] < cfg["global_cutoff_timestamp"] for x in r["history"])
                                        for r in inputs + test_inputs)
    checks["postcutoff_development_labels"] = all(x["timestamp"] >= cfg["global_cutoff_timestamp"]
                                                 for label in labels.values() for x in label["future_ratings"])
    checks["catalog_constraint"] = all(item["year"] <= cfg["latest_catalog_year"] and
                                         item["train_count"] >= cfg["minimum_train_item_ratings"]
                                         for item in priors["items"].values())
    expected = {(r["user_id"], b) for r in inputs for b in cfg["history_budgets"]}
    observed, invalid, numeric_differences = set(), [], []
    by_user = {r["user_id"]: r for r in inputs}
    for path in sorted((run / "cases").glob("*.json")):
        record = json.loads(path.read_text())
        pred = record["prediction"]
        uid, budget = pred["user_id"], pred["history_budget"]
        key = (uid, budget)
        if key in observed or key not in expected:
            invalid.append(str(path))
            continue
        observed.add(key)
        if record["dependencies"] != summary["dependencies"]:
            invalid.append(f"dependency: {uid}/{budget}")
        if record["metrics"] != score_prediction(pred, labels[uid], priors["items"]):
            invalid.append(f"score: {uid}/{budget}")
        if set(pred["candidate_ids"]) & set(by_user[uid]["seen_movie_ids"]):
            invalid.append(f"seen: {uid}/{budget}")
        if not set(pred["candidate_ids"]) <= set(priors["items"]):
            invalid.append(f"catalog: {uid}/{budget}")
        original = OUTPUT / "development/cases" / path.name
        if original.exists():
            previous = json.loads(original.read_text())["prediction"]
            if any(previous["predictions"][m]["movie_ids"] != pred["predictions"][m]["movie_ids"]
                   for m in cfg["methods"]):
                numeric_differences.append({"user_id": uid, "budget": budget})
    checks["all_cases_complete"] = observed == expected and summary["cases"] == len(expected)
    checks["scores_and_constraints_valid"] = not invalid
    checks["no_test_scoring"] = summary["reserved_test_evaluated"] is False
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("experiments.tests.test_slate_choice")
    with patch.object(socket, "create_connection", side_effect=RuntimeError("network forbidden")), \
            patch.object(socket, "getaddrinfo", side_effect=RuntimeError("DNS forbidden")):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    checks["unit_tests"] = result.wasSuccessful() and not result.skipped
    acceptance = {"checks": checks, "pass": all(checks.values()), "tests_run": result.testsRun,
                  "test_log": re.sub(r"Ran (\d+) tests in [\d.]+s", r"Ran \1 tests (duration omitted)", stream.getvalue()),
                  "cases_verified": len(observed), "invalid_cases": invalid,
                  "numeric_reimplementation_changed_slates": numeric_differences,
                  "reserved_test_labels_deserialized": False, "paid_api_calls": 0,
                  "verifier_sha256": sha(Path(__file__))}
    write_once(OUTPUT / "acceptance.json", acceptance)
    print(json.dumps({k: v for k, v in acceptance.items() if k != "test_log"}, indent=2))
    if not acceptance["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
