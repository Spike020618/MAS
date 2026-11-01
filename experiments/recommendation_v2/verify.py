"""Offline acceptance for the conservative-prefix preparation and data audit."""
from __future__ import annotations

import argparse
import io
import json
import re
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

from .prepare import ROOT, DEFAULT_OUTPUT, digest, save, snapshot_legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
                        default=ROOT / "experiments/results/recommendation_v2/audit_verified")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/results/recommendation_v2/acceptance_verified.json")
    args = parser.parse_args()
    checks = {}
    manifest = json.loads((DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    checks["prepared_files_hashes"] = all(digest(DEFAULT_OUTPUT / name) == meta["sha256"]
                                         for name, meta in manifest["outputs"].items())
    checks["builder_hash"] = digest(Path(__file__).with_name("prepare.py")) == manifest["code_sha256"]
    checks["raw_redial_hashes"] = all(
        digest(ROOT / f"experiments/data/raw/redial/{split}.jsonl") == value
        for split, value in manifest["sources"].items())
    checks["catalog_hash"] = digest(ROOT / "experiments/data/processed/movie_catalog.jsonl") == manifest["catalog_sha256"]
    legacy = json.loads((DEFAULT_OUTPUT / "legacy_snapshot.json").read_text(encoding="utf-8"))
    checks["legacy_files_unchanged"] = snapshot_legacy() == legacy
    reports = {}
    for name in ("redial_integrity", "retrieval_coverage", "movielens_inventory"):
        report = json.loads((args.audit / f"{name}.json").read_text(encoding="utf-8"))
        reports[name] = report
        checks[f"{name}_code_hash"] = report["dependencies"]["audit_code"] == digest(Path(__file__).with_name("audit.py"))
        if name == "movielens_inventory":
            checks["movielens_sources"] = all(
                digest(ROOT / "experiments/data/raw/movielens/ml-25m" / key) == value
                for key, value in report["dependencies"].items() if key != "audit_code")
        else:
            checks[f"{name}_manifest"] = report["dependencies"]["prepared_manifest"] == digest(DEFAULT_OUTPUT / "manifest.json")
        checks[f"{name}_artifacts"] = all(digest(args.audit / key) == value
                                          for key, value in report.get("artifacts", {}).items())
    checks["prefix_reconstruction_pass"] = reports["redial_integrity"]["pass"]
    checks["ratings_valid_and_unique"] = all(reports["movielens_inventory"][key] == 0 for key in
                                             ("invalid_rows", "key_order_violations", "adjacent_duplicate_user_movie"))
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("experiments.tests.test_recommendation_v2")
    with patch.object(socket, "create_connection", side_effect=RuntimeError("network forbidden")), \
            patch.object(socket, "getaddrinfo", side_effect=RuntimeError("DNS forbidden")):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    checks["tests_pass_no_skips"] = result.wasSuccessful() and not result.skipped
    payload = {"version": "recommendation-prefix-audit.acceptance.v1", "checks": checks,
               "pass": all(checks.values()), "legacy_files_checked": len(legacy),
               "tests_run": result.testsRun,
               "test_log": re.sub(r"Ran (\d+) tests in [\d.]+s", r"Ran \1 tests (duration omitted)", stream.getvalue()),
               "audit_directory": str(args.audit.relative_to(ROOT)),
               "model_calls": 0, "scope": "P0/P1 conservative reconstruction and P2 inventory/coverage only",
               "not_claimed": ["fresh independent test", "user-level train/dev isolation",
                               "sentiment/constraint extraction", "ready personalized benchmark",
                               "new mechanism or recommendation quality advantage"],
               "verifier_sha256": digest(Path(__file__))}
    save(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "test_log"}, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
