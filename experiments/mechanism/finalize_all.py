"""One command to verify, freeze, independently recompute, and audit all suites."""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT).returncode


def verify_status(command: list[str]) -> tuple[int, dict]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"diagnostic": (completed.stderr or completed.stdout)[-1000:]}
    return completed.returncode, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path,
                        default=HERE / "suite_registry.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    frozen_root = ROOT / "experiments/results/mechanism/frozen"
    statuses = []
    for suite in registry["suites"]:
        matrix = ROOT / suite["matrix"]
        records = [Path(path) for path in sorted(glob.glob(str(ROOT / suite["records"])))]
        if not records:
            statuses.append({"name": suite["name"], "status": "missing_records"})
            continue
        verify = [sys.executable, str(HERE / "verify_matrix.py"), str(matrix),
                  *map(str, records)]
        verify_code, verification = verify_status(verify)
        progress = {
            "completed_units": verification.get("completed_units", 0),
            "expected_units": verification.get("expected_units", 0),
            "missing_count": verification.get("missing_count", 0),
            "incomplete_count": verification.get("incomplete_count", 0),
            "unexpected_count": verification.get("unexpected_count", 0),
            "route_error_count": verification.get("route_error_count", 0),
            "schema_error_count": verification.get("schema_error_count", 0),
        }
        if verify_code:
            invalid = any(progress[key] for key in (
                "incomplete_count", "unexpected_count", "route_error_count",
                "schema_error_count",
            ))
            statuses.append({"name": suite["name"],
                             "status": "invalid" if invalid else "collecting",
                             **progress})
            continue
        bundle = frozen_root / suite["name"]
        recomputed = frozen_root / f"{suite['name']}.recomputed"
        if not args.check_only and not bundle.exists():
            if run([sys.executable, str(HERE / "freeze_suite.py"), str(matrix),
                    *map(str, records), "--name", suite["name"],
                    "--reference", suite["reference"]]):
                statuses.append({"name": suite["name"], "status": "freeze_failed"})
                continue
        if bundle.exists() and not recomputed.exists() and not args.check_only:
            if run([sys.executable, str(HERE / "recompute_frozen.py"), str(bundle)]):
                statuses.append({"name": suite["name"], "status": "recompute_failed"})
                continue
        status = "ready" if args.check_only else "frozen_and_recomputed"
        statuses.append({"name": suite["name"], "status": status, **progress})

    report = {"registry_version": registry["registry_version"], "suites": statuses}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required = [suite["name"] for suite in registry["suites"]]
    if not args.check_only and all(x["status"] == "frozen_and_recomputed" for x in statuses):
        command = [sys.executable, str(HERE / "build_frozen_manifest.py")]
        for name in required:
            command.extend(["--require", name])
        raise SystemExit(run(command))
    raise SystemExit(0 if all(x["status"] == "ready" for x in statuses) else 1)


if __name__ == "__main__":
    main()
