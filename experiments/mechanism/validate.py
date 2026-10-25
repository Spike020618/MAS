"""Validate AEIC-MD JSONL records and report pilot-gate readiness."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mechanism.runtime import classify_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero on any failed or incomplete record")
    args = parser.parse_args()

    total = completed = errors = parseable = cost_complete = 0
    actions = Counter()
    error_classes = Counter()
    keys = Counter()
    for line in args.path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        item = json.loads(line)
        key = (item.get("experiment_id"), item.get("dataset"), item.get("condition"),
               item.get("question_id"), item.get("seed"))
        keys[key] += 1
        if item.get("error"):
            errors += 1
            error_classes[classify_error(item.get("error"))] += 1
            continue
        if item.get("completed"):
            completed += 1
        rounds = item.get("rounds") or []
        if rounds and all(all(k in r.get("report", {}) for k in ("A", "E", "I", "C")) for r in rounds):
            parseable += 1
        if rounds and all(r.get("token_input") is not None and
                          r.get("token_output") is not None and
                          r.get("latency_s") is not None for r in rounds):
            cost_complete += 1
        actions.update(r.get("chosen_action") for r in rounds if r.get("chosen_action"))

    duplicate_rows = sum(n - 1 for n in keys.values() if n > 1)
    report = {
        "records": total,
        "completed": completed,
        "errors": errors,
        "duplicate_rows": duplicate_rows,
        "parse_rate": parseable / total if total else 0.0,
        "cost_log_completeness": cost_complete / total if total else 0.0,
        "action_counts": dict(actions),
        "error_classes": dict(error_classes),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if duplicate_rows:
        raise SystemExit(2)
    if args.strict and (errors or completed != total or parseable != completed or
                        cost_complete != completed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
