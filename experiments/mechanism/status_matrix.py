"""Inspect an in-progress matrix without requiring it to be complete."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mechanism.runtime import classify_error
from experiments.mechanism.verify_matrix import expected_jobs, load_records


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.jsonl")))
        elif item.exists():
            paths.append(item)
    return paths


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("records", nargs="+", type=Path,
                        help="JSONL files or directories containing shards")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = expected_jobs(matrix)
    paths = expand_inputs(args.records)
    records = load_records(paths) if paths else {}
    progress = defaultdict(lambda: Counter(expected=0, present=0, completed=0, failed=0))
    error_classes = Counter()

    for key, job in jobs.items():
        label = job.get("label", job["mechanism"])
        group = f"{job['dataset']}::{label}"
        progress[group]["expected"] += 1
        record = records.get(key)
        if record is None:
            continue
        progress[group]["present"] += 1
        if record.get("completed") is True and not record.get("error"):
            progress[group]["completed"] += 1
        else:
            progress[group]["failed"] += 1
            error_classes[classify_error(record.get("error"))] += 1

    expected = set(jobs)
    unexpected = sorted(set(records) - expected)
    completed = sum(1 for key in expected if records.get(key, {}).get("completed") is True
                    and not records.get(key, {}).get("error"))
    failed = sum(1 for key in expected if key in records and
                 (records[key].get("completed") is not True or records[key].get("error")))
    report = {
        "matrix_version": matrix.get("matrix_version"),
        "record_files": [str(x) for x in paths],
        "expected_units": len(expected),
        "present_units": len(expected & set(records)),
        "completed_units": completed,
        "failed_units": failed,
        "missing_units": len(expected - set(records)),
        "unexpected_units": len(unexpected),
        "progress_percent": round(100.0 * completed / len(expected), 2) if expected else 0.0,
        "error_classes": dict(error_classes),
        "jobs": {name: dict(counts) for name, counts in sorted(progress.items())},
        "unexpected_preview": unexpected[:20],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        atomic_json(args.out, report)


if __name__ == "__main__":
    main()
