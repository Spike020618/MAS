"""Verify that a frozen collection matrix is complete and internally consistent."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.mechanism.schema import record_from_dict


def condition_name(behavior: str, mechanism: str, reward_scale: float,
                   cost_scale: float = 1.0, signal_mask: str | None = None,
                   history_policy: str = "cumulative",
                   variant_tag: str | None = None) -> str:
    suffix = f"@{reward_scale:g}" if reward_scale != 1.0 else ""
    cost_suffix = f"__cost@{cost_scale:g}" if cost_scale != 1.0 else ""
    signal_suffix = f"__signals@{signal_mask}" if signal_mask else ""
    history_suffix = (f"__history@{history_policy}"
                      if history_policy != "cumulative" else "")
    variant_suffix = f"__variant@{variant_tag}" if variant_tag else ""
    return (f"{behavior}__{mechanism}{suffix}{cost_suffix}{signal_suffix}"
            f"{history_suffix}{variant_suffix}")


def load_records(paths: Iterable[Path]) -> Dict[Tuple[str, str, str, str, int], dict]:
    records: Dict[Tuple[str, str, str, str, int], dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (
                str(item["experiment_id"]), str(item["dataset"]),
                str(item["condition"]), str(item["question_id"]), int(item["seed"]),
            )
            if key in records:
                raise ValueError(f"duplicate key across inputs: {key}")
            records[key] = item
    return records


def expected_jobs(matrix: dict) -> dict:
    defaults = matrix["defaults"]
    split_data = json.loads((ROOT / "experiments/mechanism/splits.json").read_text())
    expected = {}
    for raw in matrix["jobs"]:
        job = {**defaults, **raw}
        ids = split_data["datasets"][job["dataset"]]["splits"][job["split"]][:job["n_samples"]]
        condition = condition_name(
            job["behavior"], job["mechanism"], job["reward_scale"],
            job.get("cost_scale", 1.0), job.get("signal_mask"),
            job.get("history_policy", "cumulative"), job.get("variant_tag"),
        )
        for qid in ids:
            key = (job["experiment_id"], job["dataset"], condition,
                   str(qid), job["seed"])
            if key in expected:
                raise ValueError(f"duplicate expected key in matrix: {key}")
            expected[key] = job
    return expected


def expected_keys(matrix: dict) -> set:
    """Compatibility helper used by audits and tests."""
    return set(expected_jobs(matrix))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--full-details", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    records = load_records(args.records)
    jobs_by_key = expected_jobs(matrix)
    expected = set(jobs_by_key)
    present = set(records)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    incomplete = sorted(key for key in expected & present if not records[key].get("completed"))

    route_errors = []
    schema_errors = []
    for key in expected & present:
        record = records[key]
        job = jobs_by_key[key]
        if not record.get("completed"):
            continue
        validation = record_from_dict(record).validate(require_complete=True)
        if validation:
            schema_errors.append([key, validation])
        for item in record.get("rounds", []):
            traces = item.get("action_trace") or []
            action = item.get("chosen_action")
            execution = [x for x in traces if x.get("stage") != "choose" and
                         not str(x.get("stage", "")).startswith("format_repair")]
            backends = {x.get("backend") for x in execution}
            assigned = job["backends"][int(item.get("agent_id", 0)) % len(job["backends"])]
            tiered = bool(job.get("resource_tiered"))
            expected_backend = assigned
            if tiered and action == "low_effort":
                expected_backend = "zhipu_flash"
            elif tiered and action == "verify":
                expected_backend = "deepseek_reasoner"
            if backends != {expected_backend}:
                route_errors.append([key, item.get("round_id"), item.get("agent_id"), action, sorted(backends)])
            repair_backends = {
                x.get("backend") for x in traces
                if str(x.get("stage", "")).startswith("format_repair")
            }
            if repair_backends and repair_backends != {"deepseek"}:
                route_errors.append([key, item.get("round_id"), item.get("agent_id"),
                                     "format_repair", sorted(repair_backends)])

    hashes = {}
    for path in args.records:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    def detail(items: list) -> list:
        return items if args.full_details else items[:20]

    report = {
        "matrix_version": matrix.get("matrix_version"),
        "expected_units": len(expected),
        "present_units": len(expected & present),
        "completed_units": len(expected - set(incomplete) - set(missing)),
        "missing_count": len(missing),
        "incomplete_count": len(incomplete),
        "unexpected_count": len(unexpected),
        "route_error_count": len(route_errors),
        "schema_error_count": len(schema_errors),
        "missing_units": detail(missing),
        "incomplete_units": detail(incomplete),
        "unexpected_units": detail(unexpected),
        "route_errors": detail(route_errors),
        "schema_errors": detail(schema_errors),
        "action_counts": dict(Counter(
            item.get("chosen_action") for key in expected & present
            for item in records[key].get("rounds", []) if records[key].get("completed")
        )),
        "sha256": hashes,
    }
    report["pass"] = not (
        missing or incomplete or unexpected or route_errors or schema_errors
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
