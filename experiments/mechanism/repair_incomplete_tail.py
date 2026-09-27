"""Remove completed longitudinal units that occur after the first incomplete unit.

The terminal error itself is retained so retry history remains auditable. A
byte-for-byte backup is written before the atomic rewrite.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    split = json.loads((ROOT / "experiments/mechanism/splits.json").read_text())
    order = {qid: i for i, qid in enumerate(split["datasets"][args.dataset]["splits"]["dev"])}
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    target = [row for row in rows if row.get("dataset") == args.dataset
              and row.get("condition") == args.condition]
    incomplete = [row for row in target if not row.get("completed")]
    if not incomplete:
        print("no incomplete unit; nothing to repair")
        return
    cutoff = min(order[str(row["question_id"])] for row in incomplete)
    removed = [row for row in target if row.get("completed")
               and order[str(row["question_id"])] > cutoff]
    print(json.dumps({
        "cutoff_index": cutoff,
        "incomplete_question_ids": [row["question_id"] for row in incomplete],
        "remove_completed_question_ids": [row["question_id"] for row in removed],
    }, ensure_ascii=False, indent=2))
    if not args.apply or not removed:
        return
    remove_keys = {(row["experiment_id"], row["dataset"], row["condition"],
                    str(row["question_id"]), int(row["seed"])) for row in removed}
    kept = [row for row in rows if (row["experiment_id"], row["dataset"], row["condition"],
            str(row["question_id"]), int(row["seed"])) not in remove_keys]
    backup = args.records.with_suffix(args.records.suffix + ".pre_tail_repair")
    if backup.exists():
        raise SystemExit(f"refusing to overwrite existing backup: {backup}")
    shutil.copy2(args.records, backup)
    tmp = args.records.with_suffix(args.records.suffix + ".repair.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, args.records)
    print(f"repaired {args.records}; backup={backup}")


if __name__ == "__main__":
    main()
