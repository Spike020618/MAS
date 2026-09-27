"""Run matrix jobs concurrently into independent, resumable JSONL shards."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = matrix["jobs"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for index, job in enumerate(jobs, 1):
        merged = {**matrix["defaults"], **job}
        label = job.get("label", merged["mechanism"])
        shard = args.out_dir / (
            f"{index:02d}_{safe_name(merged['dataset'])}_{safe_name(label)}.jsonl"
        )
        command = [
            sys.executable, str(HERE / "run_matrix.py"), str(args.matrix),
            "--only-dataset", merged["dataset"],
            "--only-condition", label,
            "--out", str(shard),
        ]
        commands.append((index, merged["dataset"], label, command))

    if args.print_only:
        for index, dataset, label, command in commands:
            print(f"[{index}/{len(commands)}] {dataset}/{label}: {' '.join(command)}")
        return

    failures = []
    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    try:
        futures = {
            executor.submit(subprocess.run, command, cwd=ROOT): (index, dataset, label)
            for index, dataset, label, command in commands
        }
        for future in as_completed(futures):
            index, dataset, label = futures[future]
            completed = future.result()
            print(f"[{index}/{len(commands)}] {dataset}/{label} exit={completed.returncode}",
                  flush=True)
            if completed.returncode:
                failures.append((index, dataset, label, completed.returncode))
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    if failures:
        raise SystemExit("failed shards: " + json.dumps(failures))


if __name__ == "__main__":
    main()
