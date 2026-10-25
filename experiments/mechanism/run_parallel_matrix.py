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
sys.path.insert(0, str(ROOT))

from experiments.mechanism.run_matrix import merged_jobs


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use collector mocks; makes no external API calls")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = merged_jobs(matrix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for index, job in enumerate(jobs, 1):
        merged = job
        label = job["_matrix_job_label"]
        shard = args.out_dir / (
            f"{index:02d}_{safe_name(merged['dataset'])}_{safe_name(label)}.jsonl"
        )
        command = [
            sys.executable, str(HERE / "run_matrix.py"), str(args.matrix),
            "--job-index", str(merged["_matrix_job_index"]),
            "--out", str(shard),
        ]
        if args.dry_run:
            command.append("--dry-run")
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
