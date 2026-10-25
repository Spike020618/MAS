"""Run the frozen report-interface matrix sequentially and resumably."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--only-dataset", action="append", default=[])
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    required = {
        "experiment_id", "split", "n_samples_per_dataset", "protocols", "backend",
        "max_tokens", "temperature", "seed", "datasets", "out",
    }
    missing = sorted(required - set(matrix))
    if missing:
        raise ValueError(f"matrix missing fields: {missing}")
    datasets = [item for item in matrix["datasets"]
                if not args.only_dataset or item in args.only_dataset]
    if not datasets:
        raise SystemExit("matrix selection contains no datasets")
    out = args.out or ROOT / matrix["out"]
    if not out.is_absolute():
        out = ROOT / out
    for index, dataset in enumerate(datasets, 1):
        cmd = [
            sys.executable, str(HERE / "collect_report_interface.py"),
            "--dataset", dataset, "--split", matrix["split"],
            "--n-samples", str(matrix["n_samples_per_dataset"]),
            "--protocols", *matrix["protocols"],
            "--backend", matrix["backend"], "--max-tokens", str(matrix["max_tokens"]),
            "--temperature", str(matrix["temperature"]), "--seed", str(matrix["seed"]),
            "--experiment-id", matrix["experiment_id"], "--out", str(out),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[{index}/{len(datasets)}] {dataset}", flush=True)
        print(" ".join(cmd), flush=True)
        if not args.print_only:
            completed = subprocess.run(cmd, cwd=ROOT)
            if completed.returncode:
                raise SystemExit(f"matrix stopped at {dataset}")


if __name__ == "__main__":
    main()
