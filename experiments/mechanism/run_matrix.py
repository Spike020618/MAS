"""Run a frozen AEIC-MD collection matrix sequentially and resumably."""
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
    parser.add_argument("--only-condition", action="append", default=[])
    parser.add_argument("--out", type=Path,
                        help="Override matrix output path (useful for independent dataset shards)")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    defaults = matrix["defaults"]
    jobs = matrix["jobs"]
    selected = []
    for job in jobs:
        merged = {**defaults, **job}
        if args.only_dataset and merged["dataset"] not in args.only_dataset:
            continue
        label = merged.get("label", merged["mechanism"])
        if args.only_condition and label not in args.only_condition:
            continue
        selected.append(merged)
    if not selected:
        raise SystemExit("matrix selection contains no jobs")

    for index, job in enumerate(selected, 1):
        out = args.out if args.out else ROOT / job["out"]
        if not out.is_absolute():
            out = ROOT / out
        cmd = [
            sys.executable, str(HERE / "collect.py"),
            "--dataset", job["dataset"],
            "--split", job["split"],
            "--n-samples", str(job["n_samples"]),
            "--behavior", job["behavior"],
            "--mechanism", job["mechanism"],
            "--reward-scale", str(job["reward_scale"]),
            "--cost-scale", str(job.get("cost_scale", 1.0)),
            "--n-agents", str(job["n_agents"]),
            "--n-rounds", str(job["n_rounds"]),
            "--seed", str(job["seed"]),
            "--max-attempts", str(job["max_attempts"]),
            "--experiment-id", job["experiment_id"],
            "--out", str(out),
            "--backends", *job["backends"],
        ]
        if job.get("resource_tiered"):
            cmd.append("--resource-tiered")
        if job.get("similarity"):
            cmd.extend(["--similarity", job["similarity"]])
        if job.get("signal_mask"):
            cmd.extend(["--signal-mask", job["signal_mask"]])
        if job.get("history_policy"):
            cmd.extend(["--history-policy", job["history_policy"]])
        if job.get("variant_tag"):
            cmd.extend(["--variant-tag", job["variant_tag"]])
        if "flip_threshold" in job:
            cmd.extend(["--flip-threshold", str(job["flip_threshold"])])
        print(f"[{index}/{len(selected)}] {job['dataset']} {job.get('label', job['mechanism'])}", flush=True)
        print(" ".join(cmd), flush=True)
        if args.print_only:
            continue
        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode:
            raise SystemExit(
                f"matrix stopped at job {index}; resume with the same command after correction"
            )


if __name__ == "__main__":
    main()
