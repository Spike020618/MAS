"""Run a frozen AEIC-MD collection matrix sequentially and resumably."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


REQUIRED_JOB_FIELDS = {
    "dataset", "split", "n_samples", "behavior", "mechanism",
    "reward_scale", "n_agents", "n_rounds", "seed", "max_attempts",
    "experiment_id", "out", "backends",
}


def merged_jobs(matrix: dict) -> list[dict]:
    if not isinstance(matrix.get("defaults"), dict) or not isinstance(matrix.get("jobs"), list):
        raise ValueError("matrix requires object defaults and list jobs")
    jobs = []
    identities = set()
    for index, raw in enumerate(matrix["jobs"], 1):
        if not isinstance(raw, dict):
            raise ValueError(f"jobs[{index}] must be an object")
        job = {**matrix["defaults"], **raw}
        missing = sorted(REQUIRED_JOB_FIELDS - set(job))
        if missing:
            raise ValueError(f"jobs[{index}] missing fields: {missing}")
        if int(job["n_samples"]) < 1 or int(job["n_agents"]) < 1 or int(job["n_rounds"]) < 1:
            raise ValueError(f"jobs[{index}] sample/agent/round counts must be positive")
        if not isinstance(job["backends"], list) or not job["backends"]:
            raise ValueError(f"jobs[{index}] backends must be a nonempty list")
        identity = (
            job["experiment_id"], job["dataset"],
            raw.get("label", job["mechanism"]), job.get("seed"),
        )
        if identity in identities:
            raise ValueError(f"duplicate matrix job identity at jobs[{index}]: {identity}")
        identities.add(identity)
        job["_matrix_job_index"] = index
        job["_matrix_job_label"] = raw.get("label", job["mechanism"])
        jobs.append(job)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--only-dataset", action="append", default=[])
    parser.add_argument("--only-condition", action="append", default=[])
    parser.add_argument("--job-index", action="append", type=int, default=[],
                        help="Run an exact 1-based matrix job; safe for parallel shards")
    parser.add_argument("--out", type=Path,
                        help="Override matrix output path (useful for independent dataset shards)")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use collector mocks; makes no external API calls")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = merged_jobs(matrix)
    selected = []
    for job in jobs:
        if args.job_index and job["_matrix_job_index"] not in args.job_index:
            continue
        if args.only_dataset and job["dataset"] not in args.only_dataset:
            continue
        label = job["_matrix_job_label"]
        if args.only_condition and label not in args.only_condition:
            continue
        selected.append(job)
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
        if job.get("deepseek_only_tiers"):
            cmd.append("--deepseek-only-tiers")
        if job.get("fixed_resources"):
            cmd.append("--fixed-resources")
        if job.get("resource_schedule"):
            schedule = ROOT / job["resource_schedule"]
            cmd.extend(["--resource-schedule", str(schedule)])
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
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[{index}/{len(selected)}] job={job['_matrix_job_index']} "
              f"{job['dataset']} {job['_matrix_job_label']}", flush=True)
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
