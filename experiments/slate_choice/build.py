"""Build independent user/time-isolated MovieLens inputs and offline labels.

python3 -m experiments.slate_choice.build
Training priors use training-users only and timestamps strictly before cutoff.
Personal dev/test histories also strictly precede cutoff. No future ratings are
used for candidate generation, feature construction or fitting.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import re
import time
from collections import Counter
from pathlib import Path

from .common import ROOT, OUTPUT, PROTOCOL, sha, write_once, user_bucket, sample_key


def chronological_split(records, cutoff):
    past = sorted((r for r in records if r["timestamp"] < cutoff), key=lambda r: (r["timestamp"], r["movie_id"]))
    future = sorted((r for r in records if r["timestamp"] >= cutoff), key=lambda r: (r["timestamp"], r["movie_id"]))
    return past, future


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "data")
    args = parser.parse_args()
    cfg = json.loads(PROTOCOL.read_text())
    root = ROOT / "experiments/data/raw/movielens/ml-25m"
    dependency = {"protocol": sha(PROTOCOL), "build_code": sha(Path(__file__)),
                  "common_code": sha(Path(__file__).with_name("common.py")),
                  "ratings": sha(root / "ratings.csv"), "movies": sha(root / "movies.csv")}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["dependencies"] != dependency:
            raise ValueError("changed build dependency: use new output version")
        for name, expected in manifest["artifacts"].items():
            if sha(args.output / name) != expected:
                raise ValueError(f"altered artifact: {name}")
        print("existing build verified", flush=True)
        return
    movies = {}
    with (root / "movies.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            match = re.search(r"\((\d{4})\)\s*$", row["title"])
            if match and int(match.group(1)) <= cfg["latest_catalog_year"]:
                movies[row["movieId"]] = {"movie_id": row["movieId"], "title": row["title"],
                                          "year": int(match.group(1)),
                                          "genres": [g for g in row["genres"].split("|") if g != "(no genres listed)"]}
    prior_path = args.output / "train_priors.json"
    print("START training-user pre-cutoff priors", flush=True)
    if prior_path.exists():
        priors = json.loads(prior_path.read_text())
        if priors["dependencies"] != dependency:
            raise ValueError("stale prior checkpoint")
    else:
        sums, counts = Counter(), Counter()
        scanned, used = 0, 0
        with (root / "ratings.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scanned += 1
                if (user_bucket(row["userId"]) >= 2 and int(row["timestamp"]) < cfg["global_cutoff_timestamp"]
                        and row["movieId"] in movies):
                    counts[row["movieId"]] += 1
                    sums[row["movieId"]] += float(row["rating"])
                    used += 1
        average = sum(sums.values()) / sum(counts.values())
        priors = {"dependencies": dependency, "source_rows": scanned, "used_rows": used,
                  "global_mean": average, "items": {
                      mid: dict(movies[mid], train_count=counts[mid],
                                prior=(sums[mid] + cfg["prior_smoothing"] * average) /
                                      (counts[mid] + cfg["prior_smoothing"]))
                      for mid in sorted(counts) if counts[mid] >= cfg["minimum_train_item_ratings"]}}
        write_once(prior_path, priors)
    print(f"DONE priors; eligible items={len(priors['items'])}", flush=True)
    heaps = {"dev": [], "reserved_test": []}
    limits = {"dev": cfg["max_dev_users"], "reserved_test": cfg["max_reserved_test_users"]}
    counts = Counter()
    cutoff = cfg["global_cutoff_timestamp"]
    start = time.perf_counter()
    print("START user/time split", flush=True)
    with (root / "ratings.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Verified source ordering is userId, then movieId. Check rather than assume.
        previous_uid = -1
        for uid, group in itertools.groupby(reader, key=lambda r: r["userId"]):
            if int(uid) <= previous_uid:
                raise ValueError("source must be strictly grouped by user")
            previous_uid = int(uid)
            bucket = user_bucket(uid)
            if bucket >= 2:
                for _ in group:
                    pass
                continue
            split = "dev" if bucket == 0 else "reserved_test"
            records = [{"movie_id": r["movieId"], "rating": float(r["rating"]), "timestamp": int(r["timestamp"])}
                       for r in group]
            counts[f"{split}_users_total"] += 1
            past, future = chronological_split(records, cutoff)
            eligible_past = [r for r in past if r["movie_id"] in priors["items"]]
            if len(eligible_past) < cfg["minimum_personal_history"]:
                counts[f"{split}_insufficient_history"] += 1
                continue
            if len(future) < cfg["minimum_future_observations"]:
                counts[f"{split}_insufficient_future_observations"] += 1
                continue
            counts[f"{split}_eligible"] += 1
            inp = {"user_id": uid, "history": eligible_past[-max(cfg["history_budgets"]):],
                   "seen_movie_ids": sorted({r["movie_id"] for r in past}),
                   "cutoff": cutoff, "hard_constraints": {"max_year": cfg["latest_catalog_year"]}}
            label = {"user_id": uid, "future_ratings": future}
            key = -int(sample_key(uid), 16)
            item = (key, uid, inp, label)
            heap = heaps[split]
            if len(heap) < limits[split]:
                heapq.heappush(heap, item)
            elif key > heap[0][0]:
                heapq.heapreplace(heap, item)
    artifacts = {prior_path.name: sha(prior_path)}
    sample = {}
    for split, heap in heaps.items():
        selected = sorted(heap, key=lambda x: x[1])
        for kind, index in (("inputs", 2), ("labels", 3)):
            p = args.output / f"{split}.{kind}.jsonl"
            write_once(p, [item[index] for item in selected], jsonl=True)
            artifacts[p.name] = sha(p)
        sample[split] = len(selected)
    users_a = {x[1] for x in heaps["dev"]}
    users_b = {x[1] for x in heaps["reserved_test"]}
    if users_a & users_b:
        raise AssertionError("user split overlap")
    manifest = {"version": cfg["version"], "dependencies": dependency, "artifacts": artifacts,
                "counts": dict(counts), "selected": sample, "user_overlap": 0,
                "cutoff": cutoff, "dataset_scope": "retrospective active users, fixed cutoff",
                "reserved_test_evaluated": False, "model_calls": 0,
                "split_scan_seconds": time.perf_counter() - start}
    write_once(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
