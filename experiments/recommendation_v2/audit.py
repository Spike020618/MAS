"""Offline P1/P2 audit. No network, API, LLM or preference inference.

python3 -m experiments.recommendation_v2.audit
Each completed component is reusable only with identical source/code hashes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .prepare import (ROOT, DEFAULT_OUTPUT, digest, movie_mapping, prefix_input,
                      save, select_case, snapshot_legacy, validate_pair)
from experiments.scripts.data.build_redial_benchmark import load_catalog, read_dialogues

STOP = set("a an the i you he she it we they is are was were be been have has had do does did "
           "movie movies film films seeker recommender like liked love seen watch watching "
           "recommend recommendation hi hello how what which and or of to in on for with "
           "me my your that this thanks thank recently good great really would can so not".split())


def terms(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in STOP]


class LexicalIndex:
    """BM25 over titles and genres only. No future popularity or label signals."""

    def __init__(self, movies: dict):
        self.postings = defaultdict(list)
        self.lengths = {}
        for mid, movie in movies.items():
            counts = Counter(terms(movie["title"] + " " + " ".join(movie.get("genres", []))))
            self.lengths[mid] = sum(counts.values())
            for term, frequency in counts.items():
                self.postings[term].append((mid, frequency))
        self.average = sum(self.lengths.values()) / max(1, len(self.lengths))

    def search(self, request: dict, limit: int = 100) -> list:
        # Deliberately takes only a serving input, never its label row.
        if request.get("hard_constraints"):
            raise ValueError("this diagnostic does not implement constraint parsing")
        scores = defaultdict(float)
        n = len(self.lengths)
        for term in set(terms(request["query"])):
            posting = self.postings.get(term, [])
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for mid, tf in posting:
                denominator = tf + 1.2 * (0.25 + 0.75 * self.lengths[mid] / max(1, self.average))
                scores[mid] += idf * tf * 2.2 / denominator
        return sorted(scores, key=lambda mid: (-scores[mid], mid))[:limit]


def cached(path: Path, dependencies: dict):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["dependencies"] != dependencies:
        raise ValueError(f"stale checkpoint; choose a new output version: {path}")
    if data.get("pass") is False:
        raise ValueError(f"failed checkpoint cannot be resumed as success: {path}")
    for name, expected in data.get("artifacts", {}).items():
        if digest(path.parent / name) != expected:
            raise ValueError(f"checkpoint artifact changed: {name}")
    return data


def redial_audit(data: Path, output: Path):
    deps = {"prepared_manifest": digest(data / "manifest.json"), "audit_code": digest(Path(__file__))}
    path = output / "redial_integrity.json"
    if cached(path, deps):
        return
    catalog = ROOT / "experiments/data/processed/movie_catalog.jsonl"
    exact, titles, movies = load_catalog(catalog)
    source = {f"redial:{d['conversationId']}": d for split in ("train", "test")
              for d in read_dialogues(ROOT / f"experiments/data/raw/redial/{split}.jsonl")}
    errors, counts, ids = [], {}, {}
    for split in ("train", "dev", "legacy_test_diagnostic"):
        inputs = read_dialogues(data / f"{split}.inputs.jsonl")
        labels = read_dialogues(data / f"{split}.labels.jsonl")
        if len(inputs) != len(labels):
            raise ValueError("input/label count mismatch")
        ids[split] = {inp["case_id"] for inp in inputs}
        if len(ids[split]) != len(inputs):
            raise ValueError("duplicate input")
        for inp, label in zip(inputs, labels):
            d = source[inp["case_id"]]
            mapping = movie_mapping(d, exact, titles)
            try:
                validate_pair(inp, label, set(movies))
                if prefix_input(d, inp["cutoff_exclusive"], mapping) != inp:
                    raise ValueError("prefix reconstruction differs")
                # Verify sampler, not just the reconstruction and plausible label schema.
                if select_case(d, mapping) != (inp, label):
                    raise ValueError("case sampling / labels differ from source")
            except ValueError as exc:
                errors.append({"case_id": inp["case_id"], "error": str(exc)})
        counts[split] = len(inputs)
    overlap = {f"{a}__{b}": len(ids[a] & ids[b]) for i, a in enumerate(ids)
               for b in list(ids)[i + 1:]}
    passed = not errors and not any(overlap.values())
    save(path, {"dependencies": deps, "counts": counts, "conversation_overlap": overlap,
                "errors": errors, "pass": passed,
                "scope": "full deterministic reconstruction; no sentiment extraction certified"})
    if not passed:
        raise ValueError("ReDial integrity failed")


def retrieval_audit(data: Path, output: Path):
    deps = {"prepared_manifest": digest(data / "manifest.json"), "audit_code": digest(Path(__file__))}
    path = output / "retrieval_coverage.json"
    if cached(path, deps):
        return
    _, _, movies = load_catalog(ROOT / "experiments/data/processed/movie_catalog.jsonl")
    index = LexicalIndex(movies)
    report = {}
    for split in ("dev", "legacy_test_diagnostic"):
        inputs = read_dialogues(data / f"{split}.inputs.jsonl")
        labels = {row["case_id"]: row for row in read_dialogues(data / f"{split}.labels.jsonl")}
        hits, recalls, counts = Counter(), Counter(), []
        start = time.perf_counter()
        rows = []
        for inp in inputs:
            ranked = index.search(inp, limit=100)
            # Gold is read only after retrieval, never for pool construction.
            relevant = set(labels[inp["case_id"]]["relevant_movie_ids"])
            counts.append(len(ranked))
            for k in (5, 10, 20, 50, 100):
                matched = relevant & set(ranked[:k])
                hits[k] += bool(matched)
                recalls[k] += len(matched) / len(relevant)
            rows.append({"case_id": inp["case_id"], "ranked_movie_ids": ranked})
        elapsed = time.perf_counter() - start
        save(output / f"{split}.lexical_candidates.jsonl", rows, jsonl=True)
        n = len(inputs)
        report[split] = {"cases": n, "hits": dict(hits),
                         "hit_rate": {k: hits[k] / n for k in (5, 10, 20, 50, 100)},
                         "recall": {k: recalls[k] / n for k in (5, 10, 20, 50, 100)},
                         "empty_candidate_cases": counts.count(0),
                         "under_5_candidates": sum(c < 5 for c in counts),
                         "under_100_candidates": sum(c < 100 for c in counts),
                         "retrieval_and_scoring_elapsed_seconds": elapsed}
    artifacts = {f"{split}.lexical_candidates.jsonl": digest(output / f"{split}.lexical_candidates.jsonl")
                 for split in report}
    save(path, {"dependencies": deps, "splits": report, "artifacts": artifacts,
                "method": "title+genre BM25; conservative prefix; no cooccurrence or popularity",
                "temporal_scope": "fixed modern catalog snapshot, not historical catalog reconstruction",
                "constraint_satisfaction": "not evaluated; no extracted constraints",
                "fresh_test": False, "model_calls": 0,
                "interpretation": "coverage diagnostic, not quality or new mechanism result"})


def movielens_audit(output: Path):
    root = ROOT / "experiments/data/raw/movielens/ml-25m"
    names = ("ratings.csv", "movies.csv", "tags.csv", "genome-scores.csv", "README.txt")
    deps = {name: digest(root / name) for name in names}
    deps["audit_code"] = digest(Path(__file__))
    path = output / "movielens_inventory.json"
    if cached(path, deps):
        return
    with (root / "movies.csv").open(newline="", encoding="utf-8") as handle:
        movies = {row["movieId"] for row in csv.DictReader(handle)}
    users = Counter()
    first, last = {}, {}
    n, invalid, duplicates, order_violations = 0, 0, 0, 0
    previous = None
    start = time.perf_counter()
    with (root / "ratings.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid, mid, rating, stamp = row["userId"], row["movieId"], float(row["rating"]), int(row["timestamp"])
            n += 1
            users[uid] += 1
            first[uid] = min(first.get(uid, stamp), stamp)
            last[uid] = max(last.get(uid, stamp), stamp)
            invalid += (mid not in movies or not 0.5 <= rating <= 5.0 or rating * 2 != int(rating * 2))
            key = (int(uid), int(mid))
            duplicates += key == previous
            order_violations += previous is not None and key < previous
            previous = key
    # Adjacent duplicate count is exhaustive only if sorted key order is verified.
    elapsed = time.perf_counter() - start
    save(path, {"dependencies": deps, "ratings": n, "users": len(users), "movies": len(movies),
                "invalid_rows": invalid, "adjacent_duplicate_user_movie": duplicates,
                "key_order_violations": order_violations,
                "min_ratings_per_user": min(users.values()),
                "users_with_at_least": {threshold: sum(c >= threshold for c in users.values())
                                        for threshold in (20, 30, 50, 100)},
                "users_with_only_one_timestamp": sum(first[u] == last[u] for u in users),
                "earliest_utc": datetime.fromtimestamp(min(first.values()), timezone.utc).isoformat(),
                "latest_utc": datetime.fromtimestamp(max(last.values()), timezone.utc).isoformat(),
                "scan_elapsed_seconds": elapsed,
                "raw_data_available": True, "evaluation_ready": False,
                "remaining": ["user/time split and timestamp tie handling",
                              "training-only feature/association construction",
                              "held-out observed-rating evaluation with missingness disclosure",
                              "no joining MovieLens identities to ReDial workers"],
                "temporal_warning": "full-snapshot tag genome and global averages are not automatically historical features"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/recommendation_v2/audit")
    args = parser.parse_args()
    prepared = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if digest(Path(__file__).with_name("prepare.py")) != prepared["code_sha256"]:
        raise ValueError("prepared builder code changed; rebuild a new dataset version")
    for name, meta in prepared["outputs"].items():
        if digest(args.data / name) != meta["sha256"]:
            raise ValueError(f"prepared data altered: {name}")
    catalog = ROOT / "experiments/data/processed/movie_catalog.jsonl"
    if digest(catalog) != prepared["catalog_sha256"]:
        raise ValueError("catalog changed")
    for split, expected in prepared["sources"].items():
        if digest(ROOT / f"experiments/data/raw/redial/{split}.jsonl") != expected:
            raise ValueError("raw source changed")
    legacy = json.loads((args.data / "legacy_snapshot.json").read_text(encoding="utf-8"))
    if snapshot_legacy() != legacy:
        raise ValueError("legacy evidence changed")
    for name, fn in (("integrity", lambda: redial_audit(args.data, args.output)),
                     ("retrieval", lambda: retrieval_audit(args.data, args.output)),
                     ("movielens", lambda: movielens_audit(args.output))):
        print(f"START {name}", flush=True)
        fn()
        print(f"DONE {name}", flush=True)
    if snapshot_legacy() != legacy:
        raise ValueError("legacy evidence changed during audit")
    print("legacy SHA256 verified; new model calls: 0", flush=True)


if __name__ == "__main__":
    main()
