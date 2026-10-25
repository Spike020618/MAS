#!/usr/bin/env python3
"""Build train-only movie-to-movie co-occurrence signals for ReDial retrieval."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_redial_benchmark import (  # noqa: E402
    find_split, label_is_positive, load_catalog, map_movie, read_dialogues,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redial-root", type=Path, default=Path("experiments/data/raw/redial"))
    parser.add_argument("--catalog", type=Path,
                        default=Path("experiments/data/processed/movie_catalog.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/data/processed/redial_train_cooccurrence.json"))
    parser.add_argument("--neighbors", type=int, default=200)
    parser.add_argument("--exclude-cases", type=Path,
                        help="JSONL benchmark cases whose conversation ids must be excluded")
    args = parser.parse_args()

    exact, title_only, _ = load_catalog(args.catalog)
    source = find_split(args.redial_root, "train")
    excluded_conversations = set()
    if args.exclude_cases:
        excluded_conversations = {
            str(json.loads(line)["conversation_id"])
            for line in args.exclude_cases.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    movie_counts: Counter[str] = Counter()
    usable_dialogues = 0
    for dialogue in read_dialogues(source):
        if str(dialogue.get("conversationId")) in excluded_conversations:
            continue
        mentions = {str(k): str(v) for k, v in (dialogue.get("movieMentions") or {}).items()}
        labels = dialogue.get("initiatorQuestions") or {}
        positive = set()
        for redial_id, label in labels.items():
            if not isinstance(label, dict) or not label_is_positive(label.get("liked")):
                continue
            movie_id, _ = map_movie(mentions.get(str(redial_id), ""), exact, title_only)
            if movie_id:
                positive.add(movie_id)
        if len(positive) < 2:
            continue
        usable_dialogues += 1
        for movie_id in positive:
            movie_counts[movie_id] += 1
            for neighbor in positive - {movie_id}:
                pair_counts[movie_id][neighbor] += 1

    total = max(1, usable_dialogues)
    associations = {}
    for movie_id, neighbors in pair_counts.items():
        ranked = []
        for neighbor, count in neighbors.items():
            # Smoothed positive-PMI discounts globally frequent titles while
            # keeping count as an explicit reliability signal.
            pmi = math.log((count + 1) * total /
                           ((movie_counts[movie_id] + 1) * (movie_counts[neighbor] + 1)))
            score = max(0.0, pmi) * math.log1p(count)
            ranked.append({"movie_id": neighbor, "count": count, "score": round(score, 8)})
        associations[movie_id] = sorted(
            ranked, key=lambda item: (-item["score"], -item["count"], item["movie_id"])
        )[:args.neighbors]

    payload = {
        "version": "redial-train-cooccurrence.v1",
        "source": str(source),
        "train_dialogues": sum(1 for _ in read_dialogues(source)),
        "usable_dialogues": usable_dialogues,
        "excluded_conversations": len(excluded_conversations),
        "movies": len(movie_counts),
        "associations": associations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "associations"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
