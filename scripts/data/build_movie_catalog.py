#!/usr/bin/env python3
"""Build a traceable MovieLens catalog without loading large CSVs into RAM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
from collections import defaultdict
from pathlib import Path


TITLE_YEAR = re.compile(r"^(.*) \((\d{4})\)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_title(raw: str) -> tuple[str, int | None]:
    match = TITLE_YEAR.match(raw.strip())
    return (match.group(1), int(match.group(2))) if match else (raw.strip(), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/movielens/ml-25m"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/movie_catalog.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("data/processed/movie_catalog.stats.json"))
    parser.add_argument("--min-ratings", type=int, default=20)
    parser.add_argument("--top-tags", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = args.source
    required = ["movies.csv", "links.csv", "ratings.csv", "genome-tags.csv", "genome-scores.csv"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"missing MovieLens files: {', '.join(missing)}")

    movies: dict[int, dict] = {}
    with (source / "movies.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            movie_id = int(row["movieId"])
            title, year = parse_title(row["title"])
            movies[movie_id] = {
                "movie_id": str(movie_id), "title": title, "year": year,
                "genres": [] if row["genres"] == "(no genres listed)" else row["genres"].split("|"),
            }

    with (source / "links.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = movies.get(int(row["movieId"]))
            if item is not None:
                item["imdb_id"] = f"tt{row['imdbId']}" if row.get("imdbId") else None
                item["tmdb_id"] = row.get("tmdbId") or None

    rating_count: dict[int, int] = defaultdict(int)
    rating_sum: dict[int, float] = defaultdict(float)
    with (source / "ratings.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            movie_id = int(row["movieId"])
            rating_count[movie_id] += 1
            rating_sum[movie_id] += float(row["rating"])

    tags = {}
    with (source / "genome-tags.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tags[int(row["tagId"])] = row["tag"]
    top: dict[int, list[tuple[float, int]]] = defaultdict(list)
    with (source / "genome-scores.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            movie_id, tag_id, score = int(row["movieId"]), int(row["tagId"]), float(row["relevance"])
            heap = top[movie_id]
            value = (score, tag_id)
            if len(heap) < args.top_tags:
                heapq.heappush(heap, value)
            elif value > heap[0]:
                heapq.heapreplace(heap, value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for movie_id in sorted(movies):
            count = rating_count.get(movie_id, 0)
            if count < args.min_ratings:
                continue
            item = movies[movie_id]
            item["rating_count"] = count
            item["rating_mean"] = round(rating_sum[movie_id] / count, 4)
            item["semantic_tags"] = [
                {"tag": tags[tag_id], "relevance": round(score, 4)}
                for score, tag_id in sorted(top.get(movie_id, []), reverse=True)
            ]
            item["evidence_refs"] = {
                "metadata": f"movielens:movies.csv:movieId={movie_id}",
                "ratings": f"movielens:ratings.csv:movieId={movie_id}",
                "tags": f"movielens:genome-scores.csv:movieId={movie_id}",
            }
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            kept += 1

    stats = {
        "source": str(source), "output": str(args.output), "min_ratings": args.min_ratings,
        "top_tags": args.top_tags, "source_movies": len(movies), "catalog_movies": kept,
        "source_sha256": {name: sha256(source / name) for name in required},
        "output_sha256": sha256(args.output),
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
