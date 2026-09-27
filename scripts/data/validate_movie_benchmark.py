#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/movie_catalog.jsonl"))
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/movie_consensus"))
    args = parser.parse_args()
    catalog = {x["movie_id"] for x in rows(args.catalog)}
    dev, test = rows(args.benchmark / "dev.jsonl"), rows(args.benchmark / "test.jsonl")
    errors = []
    dev_ids, test_ids = {x["case_id"] for x in dev}, {x["case_id"] for x in test}
    if dev_ids & test_ids:
        errors.append(f"dev/test overlap: {len(dev_ids & test_ids)}")
    for split, data in (("dev", dev), ("test", test)):
        if len({x["case_id"] for x in data}) != len(data):
            errors.append(f"{split}: duplicate case ids")
        for item in data:
            relevant = item.get("relevant_movie_ids") or []
            if not relevant:
                errors.append(f"{item.get('case_id')}: empty relevance")
            unknown = set(relevant + (item.get("excluded_seen_movie_ids") or [])) - catalog
            if unknown:
                errors.append(f"{item.get('case_id')}: unknown ids {sorted(unknown)}")
    report = {"catalog_movies": len(catalog), "dev_cases": len(dev), "test_cases": len(test),
              "dev_test_overlap": len(dev_ids & test_ids), "errors": errors, "pass": not errors}
    path = args.benchmark / "validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
