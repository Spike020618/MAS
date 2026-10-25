"""Choose the hybrid retrieval weight on a leakage-isolated ReDial dev split."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.game_consensus_recsys.domain import SearchRequest
from system.game_consensus_recsys.movie_data import HybridMovieCatalog, JsonlMovieCatalog


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def request(case: dict) -> SearchRequest:
    return SearchRequest(
        case["query"], top_k=5,
        excluded_movie_ids=list(map(str, case.get("excluded_seen_movie_ids", []))),
        liked_movie_ids=list(map(str, case.get("liked_context_movie_ids", []))),
    )


def score(cases: list[dict], ranker) -> dict:
    hits = recalls = ndcgs = 0.0
    for case in cases:
        relevant = set(map(str, case["relevant_movie_ids"]))
        ranked = ranker(case)[:5]
        found = sum(movie_id in relevant for movie_id in ranked)
        hits += float(found > 0)
        recalls += found / len(relevant)
        dcg = sum(1 / math.log2(index + 2) for index, movie_id in enumerate(ranked)
                  if movie_id in relevant)
        ideal = sum(1 / math.log2(index + 2) for index in range(min(5, len(relevant))))
        ndcgs += dcg / ideal
    size = len(cases)
    return {"hit_at_5": hits / size, "recall_at_5": recalls / size,
            "ndcg_at_5": ndcgs / size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path,
                        default=Path("experiments/benchmarks/movie_consensus/dev.jsonl"))
    parser.add_argument("--catalog", type=Path,
                        default=Path("experiments/data/processed/movie_catalog.jsonl"))
    parser.add_argument("--associations", type=Path, default=Path(
        "experiments/data/processed/redial_train_cooccurrence_tuning.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results/movie/hybrid_tuning.json"))
    args = parser.parse_args()
    cases = load(args.dev)
    lexical = JsonlMovieCatalog(args.catalog)
    lexical_result = score(cases, lambda case: [
        item.movie_id for item in lexical.search(request(case), limit=5)
    ])

    grid = []
    association_catalog = HybridMovieCatalog(args.catalog, args.associations, 1.0)
    def pure_ranker(case: dict) -> list[str]:
        candidates = [
            item for item in association_catalog.search(request(case), limit=200)
            if "train_cooccurrence_score" in item.metadata
        ]
        candidates.sort(key=lambda item: (-item.metadata["train_cooccurrence_score"], item.movie_id))
        return [item.movie_id for item in candidates]
    pure_cooccurrence = score(cases, pure_ranker)
    hybrid = HybridMovieCatalog(args.catalog, args.associations, 0.0)
    for step in range(0, 21):
        weight = step / 20
        hybrid.association_weight = weight
        result = score(cases, lambda case, catalog=hybrid: [
            item.movie_id for item in catalog.search(request(case), limit=5)
        ])
        grid.append({"association_weight": weight, **result})
    selected = max(grid, key=lambda row: (
        row["ndcg_at_5"], row["recall_at_5"], row["hit_at_5"], -row["association_weight"]
    ))
    report = {
        "selection_split": "ReDial train-derived dev; 30 conversations excluded from associations",
        "cases": len(cases),
        "selection_rule": "max NDCG@5, then Recall@5, Hit@5, then lower association weight",
        "lexical": lexical_result,
        "pure_cooccurrence": pure_cooccurrence,
        "selected": selected,
        "grid": grid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
