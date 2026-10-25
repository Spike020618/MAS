"""Compare catalog retrieval and consensus workflow on explicit movie requests."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_system_acceptance import CASES
from system.game_consensus_recsys.agents import ConstraintVerifierAgent, PreferenceAgent, RetrievalAgent
from system.game_consensus_recsys.domain import SearchRequest, normalize_genres
from system.game_consensus_recsys.movie_data import JsonlMovieCatalog
from system.game_consensus_recsys.workflow import ConsensusWorkflow


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values)-1, max(0, math.ceil(q*len(values))-1))]


def metrics(items, request, catalog, evidence_complete=False, trace_valid=False):
    required, excluded = normalize_genres(request.required_genres), normalize_genres(request.excluded_genres)
    valid = [catalog.get(x.movie_id) is not None for x in items]
    constraint = []
    genres = set()
    for movie in items:
        current = normalize_genres(movie.genres); genres |= current
        constraint.append(required.issubset(current) and not bool(excluded & current)
                          and (request.min_year is None or movie.year >= request.min_year)
                          and (request.max_year is None or movie.year <= request.max_year))
    return {"valid_entities": sum(valid), "constraint_hits": sum(constraint),
            "recommendations": len(items), "unique_genres": len(genres),
            "unique_movies": len({x.movie_id for x in items}),
            "evidence_complete": evidence_complete, "trace_valid": trace_valid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("experiments/data/processed/movie_catalog.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("experiments/results/movie_acceptance/comparison.json"))
    args = parser.parse_args()
    catalog = JsonlMovieCatalog(args.catalog)
    workflow = ConsensusWorkflow(catalog=catalog, agents=[RetrievalAgent(catalog), PreferenceAgent(), ConstraintVerifierAgent()])
    rows = []
    for case_id, query, required, excluded, min_year, max_year in CASES:
        request = SearchRequest(query=query, top_k=5, required_genres=required,
                                excluded_genres=excluded, min_year=min_year, max_year=max_year)
        start = time.perf_counter(); lexical = catalog.search(request, limit=5)
        lexical_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter(); response = workflow.run(request)
        workflow_ms = (time.perf_counter()-start)*1000
        ranked = [x.candidate for x in response.decisions]
        evidence = all(x.accepted_evidence and all(e.reference for e in x.accepted_evidence)
                       for x in response.decisions)
        events = [(x.stage, x.event) for x in response.trace.events]
        trace = (("mechanism", "public_contract_committed") in events and
                 events.index(("mechanism", "public_contract_committed")) <
                 events.index(("deliberation", "agent_completed")))
        rows += [{"case_id": case_id, "method": "catalog_retrieval", "latency_ms": lexical_ms,
                  **metrics(lexical, request, catalog)},
                 {"case_id": case_id, "method": "consensus_workflow", "latency_ms": workflow_ms,
                  **metrics(ranked, request, catalog, evidence, trace)}]
    result = {"scope": "explicit-request application evaluation; no implicit preference labels",
              "cases": len(CASES), "methods": {}}
    for method in ("catalog_retrieval", "consensus_workflow"):
        group = [x for x in rows if x["method"] == method]
        nrec = sum(x["recommendations"] for x in group)
        lat = [x["latency_ms"] for x in group]
        result["methods"][method] = {
            "recommendations": nrec,
            "valid_entity_rate": sum(x["valid_entities"] for x in group)/nrec,
            "hard_constraint_rate": sum(x["constraint_hits"] for x in group)/nrec,
            "mean_unique_genres_at_5": sum(x["unique_genres"] for x in group)/len(group),
            "duplicate_free_rate": sum(x["unique_movies"] == x["recommendations"] for x in group)/len(group),
            "evidence_complete_rate": sum(x["evidence_complete"] for x in group)/len(group),
            "trace_valid_rate": sum(x["trace_valid"] for x in group)/len(group),
            "mean_latency_ms": sum(lat)/len(lat), "p95_latency_ms": percentile(lat,.95),
        }
    result["case_rows"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"cases": result["cases"], "methods": result["methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
