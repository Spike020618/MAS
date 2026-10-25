"""Resumable movie benchmark runner for lexical, single-agent, and AEIC consensus."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from system.game_consensus_recsys.deepseek import DeepSeekClient, DeepSeekConfig
from system.game_consensus_recsys.deepseek_agents import DeepSeekReviewAgent
from system.game_consensus_recsys.domain import SearchRequest
from system.game_consensus_recsys.movie_data import (
    CooccurrenceMovieCatalog, HybridMovieCatalog, JsonlMovieCatalog,
)
from system.game_consensus_recsys.workflow import ConsensusWorkflow, LLM_REVIEW_LIMIT


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def upsert(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(path) if path.is_file() else []
    by_case = {item["case_id"]: item for item in existing}
    by_case[row["case_id"]] = row
    ordered = sorted(by_case.values(), key=lambda item: item["case_id"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in ordered),
        encoding="utf-8",
    )
    temporary.replace(path)


def cost_from_reports(reports) -> tuple[int, int]:
    # Each batched agent call repeats its total on candidate reports; count once per agent.
    by_agent = {}
    for report in reports:
        by_agent[report.agent_id] = (report.token_cost, report.latency_ms)
    return sum(x[0] for x in by_agent.values()), sum(x[1] for x in by_agent.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["lexical", "single", "ours"])
    parser.add_argument("--benchmark", type=Path,
                        default=Path("experiments/benchmarks/movie_consensus/test.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("experiments/data/processed/movie_catalog.jsonl"))
    parser.add_argument("--retrieval", choices=["lexical", "cooccurrence", "hybrid"],
                        default="lexical")
    parser.add_argument("--associations", type=Path,
                        default=Path("experiments/data/processed/redial_train_cooccurrence.json"))
    parser.add_argument("--association-weight", type=float, default=0.85)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[],
                        help="rerun one or more case ids even when already present")
    args = parser.parse_args()
    prefix = "" if args.retrieval == "lexical" else f"{args.retrieval}_"
    output = args.output or Path(f"experiments/results/movie/{prefix}{args.method}.jsonl")
    done = {x["case_id"] for x in load_jsonl(output)} if output.is_file() else set()
    if args.retrieval == "hybrid":
        catalog = HybridMovieCatalog(args.catalog, args.associations, args.association_weight)
    elif args.retrieval == "cooccurrence":
        catalog = CooccurrenceMovieCatalog(args.catalog, args.associations)
    else:
        catalog = JsonlMovieCatalog(args.catalog)
    config = DeepSeekConfig.from_environment() if args.method != "lexical" else None
    if args.method != "lexical" and config is None:
        raise SystemExit("DeepSeek configuration is required for single/ours")
    client = DeepSeekClient(config) if config else None
    workflow = ConsensusWorkflow.from_environment(catalog=catalog) if args.method == "ours" else None

    cases = load_jsonl(args.benchmark)
    forced = set(args.case_id)
    if forced:
        cases = [case for case in cases if case["case_id"] in forced]
        missing = forced - {case["case_id"] for case in cases}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")
    if args.limit:
        cases = cases[:args.limit]
    for index, case in enumerate(cases, 1):
        if case["case_id"] in done and case["case_id"] not in forced:
            continue
        request = SearchRequest(
            case["query"], top_k=5, latency_budget_ms=30000,
            excluded_movie_ids=list(map(str, case.get("excluded_seen_movie_ids", []))),
            liked_movie_ids=list(map(str, case.get("liked_context_movie_ids", []))),
        )
        started = time.perf_counter()
        candidates = catalog.search(request, limit=50)
        excluded_seen = set(map(str, case.get("excluded_seen_movie_ids", [])))
        candidates = [candidate for candidate in candidates if candidate.movie_id not in excluded_seen]
        reports = []
        if args.method == "lexical":
            ranked = candidates[:5]
        elif args.method == "single":
            # Match the consensus workflow's frozen LLM shortlist. Reviewing the
            # full 50-item retrieval pool can exceed the JSON output budget and
            # makes the single-agent baseline incomparable with our method.
            review_candidates = candidates[:max(request.top_k, LLM_REVIEW_LIMIT)]
            reports = DeepSeekReviewAgent(client, "preference", "deepseek-single").review(
                request, review_candidates
            )
            score = {x.candidate_id: x.utility * x.confidence for x in reports}
            ranked = sorted(
                review_candidates,
                key=lambda x: (-score.get(x.movie_id, 0.0), -x.retrieval_score),
            )[:5]
        else:
            response = workflow.run(request)
            ranked = [x.candidate for x in response.decisions]
            reports = response.trace.reports
        tokens, report_latency = cost_from_reports(reports)
        ranked_ids = [x.movie_id for x in ranked]
        catalog_ids = set(catalog.movies)
        evidence_reports = [x for x in reports if x.evidence]
        row = {
            **case, "method": args.method, "ranked_movie_ids": ranked_ids,
            "invalid_entity_rate": sum(x not in catalog_ids for x in ranked_ids) / max(1, len(ranked_ids)),
            "constraint_satisfaction": not bool(set(ranked_ids) & set(case.get("excluded_seen_movie_ids", []))),
            "evidence_support_rate": (sum(any(e.verified for e in x.evidence) for x in evidence_reports)
                                      / len(evidence_reports) if evidence_reports else 0.0),
            "tokens": tokens, "reported_agent_latency_ms": report_latency,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        upsert(output, row)
        print(f"[{index}/{len(cases)}] {case['case_id']} {args.method}", flush=True)


if __name__ == "__main__":
    main()
