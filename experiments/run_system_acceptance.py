"""Run a deterministic application-level acceptance audit on the MovieLens catalog.

This audit verifies system properties that do not require subjective relevance
labels: valid catalog entities, hard-constraint enforcement, complete Agent
coverage, trace ordering, evidence references, and runtime. It is deliberately
reported separately from the ReDial recommendation-quality benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

from system.game_consensus_recsys.agents import ConstraintVerifierAgent, PreferenceAgent, RetrievalAgent
from system.game_consensus_recsys.domain import SearchRequest, normalize_genres
from system.game_consensus_recsys.movie_data import JsonlMovieCatalog
from system.game_consensus_recsys.workflow import ConsensusWorkflow


CASES = [
    ("drama_2010", "thought-provoking drama after 2010", ["Drama"], [], 2010, None),
    ("scifi_no_action", "cerebral science fiction without action", ["Sci-Fi"], ["Action"], 2000, None),
    ("romance_2000", "romantic films after 2000", ["Romance"], [], 2000, None),
    ("comedy_family", "light family comedy", ["Comedy"], [], 1990, None),
    ("animation_2000", "animated films after 2000", ["Animation"], [], 2000, None),
    ("mystery_no_horror", "mystery without horror", ["Mystery"], ["Horror"], 1990, None),
    ("documentary", "well rated documentary", ["Documentary"], [], 1990, None),
    ("crime_drama", "crime drama", ["Crime", "Drama"], [], 1990, None),
    ("fantasy_no_horror", "fantasy without horror", ["Fantasy"], ["Horror"], 1990, None),
    ("thriller_2010", "thriller after 2010", ["Thriller"], [], 2010, None),
    ("western", "classic western", ["Western"], [], None, 1999),
    ("war_drama", "war drama", ["War", "Drama"], [], 1980, None),
    ("musical", "musical films", ["Musical"], [], 1950, None),
    ("children_animation", "children animation", ["Children", "Animation"], [], 1990, None),
    ("adventure_no_action", "adventure without action", ["Adventure"], ["Action"], 1990, None),
    ("scifi_drama_2010", "thoughtful science fiction drama after 2010", ["Sci-Fi", "Drama"], ["Action"], 2010, None),
    ("romance_drama_2010", "romance drama after 2010", ["Romance", "Drama"], [], 2010, None),
    ("comedy_no_romance", "comedy without romance", ["Comedy"], ["Romance"], 2000, None),
    ("film_noir", "film noir", ["Film-Noir"], [], None, 1999),
    ("imax_scifi", "large-format science fiction", ["IMAX", "Sci-Fi"], [], 2000, None),
    ("horror_2000", "horror after 2000", ["Horror"], [], 2000, None),
    ("action_no_comedy", "serious action without comedy", ["Action"], ["Comedy"], 2000, None),
    ("drama_no_thriller", "quiet drama without thriller", ["Drama"], ["Thriller"], 2000, None),
    ("mystery_drama", "mystery drama", ["Mystery", "Drama"], [], 1990, None),
    ("fantasy_animation", "animated fantasy", ["Fantasy", "Animation"], [], 1990, None),
    ("crime_thriller", "crime thriller", ["Crime", "Thriller"], [], 1990, None),
    ("documentary_2010", "documentary after 2010", ["Documentary"], [], 2010, None),
    ("scifi_pre2000", "science fiction before 2000", ["Sci-Fi"], [], None, 1999),
    ("comedy_2015", "comedy after 2015", ["Comedy"], [], 2015, None),
    ("drama_1960_1990", "drama from 1960 to 1990", ["Drama"], [], 1960, 1990),
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path,
                        default=Path("experiments/data/processed/movie_catalog.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/results/movie_acceptance"))
    args = parser.parse_args()

    catalog = JsonlMovieCatalog(args.catalog)
    workflow = ConsensusWorkflow(
        catalog=catalog,
        agents=[RetrievalAgent(catalog), PreferenceAgent(), ConstraintVerifierAgent()],
    )
    rows = []
    audit_rows = []
    for case_id, query, required, excluded, min_year, max_year in CASES:
        request = SearchRequest(
            query=query, top_k=5, required_genres=required, excluded_genres=excluded,
            min_year=min_year, max_year=max_year,
        )
        started = time.perf_counter()
        response = workflow.run(request)
        latency_ms = (time.perf_counter() - started) * 1000
        required_set, excluded_set = normalize_genres(required), normalize_genres(excluded)
        valid, constraints, evidence = True, True, True
        ranked_ids = []
        for rank, decision in enumerate(response.decisions, 1):
            movie = decision.candidate
            ranked_ids.append(movie.movie_id)
            item = catalog.get(movie.movie_id)
            valid &= item is not None
            genres = normalize_genres(movie.genres)
            constraints &= required_set.issubset(genres) and not bool(excluded_set & genres)
            constraints &= min_year is None or movie.year >= min_year
            constraints &= max_year is None or movie.year <= max_year
            evidence &= bool(decision.accepted_evidence) and all(
                ev.reference for ev in decision.accepted_evidence
            )
            audit_rows.append({
                "case_id": case_id, "query": query, "rank": rank,
                "movie_id": movie.movie_id, "title": movie.title, "year": movie.year,
                "genres": "|".join(movie.genres), "score": f"{decision.score:.6f}",
                "relevance_0_2": "", "reason_factual_0_1": "", "list_fit_0_1": "",
                "reviewer_note": "",
            })
        events = [(event.stage, event.event) for event in response.trace.events]
        contract_index = events.index(("mechanism", "public_contract_committed"))
        completed = [i for i, event in enumerate(events) if event == ("deliberation", "agent_completed")]
        trace_order = bool(completed) and contract_index < min(completed)
        coverage_event = next(event for event in response.trace.events
                              if event.event == "coverage_checked")
        rows.append({
            "case_id": case_id, "query": query, "required_genres": required,
            "excluded_genres": excluded, "min_year": min_year, "max_year": max_year,
            "ranked_movie_ids": ranked_ids, "result_count": len(ranked_ids),
            "valid_entity": valid, "constraint_satisfaction": constraints,
            "evidence_references_present": evidence, "trace_order_valid": trace_order,
            "uniform_coverage": coverage_event.payload["uniform"],
            "complete_coverage": coverage_event.payload["complete"],
            "latency_ms": round(latency_ms, 3), "request_id": response.request_id,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "acceptance_results.jsonl"
    result_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in rows), encoding="utf-8")
    audit_path = args.output_dir / "manual_review.csv"
    with audit_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    latencies = [row["latency_ms"] for row in rows]
    checks = ["valid_entity", "constraint_satisfaction", "evidence_references_present",
              "trace_order_valid", "uniform_coverage", "complete_coverage"]
    summary = {
        "scope": "deterministic application acceptance; not recommendation-quality evidence",
        "cases": len(rows), "recommendations": sum(row["result_count"] for row in rows),
        **{key + "_rate": sum(bool(row[key]) for row in rows) / len(rows) for key in checks},
        "mean_latency_ms": round(sum(latencies) / len(latencies), 3),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "catalog_movies": len(catalog.movies),
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "manual_review_rows": len(audit_rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 影视系统验收结果", "",
        "> 本结果只验证工程正确性，不用于声称推荐质量优于基线。", "",
        f"- 测试请求：{summary['cases']} 条",
        f"- Top-K结果：{summary['recommendations']} 条",
        f"- 目录实体有效率：{summary['valid_entity_rate']:.2%}",
        f"- 硬约束满足率：{summary['constraint_satisfaction_rate']:.2%}",
        f"- 证据引用完整率：{summary['evidence_references_present_rate']:.2%}",
        f"- 合同先于行动的Trace顺序：{summary['trace_order_valid_rate']:.2%}",
        f"- Agent覆盖完整率：{summary['complete_coverage_rate']:.2%}",
        f"- 平均延迟：{summary['mean_latency_ms']:.1f} ms（离线确定性链路）",
        f"- P95延迟：{summary['p95_latency_ms']:.1f} ms（离线确定性链路）", "",
        "## 人工审核", "",
        "manual_review.csv已生成。审核人只需填写相关性、理由事实性、列表适配度和备注。",
        "该表共150行，分别对应30个请求的Top-5结果。", "",
        "## 结论边界", "",
        "ReDial正式离线评测仍需完整官方压缩包。正式评测完成前，不报告Hit@5、Recall@5或NDCG@5。",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
