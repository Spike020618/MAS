"""Audit the final research evidence package without contacting external services."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments/results/mechanism/frozen"


def jsonl_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    frozen_expected = 5850
    frozen_actual = sum(jsonl_count(FROZEN / name / "question_level.jsonl")
                        for name in manifest["required_suites"])
    interface = jsonl_count(ROOT / "experiments/results/mechanism/interface/report_interface_v1.jsonl")
    decomposition_files = list((ROOT / "experiments/results/mechanism/decomposition").glob("*.jsonl"))
    decomposition = sum(jsonl_count(path) for path in decomposition_files)
    movie = json.loads((ROOT / "experiments/results/movie_acceptance/summary.json").read_text(encoding="utf-8"))
    redial = json.loads((ROOT / "experiments/results/movie/summary_hybrid_w085.json").read_text(encoding="utf-8"))
    retrieval = json.loads((ROOT / "experiments/results/movie/retrieval_ablation.json").read_text(encoding="utf-8"))
    tuning = json.loads((ROOT / "experiments/results/movie/hybrid_tuning.json").read_text(encoding="utf-8"))
    failures = json.loads((ROOT / "experiments/results/movie/failure_analysis.json").read_text(encoding="utf-8"))
    frozen_single_path = ROOT / "experiments/results/movie/hybrid_w085_single.jsonl"
    frozen_ours_path = ROOT / "experiments/results/movie/hybrid_w085_ours.jsonl"
    frozen_single_count = jsonl_count(frozen_single_path) if frozen_single_path.is_file() else 0
    frozen_ours_count = jsonl_count(frozen_ours_path) if frozen_ours_path.is_file() else 0
    redial_methods = {item["method"]: item for item in redial["methods"]}
    required = [
        "docs/research/claim_evidence_registry.md",
        "docs/research/transmission_chain_findings_2026-10-14.md",
        "docs/research/robustness_and_boundary_findings_2026-10-14.md",
        "docs/research/movie_application_evaluation_2026-10-14.md",
        "docs/research/movie_application_final_acceptance_2026-10-14.md",
        "docs/research/final_research_evidence_map_2026-10-14.md",
        "docs/research/final_delivery_checklist_2026-10-14.md",
        "docs/research/human_review_protocol_2026-10-14.md",
        "docs/research/AEIC_report_interface_human_review.xlsx",
        "docs/papers/figures/synthesis/mechanism_transmission_chain.svg",
        "docs/papers/figures/synthesis/round_accuracy_trajectory.svg",
        "docs/papers/figures/synthesis/robustness_synthesis.svg",
        "experiments/results/movie/run_manifest.json",
        "experiments/results/movie/retrieval_ablation.json",
        "experiments/results/movie/hybrid_tuning.json",
        "experiments/results/movie/failure_analysis.json",
        "experiments/results/movie/summary_hybrid_w085.json",
        "experiments/results/human_review/aeic_review_A_blind.csv",
        "experiments/results/human_review/aeic_review_B_blind.csv",
        "experiments/results/human_review/movie_review_blind.csv",
        "experiments/results/human_review/movie_review_key.csv",
        "experiments/results/human_review/review_assignments.csv",
        "experiments/prepare_human_review_packets.py",
        "experiments/summarize_human_review.py",
    ]
    checks = {
        "frozen_records": frozen_actual == frozen_expected,
        "interface_records": interface == 270,
        "decomposition_records": decomposition == 240,
        "formal_total": frozen_actual + interface + decomposition == 6360,
        "movie_entity_rate": movie["valid_entity_rate"] == 1.0,
        "movie_constraint_rate": movie["constraint_satisfaction_rate"] == 1.0,
        "movie_trace_rate": movie["trace_order_valid_rate"] == 1.0,
        "redial_paired_cases": redial["paired_case_sets_equal"],
        "redial_inputs_match": redial["benchmark_inputs_match"],
        "redial_complete": all(redial_methods[name]["cases"] == 100
                               for name in ("lexical", "single", "ours")),
        "redial_valid_entities": all(redial_methods[name]["valid_entity_rate"] == 1.0
                                     for name in ("lexical", "single", "ours")),
        "redial_constraints": all(redial_methods[name]["constraint_satisfaction_rate"] == 1.0
                                  for name in ("lexical", "single", "ours")),
        "redial_full_retrieval": retrieval["cases"] == 1203,
        "redial_full_inputs_match": retrieval["benchmark_inputs_match"],
        "redial_hybrid_retrieval_improves": (
            retrieval["metrics"]["hit_at_5"]["difference"] > 0
            and retrieval["hit_at_5_mcnemar"]["exact_two_sided_p"] < 0.05
        ),
        "redial_weight_selected_on_dev": tuning["selected"]["association_weight"] == 0.85,
        "redial_failure_taxonomy_complete": (
            failures["cases"] == 1203 and failures["taxonomy_is_mutually_exclusive"]
        ),
        "redial_frozen_weight_agent_rerun": (
            frozen_single_count == 100 and frozen_ours_count == 100
        ),
        "required_artifacts": all((ROOT / path).is_file() for path in required),
    }
    report = {
        "automated_checks_pass": all(checks.values()), "checks": checks,
        "record_counts": {"frozen": frozen_actual, "report_interface": interface,
                          "mechanism_decomposition": decomposition,
                          "formal_total": frozen_actual + interface + decomposition},
        "optional_human_validation": {
            "required_for_current_conclusions": False,
            "status": "not_run_no_reviewer_available",
            "prepared_packets": [
                "aeic_review_A_blind.csv",
                "aeic_review_B_blind.csv",
                "movie_review_blind.csv",
            ],
            "policy": "Do not fabricate labels; run only if independent reviewers become available.",
        },
        "redial": redial,
        "redial_retrieval_ablation": retrieval,
        "external_blocked_optional": [],
        "blocked_required": [],
    }
    out = ROOT / "docs/research/research_completion_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
