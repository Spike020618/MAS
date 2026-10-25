"""Validate completed blind-review packets and summarize human judgments."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKETS = ROOT / "experiments/results/human_review"


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_scores(rows: list[dict], fields: dict[str, set[str]], label: str) -> None:
    errors = []
    for index, row in enumerate(rows, 2):
        for field, allowed in fields.items():
            value = str(row.get(field, "")).strip()
            if value not in allowed:
                errors.append(f"{label} row {index} {field}={value!r}")
    if errors:
        raise ValueError("incomplete or invalid review values:\n" + "\n".join(errors[:20]))


def mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def main() -> None:
    movie = read(PACKETS / "movie_review_blind.csv")
    aeic_a = read(PACKETS / "aeic_review_A_blind.csv")
    aeic_b = read(PACKETS / "aeic_review_B_blind.csv")
    assignments = read(PACKETS / "review_assignments.csv")
    if len(assignments) != 3 or any(
        not row.get("reviewer_id", "").strip()
        or not row.get("review_date", "").strip()
        or row.get("completed_0_1", "").strip() != "1"
        for row in assignments
    ):
        raise ValueError("review_assignments.csv is incomplete")
    require_scores(movie, {
        "relevance_0_2": {"0", "1", "2"}, "reason_factual_0_1": {"0", "1"},
        "list_fit_0_1": {"0", "1"}, "hard_constraint_issue_0_1": {"0", "1"},
    }, "movie")
    aeic_fields = {
        "结论可识别(0/1)": {"0", "1"}, "证据边界(1-5)": {"1", "2", "3", "4", "5"},
        "推理可追溯(1-5)": {"1", "2", "3", "4", "5"}, "无依据断言(0/1)": {"0", "1"},
        "总体质量(1-5)": {"1", "2", "3", "4", "5"},
    }
    require_scores(aeic_a, aeic_fields, "AEIC-A")
    require_scores(aeic_b, aeic_fields, "AEIC-B")
    by_a = {row["盲审编号"]: row for row in aeic_a}
    by_b = {row["盲审编号"]: row for row in aeic_b}
    if by_a.keys() != by_b.keys() or len(by_a) != 60:
        raise ValueError("AEIC reviewer case sets differ")
    agreements = {field: sum(by_a[key][field] == by_b[key][field] for key in by_a) / len(by_a)
                  for field in aeic_fields}
    report = {
        "complete": True,
        "movie": {"records": len(movie),
                  "mean_relevance_0_2": mean(movie, "relevance_0_2"),
                  "reason_factual_rate": mean(movie, "reason_factual_0_1"),
                  "list_fit_rate": mean(movie, "list_fit_0_1"),
                  "hard_constraint_issue_rate": mean(movie, "hard_constraint_issue_0_1")},
        "aeic": {"records_per_reviewer": len(aeic_a), "raw_agreement": agreements,
                 "reviewer_a_mean_quality": mean(aeic_a, "总体质量(1-5)"),
                 "reviewer_b_mean_quality": mean(aeic_b, "总体质量(1-5)")},
    }
    output = PACKETS / "human_review_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
