"""Build deterministic, method-blind human review packets without changing source files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def movie_packets(source: Path, output_dir: Path) -> None:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 150:
        raise ValueError(f"expected 150 movie rows, got {len(rows)}")
    if len({(row["case_id"], row["rank"]) for row in rows}) != len(rows):
        raise ValueError("duplicate movie review records")
    blinded = [{
        "blind_id": f"M{index:03d}",
        "query": row["query"],
        "rank": row["rank"],
        "movie_id": row["movie_id"],
        "title": row["title"],
        "year": row["year"],
        "genres": row["genres"],
        "relevance_0_2": "",
        "reason_factual_0_1": "",
        "list_fit_0_1": "",
        "hard_constraint_issue_0_1": "",
        "reviewer_note": "",
    } for index, row in enumerate(rows, 1)]
    rng = random.Random(20261015)
    rng.shuffle(blinded)
    fields = list(blinded[0])
    write_csv(output_dir / "movie_review_blind.csv", fields, blinded)
    key = [{"blind_id": row["blind_id"], "case_id": source_row["case_id"],
            "original_rank": source_row["rank"]}
           for row, source_row in zip(
               [{"blind_id": f"M{index:03d}"} for index in range(1, len(rows) + 1)], rows)]
    write_csv(output_dir / "movie_review_key.csv", list(key[0]), key)


def aeic_packets(source: Path, output_dir: Path) -> None:
    try:
        import openpyxl
    except ImportError as exc:
        raise SystemExit("openpyxl is required; use the bundled workspace Python") from exc
    workbook = openpyxl.load_workbook(source, data_only=True, read_only=True)
    sheet = workbook["审核员A"]
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    source_rows = [dict(zip(headers, [cell.value for cell in row]))
                   for row in sheet.iter_rows(min_row=2)]
    if len(source_rows) != 60:
        raise ValueError(f"expected 60 AEIC rows, got {len(source_rows)}")
    fields = ["盲审编号", "数据集", "问题", "待评报告", "响应哈希",
              "结论可识别(0/1)", "证据边界(1-5)", "推理可追溯(1-5)",
              "无依据断言(0/1)", "总体质量(1-5)", "备注"]
    base = [{field: (row.get(field) if field in {"盲审编号", "数据集", "问题", "待评报告", "响应哈希"} else "")
             for field in fields} for row in source_rows]
    for reviewer, seed in (("A", 202610151), ("B", 202610152)):
        packet = [dict(row) for row in base]
        random.Random(seed).shuffle(packet)
        write_csv(output_dir / f"aeic_review_{reviewer}_blind.csv", fields, packet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/results/human_review")
    args = parser.parse_args()
    movie_packets(ROOT / "experiments/results/movie_acceptance/manual_review.csv", args.output_dir)
    aeic_packets(ROOT / "docs/research/AEIC_report_interface_human_review.xlsx", args.output_dir)
    write_csv(args.output_dir / "review_assignments.csv",
              ["packet", "reviewer_id", "review_date", "completed_0_1"], [
                  {"packet": "aeic_review_A_blind.csv", "reviewer_id": "", "review_date": "", "completed_0_1": "0"},
                  {"packet": "aeic_review_B_blind.csv", "reviewer_id": "", "review_date": "", "completed_0_1": "0"},
                  {"packet": "movie_review_blind.csv", "reviewer_id": "", "review_date": "", "completed_0_1": "0"},
              ])
    for path in sorted(args.output_dir.glob("*.csv")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(ROOT)} {digest}")


if __name__ == "__main__":
    main()
