"""Condense all frozen extension suites into a single auditable evidence table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITES = (
    "boundary_v11_dev", "information_v11_dev", "seed_robustness_v11_dev",
    "missing_agent_v11_dev", "signal_ablation_v11_dev",
    "history_ablation_v11_dev", "heldout_glm_v11_dev",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"source": str(args.frozen_dir), "suites": {}, "registered_contrasts": []}
    for suite in SUITES:
        root = args.frozen_dir / f"{suite}.recomputed"
        analysis = load(root / "analysis.json")
        planned = load(root / "planned_contrasts.json")
        conditions = {}
        for name, row in analysis["conditions"].items():
            conditions[name] = {
                "n": row["n_valid"],
                "accuracy": row["decision_quality"],
                "high_effort_rate": row["high_effort_choice_rate"],
                "tokens_per_agent_round": row["mean_tokens_per_agent_round"],
                "parse_rate": row["parse_rate"],
                "terminal_failure_rate": row["terminal_record_failure_rate"],
            }
        contrasts = []
        for item in planned["contrasts"]:
            row = {"suite": suite, "name": item["name"], "left": item["left"],
                   "right": item["right"], "n_questions": item["n_complete_case_units"]}
            for endpoint in ("high_effort", "decision_quality", "tokens"):
                metric = item[endpoint]
                row[endpoint] = {
                    "delta": metric.get("delta", metric.get("mean_delta")),
                    "p_holm": metric.get("pvalue_holm_across_planned_contrasts"),
                    "p_raw": metric.get("pvalue"),
                }
            contrasts.append(row)
            result["registered_contrasts"].append(row)
        result["suites"][suite] = {
            "n_records": sum(x["n"] for x in conditions.values()),
            "conditions": conditions, "contrasts": contrasts,
        }
    result["total_records"] = sum(x["n_records"] for x in result["suites"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"suites": len(result["suites"]), "records": result["total_records"],
                      "contrasts": len(result["registered_contrasts"]), "out": str(args.out)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
