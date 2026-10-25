"""Recompute the mechanism transmission chain from frozen question-level records."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.core.mechanism.behavioral_metrics import evidence_copy_score, majority_stance


HIGH_EFFORT = {"aeic_effort", "verify"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def entropy(stances: list[str]) -> float:
    counts = Counter(x for x in stances if x)
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    raw = -sum((n / total) * math.log(n / total) for n in counts.values())
    return raw / math.log(len(counts))


def visible_evidence(item: dict) -> list[str]:
    return [str(e) for peer in item.get("visible_peer_reports", [])
            for e in peer.get("E", [])]


def summarize(records: list[dict]) -> dict:
    valid = [x for x in records if x.get("completed") and not x.get("error")]
    action_cells = defaultdict(list)
    resource_cells = defaultdict(list)
    round_cells = defaultdict(list)
    signal_cells = defaultdict(list)
    question_rows = []
    for record in valid:
        condition = record["condition"]
        dataset = record["dataset"]
        truth = str(record.get("ground_truth", "")).lower()
        by_round = defaultdict(list)
        by_agent = defaultdict(list)
        for item in record.get("rounds", []):
            round_id = int(item["round_id"])
            by_round[round_id].append(item)
            by_agent[int(item["agent_id"])].append(item)
            key = (condition, dataset, round_id)
            action_cells[key].append(str(item["chosen_action"]))
            traces = item.get("action_trace") or []
            execution = [x for x in traces if x.get("stage") != "choose"
                         and not str(x.get("stage", "")).startswith("format_repair")]
            resource_cells[key].append({
                "tokens": int(item.get("token_input") or 0) + int(item.get("token_output") or 0),
                "calls": len(traces), "execution_calls": len(execution),
                "latency_s": float(item.get("latency_s") or 0),
                "route_consistent": all(x.get("backend") for x in execution),
                "format_repairs": sum(str(x.get("stage", "")).startswith("format_repair") for x in traces),
            })
            report = item.get("report") or {}
            components = item.get("reward_components") or {}
            signal_cells[key].append({
                "complete_aeic": all(bool(report.get(x)) for x in ("A", "E", "I", "C")),
                "field_A": bool(report.get("A")), "field_E": bool(report.get("E")),
                "field_I": bool(report.get("I")), "field_C": bool(report.get("C")),
                "novel_evidence": float(components.get("component_novel_evidence", 0)),
                "verified_evidence": float(components.get("component_verified_evidence", 0)),
                "useful_correction": float(components.get("component_useful_correction", 0)),
                "unsupported_flip": float(components.get("component_unsupported_flip", 0)),
                "copy_score": evidence_copy_score(
                    type("R", (), {"E": report.get("E") or []})(), visible_evidence(item)
                ),
            })
        rounds_summary = []
        for round_id, items in sorted(by_round.items()):
            stances = [str(x.get("stance") or "").lower() for x in items]
            majority = majority_stance(stances)
            rounds_summary.append({
                "round": round_id, "majority": majority,
                "majority_correct": majority == truth,
                "agent_accuracy": sum(x == truth for x in stances) / len(stances),
                "disagreement_rate": 1 - max(Counter(stances).values()) / len(stances),
                "vote_entropy": entropy(stances),
            })
            round_cells[(condition, dataset, round_id)].append(rounds_summary[-1])
        beneficial = harmful = flips_with_new_evidence = flips_without_new_evidence = 0
        for history in by_agent.values():
            history.sort(key=lambda x: int(x["round_id"]))
            for before, after in zip(history, history[1:]):
                old, new = before.get("stance"), after.get("stance")
                if old and new and old != new:
                    beneficial += int(str(old).lower() != truth and str(new).lower() == truth)
                    harmful += int(str(old).lower() == truth and str(new).lower() != truth)
                    has_new = bool(visible_evidence(after))
                    flips_with_new_evidence += int(has_new)
                    flips_without_new_evidence += int(not has_new)
        question_rows.append({
            "condition": condition, "dataset": dataset,
            "question_id": str(record["question_id"]), "ground_truth": truth,
            "final_correct": str(record.get("final_decision", "")).lower() == truth,
            "high_effort_rate": sum(
                x.get("chosen_action") in HIGH_EFFORT for x in record.get("rounds", [])
            ) / len(record.get("rounds", [])),
            "rounds": rounds_summary, "beneficial_flips": beneficial,
            "harmful_flips": harmful, "flips_with_new_evidence": flips_with_new_evidence,
            "flips_without_new_evidence": flips_without_new_evidence,
        })

    def grouped_actions():
        output = []
        for (condition, dataset, round_id), actions in sorted(action_cells.items()):
            counts = Counter(actions)
            output.append({"condition": condition, "dataset": dataset, "round": round_id,
                           "n": len(actions), "counts": dict(counts),
                           "shares": {k: v / len(actions) for k, v in counts.items()},
                           "high_effort_rate": sum(counts[x] for x in HIGH_EFFORT) / len(actions)})
        return output

    def grouped_resources():
        output = []
        for (condition, dataset, round_id), items in sorted(resource_cells.items()):
            latencies = sorted(x["latency_s"] for x in items)
            output.append({"condition": condition, "dataset": dataset, "round": round_id,
                           "n": len(items), "mean_tokens": mean(x["tokens"] for x in items),
                           "mean_calls": mean(x["calls"] for x in items),
                           "mean_execution_calls": mean(x["execution_calls"] for x in items),
                           "mean_latency_s": mean(latencies),
                           "p95_latency_s": latencies[min(len(latencies)-1, math.ceil(.95*len(latencies))-1)],
                           "format_repair_rate": sum(x["format_repairs"] > 0 for x in items) / len(items),
                           "route_log_completeness": sum(x["route_consistent"] for x in items) / len(items)})
        return output

    def grouped_rounds():
        output = []
        for (condition, dataset, round_id), items in sorted(round_cells.items()):
            output.append({"condition": condition, "dataset": dataset, "round": round_id,
                           "n_questions": len(items),
                           "majority_accuracy": mean(x["majority_correct"] for x in items),
                           "agent_accuracy": mean(x["agent_accuracy"] for x in items),
                           "mean_disagreement_rate": mean(x["disagreement_rate"] for x in items),
                           "mean_vote_entropy": mean(x["vote_entropy"] for x in items)})
        return output

    return {"n_records": len(valid), "actions": grouped_actions(),
            "resources": grouped_resources(), "round_trajectory": grouped_rounds(),
            "question_level": question_rows}


def error_flow(question_rows: list[dict], left: str, right: str) -> dict:
    index = defaultdict(dict)
    for row in question_rows:
        index[(row["dataset"], row["question_id"])][row["condition"]] = row
    counts = Counter()
    strata = defaultdict(Counter)
    for (dataset, _qid), pair in index.items():
        if left not in pair or right not in pair:
            continue
        a, b = pair[left]["final_correct"], pair[right]["final_correct"]
        label = ("correct_correct" if a and b else "correct_wrong" if a else
                 "wrong_correct" if b else "wrong_wrong")
        counts[label] += 1
        strata[dataset][label] += 1
    return {"left": left, "right": right, "counts": dict(counts),
            "by_dataset": {k: dict(v) for k, v in strata.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--left", default="independent__no_mechanism@0.2")
    parser.add_argument("--right", default="independent__static_aeic@0.2")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [item for path in args.records for item in load(path)]
    result = summarize(records)
    flow = error_flow(result["question_level"], args.left, args.right)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "transmission_chain.json": {k: result[k] for k in ("n_records", "actions", "resources")},
        "round_trajectory.json": result["round_trajectory"],
        "error_flow.json": flow,
        "question_transmission.json": result["question_level"],
    }
    for name, payload in mapping.items():
        (args.out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"n_records": result["n_records"], "error_flow": flow["counts"],
                      "outputs": sorted(mapping)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
