"""Question-clustered manipulation checks for AEIC behavioral observability."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from .statistical_tests import compare_maps, deduplicate, holm, load
except ImportError:  # Direct script execution.
    from statistical_tests import compare_maps, deduplicate, holm, load

from core.mechanism.behavioral_metrics import evidence_copy_score, evidence_novelty, unsupported_flip
from core.mechanism.schema import AEICReport


ENDPOINTS = ("unsupported_transition_rate", "supported_flip_rate",
             "evidence_novelty", "evidence_copy_score")


def record_metrics(record: dict) -> dict[str, float]:
    by_agent = defaultdict(list)
    novelty, copying = [], []
    for item in record.get("rounds", []):
        by_agent[int(item["agent_id"])].append(item)
        visible = [e for peer in item.get("visible_peer_reports", []) for e in peer.get("E", [])]
        report = AEICReport(**item["report"])
        novelty.append(evidence_novelty(report, visible))
        copying.append(evidence_copy_score(report, visible))
    transitions = unsupported = supported = 0
    for history in by_agent.values():
        history.sort(key=lambda item: int(item["round_id"]))
        for old, new in zip(history, history[1:]):
            transitions += 1
            if old.get("stance") == new.get("stance"):
                continue
            visible = [e for peer in new.get("visible_peer_reports", []) for e in peer.get("E", [])]
            is_unsupported = unsupported_flip(
                AEICReport(**old["report"]), AEICReport(**new["report"]),
                old.get("stance"), new.get("stance"), visible,
            )
            unsupported += int(is_unsupported)
            supported += int(not is_unsupported)
    return {
        "unsupported_transition_rate": unsupported / transitions if transitions else 0.0,
        "supported_flip_rate": supported / transitions if transitions else 0.0,
        "evidence_novelty": mean(novelty) if novelty else 0.0,
        "evidence_copy_score": mean(copying) if copying else 0.0,
    }


def build_report(records: list[dict], matrix: dict) -> dict:
    maps = defaultdict(dict)
    for item in records:
        if not item.get("completed") or item.get("error"):
            continue
        unit = (str(item.get("dataset")), str(item.get("question_id")), int(item.get("seed", -1)))
        metrics = record_metrics(item)
        for endpoint, value in metrics.items():
            maps[(str(item.get("condition")), endpoint)][unit] = value
    comparisons = []
    for spec in matrix.get("behavioral_contrasts", []):
        result = dict(spec)
        for endpoint in ENDPOINTS:
            result[endpoint] = compare_maps(
                maps[(spec["left"], endpoint)], maps[(spec["right"], endpoint)], "continuous")
        comparisons.append(result)
    for endpoint in ENDPOINTS:
        adjusted = holm([item[endpoint]["pvalue"] for item in comparisons])
        for item, value in zip(comparisons, adjusted):
            item[endpoint]["pvalue_holm_across_behavioral_contrasts"] = value
    return {
        "matrix_version": matrix.get("matrix_version"),
        "n_registered_contrasts": len(comparisons),
        "multiplicity": "Holm across registered behavioral contrasts within endpoint",
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if not matrix.get("behavioral_contrasts"):
        raise SystemExit("matrix has no registered behavioral contrasts")
    records = deduplicate([item for path in args.records for item in load(path)])
    report = build_report(records, matrix)
    report["records"] = [str(path) for path in args.records]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
