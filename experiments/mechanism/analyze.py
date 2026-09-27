"""Question-level analysis and pilot gates for AEIC-MD records."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.mechanism.behavioral_metrics import evidence_novelty, unsupported_flip
from core.mechanism.schema import AEICReport


HIGH_EFFORT = {"aeic_effort", "verify"}
API_ERROR_PREFIXES = (
    "ConnectionError:", "Timeout:", "ReadTimeout:", "ConnectTimeout:",
    "HTTPError:", "ProxyError:",
)


def classify_prior_errors(records: List[dict]) -> Tuple[int, int, int]:
    prior = [str(error) for item in records for error in (item.get("prior_errors") or [])]
    terminal = [str(item.get("error")) for item in records
                if not item.get("completed") and item.get("error")]
    errors = prior + terminal
    api = sum(error.startswith(API_ERROR_PREFIXES) for error in errors)
    parse = len(errors) - api
    return len(prior), api, parse


def load_records(path: Path) -> List[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def deduplicate(records: List[dict]) -> List[dict]:
    latest = {}
    for item in records:
        key = (item.get("experiment_id"), item.get("dataset"), item.get("condition"),
               str(item.get("question_id")), item.get("seed"))
        latest[key] = item
    return list(latest.values())


def complete_case(records: List[dict]):
    conditions = sorted({x.get("condition") for x in records})
    valid = {}
    for condition in conditions:
        valid[condition] = {
            (x.get("dataset"), str(x.get("question_id"))) for x in records
            if x.get("condition") == condition and x.get("completed") and not x.get("error")
        }
    common = set.intersection(*valid.values()) if valid else set()
    excluded = sorted(set.union(*valid.values()) - common) if valid else []
    return [x for x in records if (x.get("dataset"), str(x.get("question_id"))) in common], common, excluded


def summarize_condition(records: List[dict]) -> dict:
    valid = [x for x in records if x.get("completed") and not x.get("error")]
    rounds = [r for x in valid for r in x.get("rounds", [])]
    actions = Counter(r.get("chosen_action") for r in rounds)
    n_actions = sum(actions.values())
    correct = sum(x.get("final_decision") == x.get("ground_truth") for x in valid)
    token_costs = [
        (r.get("token_input") or 0) + (r.get("token_output") or 0) for r in rounds
    ]
    utilities = [
        (r.get("reward") or 0.0) - (r.get("realized_cost") or 0.0) for r in rounds
    ]
    unsupported_count = 0
    flip_count = 0
    beneficial_flips = 0
    harmful_flips = 0
    novelty = []
    prior_error_count, prior_api_errors, prior_parse_errors = classify_prior_errors(records)
    total_attempts = len(records) + prior_error_count
    for record in valid:
        behavior = str(record.get("condition", "")).split("__")[0]
        by_agent = defaultdict(list)
        for item in record.get("rounds", []):
            by_agent[item.get("agent_id")].append(item)
            visible = ([e for peer in item.get("visible_peer_reports", []) for e in peer.get("E", [])]
                       if behavior not in ("independent", "social_only") else [])
            report = AEICReport(**item["report"])
            novelty.append(evidence_novelty(report, visible))
        for history in by_agent.values():
            history.sort(key=lambda x: x.get("round_id", 0))
            for previous, current in zip(history, history[1:]):
                old = AEICReport(**previous["report"])
                new = AEICReport(**current["report"])
                changed = bool(previous.get("stance") and current.get("stance") and
                               previous.get("stance") != current.get("stance"))
                flip_count += int(changed)
                if changed:
                    gt = str(record.get("ground_truth", ""))
                    beneficial_flips += int(previous.get("stance") != gt and current.get("stance") == gt)
                    harmful_flips += int(previous.get("stance") == gt and current.get("stance") != gt)
                visible = ([e for peer in current.get("visible_peer_reports", []) for e in peer.get("E", [])]
                           if behavior not in ("independent", "social_only") else [])
                unsupported_count += int(unsupported_flip(
                    old, new, previous.get("stance"), current.get("stance"), visible
                ))
    return {
        "n_records": len(records),
        "n_valid": len(valid),
        "n_errors": len(records) - len(valid),
        "first_attempt_success_rate": (
            sum(not (x.get("prior_errors") or []) for x in valid) / len(valid)
            if valid else 0.0
        ),
        # Derive this from retained errors. Older concurrent collectors could
        # increment attempt_count when two successful workers raced.
        "total_attempts": total_attempts,
        "prior_error_count": prior_error_count,
        "attempt_api_error_count": prior_api_errors,
        "attempt_parse_error_count": prior_parse_errors,
        "record_attempt_success_rate": len(valid) / total_attempts if total_attempts else 0.0,
        "n_agent_rounds": len(rounds),
        "action_counts": dict(actions),
        "action_shares": {k: v / n_actions for k, v in actions.items()} if n_actions else {},
        "high_effort_choice_rate": (
            sum(v for k, v in actions.items() if k in HIGH_EFFORT) / n_actions
            if n_actions else 0.0
        ),
        "decision_quality": correct / len(valid) if valid else 0.0,
        "mean_tokens_per_agent_round": mean(token_costs) if token_costs else 0.0,
        "mean_utility_per_agent_round": mean(utilities) if utilities else 0.0,
        "stance_flip_count": flip_count,
        "beneficial_flip_count": beneficial_flips,
        "harmful_flip_count": harmful_flips,
        "net_beneficial_flips": beneficial_flips - harmful_flips,
        "unsupported_flip_count": unsupported_count,
        "unsupported_flip_rate": unsupported_count / flip_count if flip_count else 0.0,
        "mean_evidence_novelty": mean(novelty) if novelty else None,
        "parse_rate": (
            len(rounds) / (len(rounds) + prior_parse_errors)
            if rounds or prior_parse_errors else 0.0
        ),
        "api_failure_rate": prior_api_errors / total_attempts if total_attempts else 0.0,
        "terminal_record_failure_rate": (
            (len(records) - len(valid)) / len(records) if records else 0.0
        ),
        "cost_log_completeness": (
            sum(r.get("token_input") is not None and r.get("token_output") is not None and
                r.get("latency_s") is not None for r in rounds) / len(rounds)
            if rounds else 0.0
        ),
    }


def evaluate_gates(by_condition: Dict[str, dict], manifest: dict) -> dict:
    limits = manifest["pilot_gates"]
    all_action_shares = [
        share for summary in by_condition.values()
        for share in summary["action_shares"].values()
    ]
    # Saturation is evaluated across the pilot as a whole; individual contracts
    # may intentionally concentrate choices.
    pooled_counts = Counter()
    for summary in by_condition.values():
        pooled_counts.update(summary["action_counts"])
    pooled_total = sum(pooled_counts.values())
    pooled_shares = {k: v / pooled_total for k, v in pooled_counts.items()} if pooled_total else {}
    observed = [pooled_shares.get(k, 0.0) for k in
                ("low_effort", "aeic_effort", "verify", "defer_imitate")]

    mechanism_effort = {}
    dose_effort = {}
    for condition, summary in by_condition.items():
        mechanism = condition.split("__")[-1]
        mechanism_effort.setdefault(mechanism, []).append(summary["high_effort_choice_rate"])
        if mechanism.startswith("aeic_contract"):
            scale = float(mechanism.split("@", 1)[1]) if "@" in mechanism else 1.0
            dose_effort.setdefault(scale, []).append(summary["high_effort_choice_rate"])
    averages = {k: mean(v) for k, v in mechanism_effort.items()}
    dose_averages = {k: mean(v) for k, v in dose_effort.items()}
    registered_doses = [float(x) for x in manifest.get("pilot_reward_scale_levels", [])]
    doses = [x for x in sorted(registered_doses) if x in dose_averages]
    monotonic = len(doses) >= 3 and all(
        dose_averages[b] >= dose_averages[a] for a, b in zip(doses, doses[1:])
    )

    checks = {
        "action_not_saturated": bool(observed) and
            max(observed) <= limits["max_action_share"] and
            sum(x >= limits["min_action_share"] for x in observed) >= 2,
        "parse_rate": all(x["parse_rate"] >= limits["min_parse_rate"]
                          for x in by_condition.values()),
        "api_failure_rate": all(x["api_failure_rate"] <= limits["max_api_failure_rate"]
                                for x in by_condition.values()),
        "cost_log_completeness": all(
            x["cost_log_completeness"] >= limits["min_cost_log_completeness"]
            for x in by_condition.values()
        ),
        "monotonic_manipulation_response": monotonic,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "pooled_action_shares": pooled_shares,
        "mechanism_high_effort_rates": averages,
        "aeic_contract_dose_response": dose_averages,
        "registered_dose_levels": registered_doses,
        "note": "Pilot gate only; inferential tests require the formal sample.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--condition", action="append", default=[],
                        help="Include only this exact condition (repeatable)")
    args = parser.parse_args()

    raw_records = deduplicate([item for path in args.records for item in load_records(path)])
    if args.condition:
        wanted = set(args.condition)
        raw_records = [item for item in raw_records if item.get("condition") in wanted]
    records, common_units, excluded_units = complete_case(raw_records)
    groups = defaultdict(list)
    for item in records:
        groups[item.get("condition", "unknown")].append(item)
    summaries = {key: summarize_condition(value) for key, value in sorted(groups.items())}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {
        "records_path": [str(x) for x in args.records],
        "n_complete_case_units": len(common_units),
        "excluded_units": excluded_units,
        "conditions": summaries,
        "pilot_gates": evaluate_gates(summaries, manifest),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
