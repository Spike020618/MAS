"""
Analyze Dynamic Sycophancy Regulation experiment results.

Compares three conditions:
  debate_offline  — 3-round debate, no feedback
  debate_online   — 3-round debate, static quality feedback
  debate_dynamic  — 3-round debate, adaptive dynamics + differentiated prompts

Usage:
  python experiments/analyze_dynamic_results.py
  python experiments/analyze_dynamic_results.py --results_path experiments/results/online_stackelberg/online_ab_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
CONDITIONS = ["debate_offline", "debate_online", "debate_dynamic"]
COND_LABELS = {
    "debate_offline": "Baseline (no fb)",
    "debate_online": "Static (online)",
    "debate_dynamic": "Dynamic (adaptive)",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def accuracy_table(results):
    """Print per-dataset accuracy for each condition."""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for exp in results.get("experiments", []):
        if "error" in exp:
            continue
        key = (exp["dataset"], exp["condition"])
        stats[key]["total"] += 1
        if exp.get("maj_correct"):
            stats[key]["correct"] += 1

    print("\n" + "=" * 72)
    print("ACCURACY COMPARISON: Offline vs Static-Online vs Dynamic")
    print("=" * 72)

    header = f"{'Condition':<22}"
    for ds in DATASETS:
        header += f" {ds:>12}"
    header += f" {'Average':>10}"
    print(header)
    print("-" * len(header))

    for cond in CONDITIONS:
        label = COND_LABELS.get(cond, cond)
        line = f"{label:<22}"
        accs = []
        for ds in DATASETS:
            s = stats.get((ds, cond))
            if s and s["total"] > 0:
                acc = s["correct"] / s["total"] * 100
                line += f" {acc:>10.1f}% "
                accs.append(acc)
            else:
                line += f" {'---':>12}"
        if accs:
            line += f" {sum(accs)/len(accs):>9.1f}%"
        else:
            line += f" {'---':>10}"
        print(line)

    # Deltas
    print()
    for ref, cmp, label in [
        ("debate_offline", "debate_online", "Δ static - baseline"),
        ("debate_offline", "debate_dynamic", "Δ dynamic - baseline"),
        ("debate_online", "debate_dynamic", "Δ dynamic - static"),
    ]:
        line = f"{label:<22}"
        for ds in DATASETS:
            s_ref = stats.get((ds, ref))
            s_cmp = stats.get((ds, cmp))
            if s_ref and s_cmp and s_ref["total"] > 0 and s_cmp["total"] > 0:
                d = (s_cmp["correct"] / s_cmp["total"] -
                     s_ref["correct"] / s_ref["total"]) * 100
                line += f" {d:>+10.1f}pp"
            else:
                line += f" {'---':>12}"
        print(line)


def dynamics_analysis(results):
    """Analyze dynamics detection patterns."""
    dyn_entries = [e for e in results.get("experiments", [])
                   if e.get("condition") == "debate_dynamic" and "error" not in e
                   and e.get("dynamics_history")]

    if not dyn_entries:
        print("\nNo dynamic entries with dynamics_history found.")
        return

    print("\n" + "=" * 72)
    print("DYNAMICS ANALYSIS")
    print("=" * 72)

    # State distribution per dataset
    state_counts = defaultdict(lambda: Counter())
    template_counts = defaultdict(lambda: Counter())
    for exp in dyn_entries:
        ds = exp["dataset"]
        for dyn in exp["dynamics_history"]:
            state_counts[ds][dyn["state"]] += 1
            tpl = dyn.get("template_applied", "n/a")
            if tpl != "n/a":
                template_counts[ds][tpl] += 1

    print("\nDetected states (per round transition):")
    print(f"{'Dataset':<15} {'sycophantic':>12} {'deadlock':>12} {'healthy':>12}")
    print("-" * 55)
    for ds in DATASETS:
        sc = state_counts.get(ds, Counter())
        total = sum(sc.values()) or 1
        line = f"{ds:<15}"
        for state in ["sycophantic", "deadlock", "healthy"]:
            n = sc.get(state, 0)
            line += f" {n:>5} ({n/total*100:4.0f}%)"
        print(line)

    print("\nTemplates applied (excluding final observation round):")
    print(f"{'Dataset':<15} {'anti_syco':>12} {'exchange':>12} {'neutral':>12}")
    print("-" * 55)
    for ds in DATASETS:
        tc = template_counts.get(ds, Counter())
        total = sum(tc.values()) or 1
        line = f"{ds:<15}"
        for tpl in ["anti_sycophancy", "encourage_exchange", "neutral"]:
            n = tc.get(tpl, 0)
            line += f" {n:>5} ({n/total*100:4.0f}%)"
        print(line)

    # Accuracy when anti-sycophancy was triggered vs not
    print("\nAccuracy by dynamics state (dynamic condition only):")
    state_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for exp in dyn_entries:
        states = exp.get("dynamic_states", [])
        has_syco = "sycophantic" in states
        has_dead = "deadlock" in states
        key = "sycophantic" if has_syco else ("deadlock" if has_dead else "healthy")
        state_acc[key]["total"] += 1
        if exp.get("maj_correct"):
            state_acc[key]["correct"] += 1

    for state in ["sycophantic", "deadlock", "healthy"]:
        s = state_acc[state]
        if s["total"] > 0:
            acc = s["correct"] / s["total"] * 100
            print(f"  {state:<15}: {acc:5.1f}% ({s['correct']}/{s['total']})")
        else:
            print(f"  {state:<15}: --- (0 questions)")


def sycophancy_rate_analysis(results):
    """Compute sycophancy rate: agents flipping from correct to incorrect stance."""
    print("\n" + "=" * 72)
    print("SYCOPHANCY RATE ANALYSIS")
    print("=" * 72)

    syco_rates = defaultdict(lambda: {"flips_to_wrong": 0, "total_flips": 0,
                                      "questions": 0})
    for exp in results.get("experiments", []):
        if "error" in exp or not exp.get("dynamics_history"):
            continue
        cond = exp["condition"]
        gt = exp["ground_truth"]
        for dyn in exp["dynamics_history"]:
            prev_st = dyn.get("prev_stances", [])
            cur_st = dyn.get("cur_stances", [])
            for i in range(len(prev_st)):
                if prev_st[i] != cur_st[i]:
                    syco_rates[cond]["total_flips"] += 1
                    if prev_st[i] == gt and cur_st[i] != gt:
                        syco_rates[cond]["flips_to_wrong"] += 1
        syco_rates[cond]["questions"] += 1

    if not syco_rates:
        print("  No dynamics data available for sycophancy rate analysis.")
        return

    print(f"\n{'Condition':<22} {'Flips→Wrong':>12} {'Total Flips':>12} {'Syco Rate':>10}")
    print("-" * 60)
    for cond in CONDITIONS:
        s = syco_rates.get(cond)
        if s and s["total_flips"] > 0:
            rate = s["flips_to_wrong"] / s["total_flips"] * 100
            print(f"{COND_LABELS.get(cond, cond):<22} "
                  f"{s['flips_to_wrong']:>12} {s['total_flips']:>12} "
                  f"{rate:>9.1f}%")
        elif s:
            print(f"{COND_LABELS.get(cond, cond):<22} {'0':>12} {'0':>12} {'---':>10}")


def per_question_comparison(results):
    """Case-by-case comparison of the three conditions on shared questions."""
    by_q = defaultdict(dict)
    for exp in results.get("experiments", []):
        if "error" in exp:
            continue
        key = (exp["dataset"], exp["question_id"])
        by_q[key][exp["condition"]] = exp

    shared = {k: v for k, v in by_q.items()
              if all(c in v for c in CONDITIONS)}

    if not shared:
        print("\nNo questions shared across all 3 conditions yet.")
        return

    print("\n" + "=" * 72)
    print(f"CASE-BY-CASE COMPARISON ({len(shared)} shared questions)")
    print("=" * 72)

    patterns = Counter()
    for (ds, qid), exps in sorted(shared.items()):
        results_tuple = tuple(
            "✓" if exps[c].get("maj_correct") else "✗" for c in CONDITIONS)
        patterns[results_tuple] += 1

    print(f"\n{'Offline':>8} {'Static':>8} {'Dynamic':>8}  {'Count':>6}  {'Pct':>6}")
    print("-" * 45)
    for pat, cnt in patterns.most_common():
        pct = cnt / len(shared) * 100
        print(f"{pat[0]:>8} {pat[1]:>8} {pat[2]:>8}  {cnt:>6}  {pct:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Dynamic Sycophancy Regulation results")
    parser.add_argument("--results_path",
                        default="experiments/results/online_stackelberg/online_ab_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        print(f"Results file not found: {args.results_path}")
        print("Run the experiment first:")
        print("  python experiments/run_online_stackelberg.py "
              "--conditions debate_offline debate_online debate_dynamic")
        sys.exit(1)

    results = load_results(args.results_path)
    n_total = len(results.get("experiments", []))
    n_errors = sum(1 for e in results.get("experiments", []) if "error" in e)
    print(f"Loaded {n_total} entries ({n_errors} errors) from {args.results_path}")

    accuracy_table(results)
    dynamics_analysis(results)
    sycophancy_rate_analysis(results)
    per_question_comparison(results)
    print()


if __name__ == "__main__":
    main()
