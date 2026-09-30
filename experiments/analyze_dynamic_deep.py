"""
Deep analysis of Dynamic Sycophancy Regulation experiment results.

Produces:
  1. Statistical tests (McNemar + bootstrap CI)
  2. Per-question behavioral patterns
  3. Threshold sensitivity analysis
  4. SCR/MR/CS distributions by outcome
  5. Error pattern analysis

Usage:
  python experiments/analyze_dynamic_deep.py
  python experiments/analyze_dynamic_deep.py --results_path experiments/results/online_stackelberg/online_ab_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

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


def _dedup_experiments(raw_experiments):
    """Keep only the latest (last) entry per (dataset, condition, question_id)."""
    by_key = {}
    for exp in raw_experiments:
        key = (exp["dataset"], exp["condition"], exp["question_id"])
        by_key[key] = exp  # later entries overwrite
    return list(by_key.values())


def accuracy_summary(experiments):
    """Return per-condition, per-dataset accuracy dict."""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for exp in experiments:
        if "error" in exp:
            continue
        key = (exp["dataset"], exp["condition"])
        stats[key]["total"] += 1
        if exp.get("maj_correct"):
            stats[key]["correct"] += 1
    return stats


def mcnemar_test(experiments):
    """McNemar test for paired comparisons between conditions."""
    from scipy.stats import binomtest

    print("\n" + "=" * 72)
    print("McNEMAR TESTS (paired, per-dataset)")
    print("=" * 72)

    by_q = defaultdict(dict)
    for exp in experiments:
        if "error" in exp:
            continue
        key = (exp["dataset"], exp["question_id"])
        by_q[key][exp["condition"]] = exp

    pairs = [
        ("debate_offline", "debate_online", "Baseline vs Static"),
        ("debate_offline", "debate_dynamic", "Baseline vs Dynamic"),
        ("debate_online", "debate_dynamic", "Static vs Dynamic"),
    ]

    for c1, c2, label in pairs:
        print(f"\n--- {label} ---")
        for ds in DATASETS:
            b, c, n1, n2 = 0, 0, 0, 0
            for (d, qid), exps in by_q.items():
                if d != ds:
                    continue
                if c1 not in exps or c2 not in exps:
                    continue
                r1 = exps[c1].get("maj_correct")
                r2 = exps[c2].get("maj_correct")
                if r1 and not r2:
                    b += 1
                elif not r1 and r2:
                    c += 1
                if r1:
                    n1 += 1
                if r2:
                    n2 += 1
            n = b + c
            if n == 0:
                print(f"  {ds:<15}: no discordant pairs")
                continue
            p = binomtest(min(b, c), n, 0.5).pvalue
            acc1 = n1 / (n1 + n2 - (b + c - n)) * 100 if n > 0 else 0
            print(f"  {ds:<15}: C1→wrong,C2→right={c:>3}  C1→right,C2→wrong={b:>3}  "
                  f"p={p:.4f}" + (" *" if p < 0.05 else ""))


def bootstrap_ci(experiments, n_bootstrap=10000, seed=42):
    """Bootstrap 95% CI for pairwise accuracy differences."""
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 72)
    print("BOOTSTRAP 95% CI (per-dataset, per-pair)")
    print("=" * 72)

    by_q = defaultdict(dict)
    for exp in experiments:
        if "error" in exp:
            continue
        key = (exp["dataset"], exp["question_id"])
        by_q[key][exp["condition"]] = exp

    pairs = [
        ("debate_offline", "debate_online", "Static-Baseline"),
        ("debate_offline", "debate_dynamic", "Dynamic-Baseline"),
        ("debate_online", "debate_dynamic", "Dynamic-Static"),
    ]

    for c1, c2, label in pairs:
        print(f"\n--- {label} ---")
        for ds in DATASETS:
            shared = [(exps[c1].get("maj_correct"), exps[c2].get("maj_correct"))
                      for (d, _), exps in by_q.items()
                      if d == ds and c1 in exps and c2 in exps]
            if len(shared) < 5:
                print(f"  {ds:<15}: insufficient data (n={len(shared)})")
                continue
            diffs = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(shared), len(shared))
                s_acc1 = np.mean([shared[i][0] for i in idx])
                s_acc2 = np.mean([shared[i][1] for i in idx])
                diffs.append((s_acc2 - s_acc1) * 100)
            diffs = np.sort(diffs)
            lo, hi = diffs[250], diffs[9750]
            mean = np.mean(diffs)
            sig = "*" if lo * hi > 0 else ""  # CI excludes zero
            print(f"  {ds:<15}: {mean:+6.2f}pp  [95% CI: {lo:+6.2f}, {hi:+6.2f}] "
                  f"{'SIG' if sig else 'ns'}")


def threshold_sensitivity(experiments):
    """Grid-search optimal detection thresholds from recorded dynamics data.

    Uses a replay approach: for each question in dynamic condition, we have
    recorded SCR, MR, CS values and the actual intervention applied. We simulate
    what would happen under different thresholds and identify which threshold
    combination maximizes overall accuracy.

    Since we can't re-run with different thresholds (path-dependency), we use
    a coarser analysis: for each transition, we compare which state-label a
    threshold tuple would assign vs. the actual accuracy outcome, and compute
    the "regret" (correct→intervention or error→no-intervention).
    """
    dyn_exps = [e for e in experiments
                 if e.get("condition") == "debate_dynamic"
                 and "error" not in e
                 and e.get("dynamics_history")]

    if not dyn_exps:
        print("\nNo dynamic history data for threshold analysis.")
        return

    print("\n" + "=" * 72)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 72)
    print("(Grid search over τ_h, τ_l using recorded dynamics metrics)")

    # Collect all transitions with actual SCR, MR, and question outcome
    transitions = []
    for exp in dyn_exps:
        q_correct = exp.get("maj_correct", False)
        q_ds = exp["dataset"]
        for dyn in exp["dynamics_history"]:
            scr = dyn.get("stance_change_rate", 0)
            mr = dyn.get("majority_ratio", 0)
            transitions.append({
                "scr": scr, "mr": mr, "dataset": q_ds,
                "question_correct": q_correct,
                "actual_state": dyn.get("state", "healthy"),
            })

    # Grid search
    tau_h_vals = np.arange(0.3, 0.95, 0.05)
    tau_l_vals = np.arange(0.0, 0.45, 0.05)

    best_acc = 0
    best_params = (0.6, 0.2)

    print(f"\n{'tau_h':>6} {'tau_l':>6}  {'Syco%':>7} {'Dead%':>7} "
          f"{'Acc(syco)':>10} {'Acc(dead)':>10} {'Acc(healthy)':>13}")

    results = []
    for tau_h in tau_h_vals:
        for tau_l in tau_l_vals:
            if tau_l >= tau_h:
                continue

            states = {"sycophantic": [], "deadlock": [], "healthy": []}
            for t in transitions:
                if t["scr"] >= tau_h:
                    states["sycophantic"].append(t["question_correct"])
                elif t["scr"] <= tau_l and t["mr"] < 0.8:
                    states["deadlock"].append(t["question_correct"])
                else:
                    states["healthy"].append(t["question_correct"])

            syco_acc = np.mean(states["sycophantic"]) * 100 if states["sycophantic"] else 0
            dead_acc = np.mean(states["deadlock"]) * 100 if states["deadlock"] else 0
            hlth_acc = np.mean(states["healthy"]) * 100 if states["healthy"] else 0

            # Weighted by: detecting pathology → intervention helps
            # Optimal thresholds separate high-error from low-error transitions
            results.append((tau_h, tau_l, syco_acc, dead_acc, hlth_acc,
                           len(states["sycophantic"]), len(states["deadlock"])))

    # Show top 10 by worst-state accuracy (sycophantic accuracy)
    results.sort(key=lambda x: -(x[2]) if x[5] > 3 else 999)

    for tau_h, tau_l, sa, da, ha, ns, nd in results[:15]:
        pct_s = ns / len(transitions) * 100
        pct_d = nd / len(transitions) * 100
        print(f"  {tau_h:4.2f}  {tau_l:4.2f}  "
              f"{pct_s:>5.1f}%  {pct_d:>5.1f}%  "
              f"{sa:>8.1f}%  {da:>10.1f}%  {ha:>11.1f}%")

    # Show current config
    cur_syco = [t["question_correct"] for t in transitions if t["actual_state"] == "sycophantic"]
    cur_dead = [t["question_correct"] for t in transitions if t["actual_state"] == "deadlock"]
    cur_hlth = [t["question_correct"] for t in transitions if t["actual_state"] == "healthy"]
    print(f"\n  Current (0.60, 0.20): syco_acc={np.mean(cur_syco)*100:.1f}% "
          f"dead_acc={np.mean(cur_dead)*100:.1f}% "
          f"healthy_acc={np.mean(cur_hlth)*100:.1f}%")


def behavioral_patterns(experiments):
    """Analyze when dynamic helps, hurts, or has no effect."""
    print("\n" + "=" * 72)
    print("BEHAVIORAL PATTERN ANALYSIS")
    print("=" * 72)

    by_q = defaultdict(dict)
    for exp in experiments:
        if "error" in exp:
            continue
        key = (exp["dataset"], exp["question_id"])
        by_q[key][exp["condition"]] = exp

    # Compare dynamic vs baseline on shared questions
    patterns = Counter()
    dyn_wins = []
    dyn_loses = []
    for (ds, qid), exps in sorted(by_q.items()):
        c_base = exps.get("debate_offline", {})
        c_dyn = exps.get("debate_dynamic", {})
        if not c_base or not c_dyn:
            continue
        b_corr = c_base.get("maj_correct", False)
        d_corr = c_dyn.get("maj_correct", False)
        if b_corr and not d_corr:
            patterns["dyn_hurt"] += 1
            dyn_loses.append((ds, qid, c_dyn.get("question", "")))
        elif not b_corr and d_corr:
            patterns["dyn_help"] += 1
            dyn_wins.append((ds, qid, c_dyn.get("question", "")))
        elif b_corr and d_corr:
            patterns["both_correct"] += 1
        else:
            patterns["both_wrong"] += 1

    total = sum(patterns.values())
    for k in ["dyn_help", "dyn_hurt", "both_correct", "both_wrong"]:
        n = patterns.get(k, 0)
        print(f"  {k:<15}: {n:>3} ({n/total*100:5.1f}%)")

    if dyn_wins:
        print(f"\n  Dynamic HELPS (baseline wrong → dynamic correct, n={len(dyn_wins)}):")
        for ds, qid, q in dyn_wins[:5]:
            print(f"    [{ds}] Q{qid}: {q[:100]}...")

    if dyn_loses:
        print(f"\n  Dynamic HURTS (baseline correct → dynamic wrong, n={len(dyn_loses)}):")
        for ds, qid, q in dyn_loses[:5]:
            print(f"    [{ds}] Q{qid}: {q[:100]}...")

    # Per-dataset pattern breakdown
    print(f"\n  {'Dataset':<15} {'Help':>6} {'Hurt':>6} {'BothOK':>8} {'BothWrong':>10} "
          f"{'NetΔ':>8}")
    for ds in DATASETS:
        help_n = sum(1 for (d, _), _ in dyn_wins if d == ds) if False else \
            sum(1 for (d, qid), exps in by_q.items()
                if d == ds
                and "debate_offline" in exps and "debate_dynamic" in exps
                and not exps["debate_offline"].get("maj_correct")
                and exps["debate_dynamic"].get("maj_correct"))
        hurt_n = sum(1 for (d, qid), exps in by_q.items()
                if d == ds
                and "debate_offline" in exps and "debate_dynamic" in exps
                and exps["debate_offline"].get("maj_correct")
                and not exps["debate_dynamic"].get("maj_correct"))
        both_ok = sum(1 for (d, qid), exps in by_q.items()
                if d == ds
                and "debate_offline" in exps and "debate_dynamic" in exps
                and exps["debate_offline"].get("maj_correct")
                and exps["debate_dynamic"].get("maj_correct"))
        both_wr = sum(1 for (d, qid), exps in by_q.items()
                if d == ds
                and "debate_offline" in exps and "debate_dynamic" in exps
                and not exps["debate_offline"].get("maj_correct")
                and not exps["debate_dynamic"].get("maj_correct"))
        net = help_n - hurt_n
        print(f"  {ds:<15} {help_n:>4}  {hurt_n:>4}  {both_ok:>6}  {both_wr:>8}  "
              f"{net:+5}")


def dynamics_distribution(experiments):
    """Distribution of SCR, MR, CS metrics."""
    print("\n" + "=" * 72)
    print("DYNAMICS METRICS DISTRIBUTION")
    print("=" * 72)

    dyn_exps = [e for e in experiments
                 if e.get("condition") == "debate_dynamic"
                 and "error" not in e
                 and e.get("dynamics_history")]

    if not dyn_exps:
        return

    scrs, mrs, css = [], [], []
    scr_by_state = defaultdict(list)
    for exp in dyn_exps:
        for dyn in exp["dynamics_history"]:
            scrs.append(dyn.get("stance_change_rate", 0))
            mrs.append(dyn.get("majority_ratio", 0))
            css.append(dyn.get("convergence_speed", 0))
            scr_by_state[dyn.get("state", "healthy")].append(dyn.get("stance_change_rate", 0))

    print(f"\n  SCR: mean={np.mean(scrs):.3f}, median={np.median(scrs):.3f}, "
          f"std={np.std(scrs):.3f}, range=[{min(scrs):.2f},{max(scrs):.2f}]")
    print(f"  MR:  mean={np.mean(mrs):.3f}, median={np.median(mrs):.3f}, "
          f"std={np.std(mrs):.3f}")
    print(f"  CS:  mean={np.mean(css):.3f}, median={np.median(css):.3f}, "
          f"std={np.std(css):.3f}")

    # SCR histogram
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(f"\n  SCR distribution:")
    for i in range(len(bins)-1):
        n = sum(1 for s in scrs if bins[i] <= s < bins[i+1])
        bar = "#" * (n // max(1, max(n for _ in range(1))))
        print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {n:>3} ({n/len(scrs)*100:5.1f}%) {bar}")

    print(f"\n  SCR by state:")
    for state in ["healthy", "sycophantic", "deadlock"]:
        vals = scr_by_state.get(state, [])
        print(f"    {state:<14}: mean={np.mean(vals):.3f}, n={len(vals)}")

    # Accuracy by SCR bin
    scr_acc = defaultdict(list)
    for exp in dyn_exps:
        for dyn in exp["dynamics_history"]:
            scr = dyn.get("stance_change_rate", 0)
            bin_key = int(scr * 5) / 5  # 0, 0.2, 0.4, ...
            if bin_key >= 0.6:
                bin_key = 0.6
            scr_acc[bin_key].append(exp.get("maj_correct", False))

    print(f"\n  Accuracy by SCR bin:")
    for bin_key in [0.0, 0.2, 0.4, 0.6]:
        vals = scr_acc.get(bin_key, [])
        if vals:
            print(f"    SCR ∈ [{bin_key:.1f}, {bin_key+0.2:.1f}): "
                  f"acc={np.mean(vals)*100:.1f}% (n={len(vals)})")


def optimal_threshold_learning(experiments):
    """Learn optimal thresholds from observed data.

    For each transition, we know:
      - SCR, MR (the metrics)
      - Whether the question was ultimately correct
      - What state was assigned and what intervention was applied

    We treat each transition as a data point: "given (SCR, MR), should we
    intervene (state=sycophantic/deadlock) or not (state=healthy)?"
    The ground-truth answer is whether the question was ultimately correct.
    """
    print("\n" + "=" * 72)
    print("OPTIMAL THRESHOLD LEARNING")
    print("=" * 72)

    dyn_exps = [e for e in experiments
                 if e.get("condition") == "debate_dynamic"
                 and "error" not in e
                 and e.get("dynamics_history")]

    if not dyn_exps:
        return

    # Collect (SCR, MR, question_correct, actual_state) tuples
    points = []
    for exp in dyn_exps:
        q_correct = exp.get("maj_correct")
        for dyn in exp["dynamics_history"]:
            points.append({
                "scr": dyn.get("stance_change_rate", 0),
                "mr": dyn.get("majority_ratio", 0),
                "cs": dyn.get("convergence_speed", 0),
                "correct": q_correct,
                "state": dyn.get("state", "healthy"),
            })

    # For each possible (tau_h, tau_l), compute:
    #   - Number of transitions classified as sycophantic/deadlock (would intervene)
    #   - Accuracy of intervened vs non-intervened groups
    # Goal: find thresholds where intervened transitions have lower accuracy
    #        (i.e., intervention is actually needed)

    tau_h_range = np.arange(0.35, 0.90, 0.05)
    tau_l_range = np.arange(0.05, 0.40, 0.05)

    best_separation = -1
    best_params = None

    print(f"\n  Searching (tau_h, tau_l) for max accuracy separation...")
    candidates = []
    for th in tau_h_range:
        for tl in tau_l_range:
            if tl >= th:
                continue
            intervened_acc = []
            non_intervened_acc = []
            for p in points:
                if p["scr"] >= th:
                    intervened_acc.append(p["correct"])
                elif p["scr"] <= tl and p["mr"] < 0.8:
                    intervened_acc.append(p["correct"])
                else:
                    non_intervened_acc.append(p["correct"])

            int_mean = np.mean(intervened_acc) if intervened_acc else 0.5
            nonint_mean = np.mean(non_intervened_acc) if non_intervened_acc else 0.5
            n_int = len(intervened_acc)
            n_noint = len(non_intervened_acc)

            # Separation: lower intervened accuracy → better detection
            separation = nonint_mean - int_mean
            candidates.append((th, tl, separation, int_mean, nonint_mean, n_int, n_noint))

    candidates.sort(key=lambda x: (-x[2], x[5]))  # Sort by separation, then fewer interventions

    print(f"  {'tau_h':>6} {'tau_l':>6} {'Separation':>12} {'Intv Acc':>9} "
          f"{'Non-Int Acc':>11} {'Intv N':>7} {'Non-Int N':>9}")
    for th, tl, sep, ia, na, ni, nn in candidates[:10]:
        print(f"  {th:5.2f}  {tl:5.2f}  {sep:>+12.3f}  {ia*100:>7.1f}%  "
              f"{na*100:>9.1f}%  {ni:>5}  {nn:>7}")

    best = candidates[0]
    print(f"\n  Best: tau_h={best[0]:.2f}, tau_l={best[1]:.2f} "
          f"(separation={best[2]:.3f})")
    print(f"  Current (0.60, 0.20) vs best ({best[0]:.2f}, {best[1]:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Deep analysis of dynamic experiment")
    parser.add_argument("--results_path",
                        default="experiments/results/online_stackelberg/online_ab_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        print(f"Results file not found: {args.results_path}")
        sys.exit(1)

    results = load_results(args.results_path)
    raw = results.get("experiments", [])
    experiments = _dedup_experiments(raw)

    valid = [e for e in experiments if "error" not in e]
    errors = [e for e in experiments if "error" in e]
    print(f"Loaded {len(valid)} valid experiments ({len(errors)} errors, "
          f"{len(raw)-len(experiments)} duplicates removed)")

    # Summary table
    stats = accuracy_summary(experiments)
    print("\n" + "=" * 72)
    print("ACCURACY SUMMARY (n=50 per condition)")
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
        print(line)

    # Deltas
    print()
    pairs = [
        ("debate_offline", "debate_online", "Δ static - baseline"),
        ("debate_offline", "debate_dynamic", "Δ dynamic - baseline"),
        ("debate_online", "debate_dynamic", "Δ dynamic - static"),
    ]
    for c1, c2, label in pairs:
        line = f"{label:<22}"
        for ds in DATASETS:
            s1 = stats.get((ds, c1))
            s2 = stats.get((ds, c2))
            if s1 and s2 and s1["total"] > 0 and s2["total"] > 0:
                d = (s2["correct"]/s2["total"] - s1["correct"]/s1["total"]) * 100
                line += f" {d:>+10.1f}pp"
            else:
                line += f" {'---':>12}"
        print(line)

    # Dynamic states
    from collections import Counter
    dyn_exps = [e for e in experiments
                 if e.get("condition") == "debate_dynamic"
                 and "error" not in e
                 and e.get("dynamic_states")]
    states = Counter()
    for e in dyn_exps:
        for s in e["dynamic_states"]:
            states[s] += 1
    total = sum(states.values())
    print(f"\nDynamic states ({total} transitions):")
    for s in ["healthy", "sycophantic", "deadlock"]:
        n = states.get(s, 0)
        print(f"  {s:<14}: {n:>4} ({n/total*100:.1f}%)")

    # Run analyses
    try:
        mcnemar_test(experiments)
    except Exception as e:
        print(f"\nMcNemar test error (scipy may be missing): {e}")

    try:
        bootstrap_ci(experiments)
    except Exception as e:
        print(f"\nBootstrap error: {e}")

    behavioral_patterns(experiments)
    dynamics_distribution(experiments)
    threshold_sensitivity(experiments)
    optimal_threshold_learning(experiments)

    print()


if __name__ == "__main__":
    main()
