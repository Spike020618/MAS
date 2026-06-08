"""
Per-dataset per-topology AEIC+SG cross-validated accuracy.
Fills the "---" cells in Table 1.

Also computes MajVote per-dataset accuracy and statistical significance tests.
"""
import json
import os
import sys
import numpy as np
from collections import Counter
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import (
    evaluate_config, compute_weights, objective, compute_ic_metric,
    consensus_from_signals
)


def majvote_accuracy(questions):
    correct = 0
    total = 0
    for q in questions:
        if "signals" not in q:
            continue
        stances = q["signals"]["stances"]
        counts = Counter(s.lower() for s in stances)
        if not counts:
            continue
        decision = counts.most_common(1)[0][0]
        if decision == q["ground_truth"].lower():
            correct += 1
        total += 1
    return correct / max(total, 1), correct, total


def cv_optimize_fast(questions, n_agents, n_folds=5, seed=42):
    labels = [q["ground_truth"] for q in questions]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    bounds = ConsensusConfig.bounds()

    fold_test_accs = []
    fold_test_correct = []
    fold_test_total = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(questions, labels)):
        train_q = [questions[i] for i in train_idx]
        test_q = [questions[i] for i in test_idx]

        result = differential_evolution(
            objective, bounds,
            args=(train_q, n_agents, 0.0),
            seed=seed + fold_i,
            maxiter=30,
            popsize=16,
            tol=1e-3,
            polish=False,
            init="latinhypercube",
            workers=1,
            disp=False,
        )

        fold_cfg = ConsensusConfig.from_vector(result.x)
        test_res = evaluate_config(test_q, fold_cfg, n_agents)
        fold_test_accs.append(test_res["accuracy"])
        fold_test_correct.append(test_res["correct"])
        fold_test_total.append(test_res["total"])

    avg = np.mean(fold_test_accs)
    std = np.std(fold_test_accs)
    total_correct = sum(fold_test_correct)
    total_total = sum(fold_test_total)
    return avg, std, total_correct, total_total, fold_test_accs


def mcnemar_test(questions, n_agents, config):
    """McNemar test: AEIC+SG vs MajVote on same questions."""
    a = 0  # both correct
    b = 0  # SG correct, MV wrong
    c = 0  # SG wrong, MV correct
    d = 0  # both wrong

    for q in questions:
        if "signals" not in q:
            continue
        gt = q["ground_truth"]

        stances = q["signals"]["stances"]
        counts = Counter(s.lower() for s in stances if s.lower() != "neutral")
        mv_decision = counts.most_common(1)[0][0] if counts else "neutral"

        sg_decision = consensus_from_signals(q["signals"], config, n_agents)

        mv_correct = mv_decision == gt
        sg_correct = sg_decision == gt

        if mv_correct and sg_correct:
            a += 1
        elif sg_correct and not mv_correct:
            b += 1
        elif not sg_correct and mv_correct:
            c += 1
        else:
            d += 1

    n = b + c
    if n == 0:
        return 1.0, b, c

    from scipy.stats import binom_test
    try:
        p = binom_test(b, n, 0.5)
    except:
        from scipy.stats import binomtest
        p = binomtest(b, n, 0.5).pvalue
    return p, b, c


TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]

results = {}

for topo in TOPOS:
    topo_suffix = f"_{topo}" if topo != "parallel" else ""
    results[topo] = {}

    for ds in DATASETS:
        sig_path = f"results/signals/{ds}{topo_suffix}.json"
        if not os.path.exists(sig_path):
            print(f"SKIP: {sig_path} not found")
            results[topo][ds] = {"skip": True}
            continue

        with open(sig_path) as f:
            data = json.load(f)
        questions = data["questions"]
        n_agents = data["meta"]["n_agents"]

        mv_acc, mv_correct, mv_total = majvote_accuracy(questions)

        print(f"\n{'='*60}")
        print(f"{topo} / {ds}: {len(questions)} questions")
        print(f"  MajVote: {mv_acc:.1%} ({mv_correct}/{mv_total})")

        cv_avg, cv_std, cv_correct, cv_total, fold_accs = cv_optimize_fast(
            questions, n_agents, n_folds=5, seed=42
        )

        print(f"  AEIC+SG CV: {cv_avg:.1%} ± {cv_std:.1%} ({cv_correct}/{cv_total})")
        print(f"  Fold accs: {[f'{a:.1%}' for a in fold_accs]}")

        results[topo][ds] = {
            "majvote_acc": mv_acc,
            "majvote_correct": mv_correct,
            "majvote_total": mv_total,
            "cv_test_acc": cv_avg,
            "cv_test_std": cv_std,
            "cv_correct": cv_correct,
            "cv_total": cv_total,
            "fold_accs": fold_accs,
        }

    topo_avg_mv = np.mean([results[topo][ds]["majvote_acc"]
                           for ds in DATASETS if "skip" not in results[topo].get(ds, {})])
    topo_avg_cv = np.mean([results[topo][ds]["cv_test_acc"]
                           for ds in DATASETS if "skip" not in results[topo].get(ds, {})])
    print(f"\n  {topo} AVERAGE: MajVote={topo_avg_mv:.1%}, AEIC+SG CV={topo_avg_cv:.1%}")

print("\n\n" + "="*70)
print("SUMMARY TABLE (for paper)")
print("="*70)
header = f"{'Method':<25} {'MMLU-Hard':>10} {'StrategyQA':>12} {'TruthfulQA':>12} {'Average':>10}"
print(header)
print("-" * len(header))

for topo in TOPOS:
    vals = []
    for ds in DATASETS:
        r = results[topo].get(ds, {})
        if "skip" in r:
            vals.append("---")
        else:
            vals.append(f"{r['majvote_acc']*100:.1f}")
    avg = np.mean([results[topo][ds]["majvote_acc"] for ds in DATASETS
                   if "skip" not in results[topo].get(ds, {})])
    print(f"MajVote ({topo:12s})  {vals[0]:>10} {vals[1]:>12} {vals[2]:>12} {avg*100:>9.1f}")

print()
for topo in TOPOS:
    vals = []
    stds = []
    for ds in DATASETS:
        r = results[topo].get(ds, {})
        if "skip" in r:
            vals.append("---")
            stds.append("---")
        else:
            vals.append(f"{r['cv_test_acc']*100:.1f}")
            stds.append(f"{r['cv_test_std']*100:.1f}")
    avg = np.mean([results[topo][ds]["cv_test_acc"] for ds in DATASETS
                   if "skip" not in results[topo].get(ds, {})])
    avg_std = np.mean([results[topo][ds]["cv_test_std"] for ds in DATASETS
                       if "skip" not in results[topo].get(ds, {})])
    print(f"AEIC+SG ({topo:12s})  {vals[0]:>10} {vals[1]:>12} {vals[2]:>12} {avg*100:>7.1f}±{avg_std*100:.1f}")

out_path = "results/configs/per_dataset_cv_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
