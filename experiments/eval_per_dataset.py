"""
Per-dataset AEIC+SG accuracy using existing θ* configs.
Two approaches:
1. Direct: evaluate full-data θ* on each dataset (fast, shows per-dataset behavior)
2. Leave-one-dataset-out CV: train on 2 datasets, evaluate on held-out 1 (honest)
"""
import json
import os
import sys
import numpy as np
from collections import Counter
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config, objective

TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]


def load_questions(ds, topo):
    suffix = f"_{topo}" if topo != "parallel" else ""
    path = f"results/signals/{ds}{suffix}.json"
    with open(path) as f:
        data = json.load(f)
    return data["questions"][:300], data["meta"]["n_agents"]


def majvote_acc(questions):
    correct = total = 0
    for q in questions:
        if "signals" not in q:
            continue
        stances = q["signals"]["stances"]
        counts = Counter(s.lower() for s in stances)
        if not counts:
            continue
        if counts.most_common(1)[0][0] == q["ground_truth"].lower():
            correct += 1
        total += 1
    return correct, total


results = {}

for topo in TOPOS:
    print(f"\n{'='*60}")
    print(f"Topology: {topo}")
    results[topo] = {}

    # Load existing θ* config
    cv_path = f"results/configs/opt_{topo}_cv.json"
    path = f"results/configs/opt_{topo}.json"
    cfg_path = cv_path if os.path.exists(cv_path) else path
    with open(cfg_path) as f:
        theta = json.load(f)
    cfg = ConsensusConfig(**{k: v for k, v in theta.items() if k in ConsensusConfig.__dataclass_fields__})

    all_qs = {}
    for ds in DATASETS:
        qs, n_agents = load_questions(ds, topo)
        valid = [q for q in qs if "signals" in q]
        all_qs[ds] = valid

        # Direct evaluation with full-data θ*
        res = evaluate_config(valid, cfg, n_agents)
        mv_c, mv_t = majvote_acc(valid)

        results[topo][ds] = {
            "aeic_acc": res["accuracy"],
            "aeic_correct": res["correct"],
            "aeic_total": res["total"],
            "majvote_acc": mv_c / max(mv_t, 1),
            "majvote_correct": mv_c,
            "majvote_total": mv_t,
        }
        print(f"  {ds}: AEIC+SG={res['accuracy']:.1%} ({res['correct']}/{res['total']}), "
              f"MajVote={mv_c/max(mv_t,1):.1%} ({mv_c}/{mv_t})")

    # Leave-one-dataset-out CV
    print(f"\n  Leave-one-dataset-out CV:")
    bounds = ConsensusConfig.bounds()
    for held_out in DATASETS:
        train_qs = []
        for ds in DATASETS:
            if ds != held_out:
                train_qs.extend(all_qs[ds])

        result = differential_evolution(
            objective, bounds,
            args=(train_qs, n_agents, 0.0),
            seed=42,
            maxiter=15,
            popsize=8,
            tol=1e-3,
            polish=False,
            init="latinhypercube",
            workers=1,
            disp=False,
        )

        lodo_cfg = ConsensusConfig.from_vector(result.x)
        lodo_res = evaluate_config(all_qs[held_out], lodo_cfg, n_agents)
        results[topo][ds + "_lodo"] = {
            "acc": lodo_res["accuracy"],
            "correct": lodo_res["correct"],
            "total": lodo_res["total"],
        }
        print(f"    Train on {[d for d in DATASETS if d != held_out]}, "
              f"test on {held_out}: {lodo_res['accuracy']:.1%}")

    # Summary
    avg_aeic = np.mean([results[topo][ds]["aeic_acc"] for ds in DATASETS])
    avg_mv = np.mean([results[topo][ds]["majvote_acc"] for ds in DATASETS])
    avg_lodo = np.mean([results[topo][ds + "_lodo"]["acc"] for ds in DATASETS])
    results[topo]["_avg_aeic"] = float(avg_aeic)
    results[topo]["_avg_mv"] = float(avg_mv)
    results[topo]["_avg_lodo"] = float(avg_lodo)
    print(f"  Averages: AEIC={avg_aeic:.1%}, MajVote={avg_mv:.1%}, LODO-CV={avg_lodo:.1%}")

# Summary table
print("\n\n" + "="*70)
print("TABLE 1 VALUES (AEIC+SG direct eval with full-data θ*)")
print("="*70)
header = f"{'Method':<25} {'MMLU-Hard':>10} {'StrategyQA':>12} {'TruthfulQA':>12} {'Average':>10}"
print(header)
print("-" * len(header))

for topo in TOPOS:
    vals = [f"{results[topo][ds]['aeic_acc']*100:.1f}" for ds in DATASETS]
    avg = results[topo]["_avg_aeic"]
    print(f"AEIC+SG ({topo:12s})  {vals[0]:>10} {vals[1]:>12} {vals[2]:>12} {avg*100:>9.1f}")

print()
print("LEAVE-ONE-DATASET-OUT CV (honest generalization)")
print("-" * len(header))
for topo in TOPOS:
    vals = [f"{results[topo][ds+'_lodo']['acc']*100:.1f}" for ds in DATASETS]
    avg = results[topo]["_avg_lodo"]
    print(f"AEIC+SG ({topo:12s})  {vals[0]:>10} {vals[1]:>12} {vals[2]:>12} {avg*100:>9.1f}")

out_path = "results/configs/per_dataset_cv_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
