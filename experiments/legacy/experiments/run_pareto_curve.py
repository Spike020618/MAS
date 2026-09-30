"""
Pareto frontier: Accuracy vs IC rate for varying lambda on star topology.
Produces experiments/results/pareto_results.json for fig_pareto.
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import (
    evaluate_config, objective, compute_ic_metric
)

TOPO = "star"
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

all_questions = []
for ds in DATASETS:
    sig_path = f"experiments/results/signals/{ds}_{TOPO}.json" if TOPO != "parallel" else f"experiments/results/signals/{ds}.json"
    with open(sig_path) as f:
        data = json.load(f)
    all_questions.extend(data["questions"][:300])
    n_agents = data["meta"]["n_agents"]

print(f"Loaded {len(all_questions)} questions for {TOPO}")

bounds = ConsensusConfig.bounds()
results = []

for lam in LAMBDAS:
    print(f"\nlambda = {lam}")

    labels = [q["ground_truth"] for q in all_questions]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    fold_ics = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(all_questions, labels)):
        train_q = [all_questions[i] for i in train_idx]
        test_q = [all_questions[i] for i in test_idx]

        result = differential_evolution(
            objective, bounds,
            args=(train_q, n_agents, lam),
            seed=42 + fold_i,
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
        ic_rate = compute_ic_metric(test_q, fold_cfg, n_agents)

        fold_accs.append(test_res["accuracy"])
        fold_ics.append(ic_rate)
        print(f"  Fold {fold_i}: acc={test_res['accuracy']:.1%}, IC={ic_rate:.1%}")

    avg_acc = np.mean(fold_accs)
    avg_ic = np.mean(fold_ics)
    print(f"  Average: acc={avg_acc:.1%}, IC={avg_ic:.1%}")

    results.append({
        "lambda": lam,
        "accuracy": float(avg_acc),
        "ic_rate": float(avg_ic),
        "fold_accs": [float(a) for a in fold_accs],
        "fold_ics": [float(i) for i in fold_ics],
    })

out_path = "experiments/results/pareto_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
