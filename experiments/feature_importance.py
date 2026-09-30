"""
Feature importance analysis for the neural topology router.

Uses permutation importance: shuffle each feature group and measure accuracy drop.
Also computes per-feature-group ablation.
"""
import json
import os
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.train_router_v10 import (
    extract_features, RouterModel, RouterDataset, FocalRankingLoss,
    TOPOS, DATASETS
)

FEATURE_GROUPS = {
    "IC (per-agent)": list(range(0, 5)),
    "EC (per-agent)": list(range(5, 10)),
    "EQ (per-agent)": list(range(10, 15)),
    "EV (per-agent)": list(range(15, 20)),
    "IC stats": [20, 21],
    "EC stats": [22, 23],
    "EQ/EV mean": [24, 25],
    "Agreement stats": [26, 27, 28, 29],
    "Voting features": [30, 31, 32, 33],
    "Interaction terms": [34, 35, 36, 37, 38, 39, 40, 41],
}

QUESTION_FEATURE_NAMES = [
    "Q: text length", "Q: word count", "Q: cross-topo IC std",
    "Q: cross-topo agr std", "Q: IC range", "Q: mean maj ratio",
    "Q: maj ratio std",
]


def train_and_evaluate(X_topo_train, X_q_train, cm_train, ds_train,
                       X_topo_test, X_q_test, cm_test, ds_test,
                       hidden=192, lr=0.001, dropout=0.2, epochs=60):
    """Train router model and return test accuracy."""
    scaler_topo = StandardScaler()
    n_train = X_topo_train.shape[0]
    n_test = X_topo_test.shape[0]
    n_topos = X_topo_train.shape[1]
    feat_dim = X_topo_train.shape[2]

    flat_train = X_topo_train.reshape(-1, feat_dim)
    scaler_topo.fit(flat_train)
    X_topo_train_s = scaler_topo.transform(flat_train).reshape(n_train, n_topos, feat_dim)
    X_topo_test_s = scaler_topo.transform(X_topo_test.reshape(-1, feat_dim)).reshape(n_test, n_topos, feat_dim)

    scaler_q = StandardScaler()
    X_q_train_s = scaler_q.fit_transform(X_q_train)
    X_q_test_s = scaler_q.transform(X_q_test)

    train_ds = RouterDataset(X_topo_train_s, X_q_train_s, cm_train, ds_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    model = RouterModel(topo_feat_dim=feat_dim, q_feat_dim=X_q_train.shape[1],
                        hidden=hidden, dropout=dropout)
    criterion = FocalRankingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            xt, xq, cm_b, ds_b = batch
            scores = model(xt, xq, ds_b)
            loss = criterion(scores, cm_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        test_ds = RouterDataset(X_topo_test_s, X_q_test_s, cm_test, ds_test)
        xt, xq, cm_b, ds_b = test_ds[:]
        scores = model(xt, xq, ds_b)
        preds = scores.argmax(dim=1).numpy()
        correct = sum(cm_b[i, preds[i]] > 0 for i in range(len(preds)))
        acc = correct / len(preds)

    return float(acc), model


def permutation_importance(X_topo, X_q, cm, ds, feature_indices, n_permutations=10,
                           n_folds=5, seed=42):
    """Compute permutation importance for a set of topology features."""
    labels = (cm.sum(axis=1) > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    base_accs = []
    perm_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_topo, labels)):
        X_topo_train, X_topo_test = X_topo[train_idx], X_topo[test_idx].copy()
        X_q_train, X_q_test = X_q[train_idx], X_q[test_idx]
        cm_train, cm_test = cm[train_idx], cm[test_idx]
        ds_train, ds_test = ds[train_idx], ds[test_idx]

        base_acc, _ = train_and_evaluate(
            X_topo_train, X_q_train, cm_train, ds_train,
            X_topo_test, X_q_test, cm_test, ds_test,
            epochs=40
        )
        base_accs.append(base_acc)

        fold_perm_accs = []
        for perm_i in range(n_permutations):
            rng = np.random.RandomState(seed + fold_i * 100 + perm_i)
            X_topo_perm = X_topo_test.copy()
            for fi in feature_indices:
                perm_order = rng.permutation(len(X_topo_perm))
                X_topo_perm[:, :, fi] = X_topo_perm[perm_order, :, fi]

            perm_acc, _ = train_and_evaluate(
                X_topo_train, X_q_train, cm_train, ds_train,
                X_topo_perm, X_q_test, cm_test, ds_test,
                epochs=40
            )
            fold_perm_accs.append(perm_acc)

        perm_accs.append(np.mean(fold_perm_accs))

    importance = np.mean(base_accs) - np.mean(perm_accs)
    return importance, np.mean(base_accs), np.mean(perm_accs)


def ablation_importance(X_topo, X_q, cm, ds, n_folds=5, seed=42):
    """Leave-one-group-out ablation: zero out each feature group and measure drop."""
    labels = (cm.sum(axis=1) > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    print("\nComputing baseline accuracy...")
    base_accs = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_topo, labels)):
        acc, _ = train_and_evaluate(
            X_topo[train_idx], X_q[train_idx], cm[train_idx], ds[train_idx],
            X_topo[test_idx], X_q[test_idx], cm[test_idx], ds[test_idx],
            epochs=50
        )
        base_accs.append(acc)
    baseline = np.mean(base_accs)
    baseline_std = np.std(base_accs)
    print(f"Baseline: {baseline:.1%} ± {baseline_std:.1%}")

    results = {}
    for group_name, indices in FEATURE_GROUPS.items():
        print(f"\nAblating: {group_name} (features {indices})...")
        X_topo_abl = X_topo.copy()
        X_topo_abl[:, :, indices] = 0.0

        abl_accs = []
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_topo_abl, labels)):
            acc, _ = train_and_evaluate(
                X_topo_abl[train_idx], X_q[train_idx], cm[train_idx], ds[train_idx],
                X_topo_abl[test_idx], X_q[test_idx], cm[test_idx], ds[test_idx],
                epochs=50
            )
            abl_accs.append(acc)

        abl_mean = np.mean(abl_accs)
        drop = baseline - abl_mean
        results[group_name] = {
            "baseline": baseline,
            "ablated": abl_mean,
            "drop": drop,
            "indices": indices,
        }
        print(f"  {group_name}: {abl_mean:.1%} (drop: {drop:+.1%})")

    # Question features ablation
    print(f"\nAblating: Question features...")
    X_q_zero = np.zeros_like(X_q)
    q_accs = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_topo, labels)):
        acc, _ = train_and_evaluate(
            X_topo[train_idx], X_q_zero[train_idx], cm[train_idx], ds[train_idx],
            X_topo[test_idx], X_q_zero[test_idx], cm[test_idx], ds[test_idx],
            epochs=50
        )
        q_accs.append(acc)
    q_mean = np.mean(q_accs)
    q_drop = baseline - q_mean
    results["Question features"] = {
        "baseline": baseline,
        "ablated": q_mean,
        "drop": q_drop,
    }
    print(f"  Question features: {q_mean:.1%} (drop: {q_drop:+.1%})")

    return results, baseline


def main():
    print("="*60)
    print("ROUTER FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    print("\nExtracting features...")
    X_topo, X_q, cm, ds = extract_features()
    print(f"Dataset: {X_topo.shape[0]} questions, {X_topo.shape[2]} topo features, {X_q.shape[1]} q features")

    results, baseline = ablation_importance(X_topo, X_q, cm, ds, n_folds=5, seed=42)

    print("\n" + "="*60)
    print("FEATURE GROUP IMPORTANCE (sorted by drop)")
    print("="*60)
    print(f"{'Group':<25} {'Ablated':>10} {'Drop':>10}")
    print("-" * 50)
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["drop"])
    for name, r in sorted_results:
        print(f"{name:<25} {r['ablated']*100:>9.1f}% {r['drop']*100:>+9.1f}pp")

    out = {
        "baseline": baseline,
        "groups": {k: {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                      for kk, vv in v.items()}
                  for k, v in results.items()},
    }
    out_path = "experiments/results/feature_importance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
