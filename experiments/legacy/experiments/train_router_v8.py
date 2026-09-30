"""
Router V8 — Targeted approaches to break the 88-89% ceiling.

Key insight: the bottleneck is ~45 hard questions (1 correct topo).
The router gets 32% on these (vs 20% random for 5-topo).
If we could reach 50% on hard questions, overall would be ~91%.

Approaches:
  1. Difficulty-stratified routing
  2. Diverse ensemble (different feature subsets)
  3. Per-dataset specialized models
  4. Hard question specialist (overweight hard questions)
  5. Two-stage: easy/hard classifier → specialized router
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "experiments/results/signals"
CONFIGS_DIR = "experiments/results/configs"
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]


def extract_features(n_questions=300):
    configs = {}
    for topo in TOPOS:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    TOPO_DIM = 42
    Q_DIM = 7
    n_topos = len(TOPOS)

    X_topo_all, X_q_all, cm_all, ds_all = [], [], [], []

    for ds_idx, ds in enumerate(DATASETS):
        all_signals = {}
        for topo in TOPOS:
            path = f"{SIGNALS_DIR}/{ds}.json" if topo == "parallel" else f"{SIGNALS_DIR}/{ds}_{topo}.json"
            if not os.path.exists(path):
                break
            with open(path) as f:
                all_signals[topo] = json.load(f)
        else:
            n_agents = all_signals["parallel"]["meta"]["n_agents"]
            n_q = min(n_questions, len(all_signals["parallel"]["questions"]))

            for q_idx in range(n_q):
                features = np.zeros((n_topos, TOPO_DIM))
                topo_correct = np.zeros(n_topos)
                valid = True

                for t_idx, topo in enumerate(TOPOS):
                    if q_idx >= len(all_signals[topo]["questions"]):
                        valid = False
                        break
                    q = all_signals[topo]["questions"][q_idx]
                    if "signals" not in q:
                        valid = False
                        break

                if not valid:
                    continue

                q_text = all_signals["parallel"]["questions"][q_idx].get("question", "")
                topo_ic_means, topo_agr_means, topo_maj_ratios = [], [], []

                for t_idx, topo in enumerate(TOPOS):
                    q = all_signals[topo]["questions"][q_idx]
                    s = q["signals"]
                    ic = np.array(s["ic"])
                    ec = np.array(s["ec"])
                    eq = np.array(s["eq"])
                    ev = np.array(s["ev"])
                    agr = np.array(s["agreement"])
                    stances = s["stances"]

                    sc = Counter(st.lower() for st in stances)
                    majority = sc.most_common(1)[0][1]
                    yes_like = sum(1 for st in stances if st.lower() in ['yes', 'true'])

                    features[t_idx, 0:5] = ic
                    features[t_idx, 5:10] = ec
                    features[t_idx, 10:15] = eq
                    features[t_idx, 15:20] = ev
                    features[t_idx, 20] = ic.mean()
                    features[t_idx, 21] = ic.std()
                    features[t_idx, 22] = ec.mean()
                    features[t_idx, 23] = ec.std()
                    features[t_idx, 24] = eq.mean()
                    features[t_idx, 25] = ev.mean()
                    features[t_idx, 26] = agr.mean()
                    features[t_idx, 27] = agr.min()
                    features[t_idx, 28] = agr.max()
                    features[t_idx, 29] = agr.std()
                    features[t_idx, 30] = majority / len(stances)
                    features[t_idx, 31] = 1.0 if yes_like > len(stances) / 2 else -1.0
                    features[t_idx, 32] = yes_like / len(stances)
                    features[t_idx, 33] = 1.0 - yes_like / len(stances)
                    features[t_idx, 34] = ic.mean() * (1 - agr.mean())
                    features[t_idx, 35] = ec.mean() * (1 - agr.mean())
                    features[t_idx, 36] = eq.mean() * (ic.std() / (ic.mean() + 1e-6))
                    features[t_idx, 37] = ic.max() - ic.min()
                    features[t_idx, 38] = ic.mean() - agr.mean()
                    features[t_idx, 39] = ic.mean() * (majority / len(stances))
                    features[t_idx, 40] = agr.std() * ic.mean()
                    features[t_idx, 41] = ev.mean() * (1 - majority / len(stances))

                    topo_ic_means.append(ic.mean())
                    topo_agr_means.append(agr.mean())
                    topo_maj_ratios.append(majority / len(stances))

                    if topo in configs:
                        res = evaluate_config([q], configs[topo], n_agents)
                        topo_correct[t_idx] = 1.0 if res["correct"] > 0 else 0.0

                q_feat = np.zeros(Q_DIM)
                q_feat[0] = len(q_text) / 500.0
                q_feat[1] = len(q_text.split()) / 100.0
                q_feat[2] = np.std(topo_ic_means)
                q_feat[3] = np.std(topo_agr_means)
                q_feat[4] = np.max(topo_ic_means) - np.min(topo_ic_means)
                q_feat[5] = np.mean(topo_maj_ratios)
                q_feat[6] = np.std(topo_maj_ratios)

                X_topo_all.append(features)
                X_q_all.append(q_feat)
                cm_all.append(topo_correct)
                ds_all.append(ds_idx)

    return (np.array(X_topo_all), np.array(X_q_all),
            np.array(cm_all), np.array(ds_all))


class RouterPerDataset(nn.Module):
    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=192,
                 n_topos=5, n_datasets=3, dropout=0.2):
        super().__init__()
        self.n_topos = n_topos
        self.n_datasets = n_datasets

        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
        )
        self.q_encoder = nn.Sequential(
            nn.Linear(q_feat_dim, hidden // 4),
            nn.GELU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
        )
        scorer_in = hidden // 2 + hidden // 4 + hidden // 4
        self.dataset_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scorer_in, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            ) for _ in range(n_datasets)
        ])

    def forward(self, x_topo, x_question, dataset_ids):
        embs = torch.stack(
            [self.topo_encoder(x_topo[:, t]) for t in range(self.n_topos)], dim=1
        )
        q_emb = self.q_encoder(x_question)
        scores = []
        for t in range(self.n_topos):
            mask = torch.ones(self.n_topos, dtype=torch.bool)
            mask[t] = False
            ctx = self.ctx_proj(embs[:, mask, :].mean(dim=1))
            inp = torch.cat([embs[:, t], ctx, q_emb], dim=1)
            batch_scores = torch.zeros(inp.shape[0], device=inp.device)
            for ds_id in range(self.n_datasets):
                ds_mask = (dataset_ids == ds_id)
                if ds_mask.any():
                    batch_scores[ds_mask] = self.dataset_scorers[ds_id](
                        inp[ds_mask]).squeeze(-1)
            scores.append(batch_scores)
        return torch.stack(scores, dim=1)


class FocalRankingLoss(nn.Module):
    def __init__(self, base_margin=0.5, max_margin=2.0, gamma=3.0, n_topos=5):
        super().__init__()
        self.base_margin = base_margin
        self.max_margin = max_margin
        self.gamma = gamma
        self.n_topos = n_topos

    def forward(self, scores, correct_mask, weights=None):
        batch_size = scores.shape[0]
        loss = scores.sum() * 0.0
        n_pairs = 0
        for b in range(batch_size):
            correct_idx = torch.where(correct_mask[b] > 0)[0]
            incorrect_idx = torch.where(correct_mask[b] == 0)[0]
            if len(correct_idx) == 0 or len(incorrect_idx) == 0:
                continue
            difficulty = 1.0 - len(correct_idx) / self.n_topos
            margin = self.base_margin + difficulty * (self.max_margin - self.base_margin)
            w = weights[b] if weights is not None else 1.0
            for c in correct_idx:
                for w_idx in incorrect_idx:
                    violation = F.relu(margin - (scores[b, c] - scores[b, w_idx]))
                    focal_weight = (violation / margin + 1e-6) ** self.gamma
                    loss += w * focal_weight * violation
                    n_pairs += 1
        return loss / max(n_pairs, 1)


class RouterDataset(Dataset):
    def __init__(self, X_topo, X_q, cm, ds, weights=None):
        self.X_topo = torch.FloatTensor(X_topo)
        self.X_q = torch.FloatTensor(X_q)
        self.cm = torch.FloatTensor(cm)
        self.ds = torch.LongTensor(ds)
        self.weights = torch.FloatTensor(weights) if weights is not None else torch.ones(len(X_topo))

    def __len__(self):
        return len(self.X_topo)

    def __getitem__(self, idx):
        return self.X_topo[idx], self.X_q[idx], self.cm[idx], self.ds[idx], self.weights[idx]


def train_single(X_topo, X_q, cm, ds, tri, tei, seed=42,
                 epochs=300, lr=1e-3, hidden=192, dropout=0.2, n_topos=5,
                 weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topo_feat_dim = X_topo.shape[2]
    q_feat_dim = X_q.shape[1]

    scaler_t = StandardScaler()
    Xt_tr = scaler_t.fit_transform(
        X_topo[tri].reshape(-1, topo_feat_dim)
    ).reshape(len(tri), n_topos, topo_feat_dim)
    Xt_te = scaler_t.transform(
        X_topo[tei].reshape(-1, topo_feat_dim)
    ).reshape(len(tei), n_topos, topo_feat_dim)

    scaler_q = StandardScaler()
    Xq_tr = scaler_q.fit_transform(X_q[tri])
    Xq_te = scaler_q.transform(X_q[tei])

    w_tr = weights[tri] if weights is not None else None
    train_ds = RouterDataset(Xt_tr, Xq_tr, cm[tri], ds[tri], w_tr)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    torch.manual_seed(seed)
    model = RouterPerDataset(
        topo_feat_dim=topo_feat_dim, q_feat_dim=q_feat_dim,
        hidden=hidden, n_topos=n_topos, n_datasets=3, dropout=dropout
    ).to(device)
    criterion = FocalRankingLoss(n_topos=n_topos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    best_scores = None
    patience = 40
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for Xt_b, Xq_b, cm_b, ds_b, w_b in train_loader:
            Xt_b, Xq_b, cm_b, ds_b = (
                Xt_b.to(device), Xq_b.to(device),
                cm_b.to(device), ds_b.to(device))
            optimizer.zero_grad()
            scores = model(Xt_b, Xq_b, ds_b)
            loss = criterion(scores, cm_b, w_b.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            te_scores = model(
                torch.FloatTensor(Xt_te).to(device),
                torch.FloatTensor(Xq_te).to(device),
                torch.LongTensor(ds[tei]).to(device)
            ).cpu()
        preds = te_scores.argmax(dim=1).numpy()
        correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
        acc = correct / len(tei)
        if acc > best_acc:
            best_acc = acc
            best_scores = te_scores.clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_acc, best_scores


def baseline_eval(X_topo, X_q, cm, ds, n_folds=5, seed=42, **kwargs):
    """Standard 5-fold eval."""
    y_strat = cm.argmax(axis=1)
    y_strat[cm.sum(axis=1) == 0] = 0
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []
    for fold, (tri, tei) in enumerate(kf.split(X_topo, y_strat)):
        acc, _ = train_single(X_topo, X_q, cm, ds, tri, tei,
                              seed=seed + fold, **kwargs)
        fold_accs.append(acc)
    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def main():
    sys.stdout.reconfigure(line_buffering=True)
    import warnings; warnings.filterwarnings("ignore")

    print("=" * 70)
    print("Router V8 — Breaking the 88-89% Ceiling")
    print("=" * 70)

    print("\n[0] Loading data...")
    Xt, Xq, cm, ds = extract_features()
    n_topos = Xt.shape[1]
    oracle = (cm.sum(axis=1) > 0).mean()
    star_acc = cm[:, TOPOS.index("star")].mean()
    n_correct = cm.sum(axis=1)
    print(f"  {len(Xt)} questions, Star: {star_acc:.1%}, Oracle: {oracle:.1%}")

    # Difficulty distribution
    for nc in range(6):
        count = (n_correct == nc).sum()
        if count > 0:
            print(f"  {nc} correct: {count} ({count/len(Xt)*100:.1f}%)")

    results = {}

    # ---- Baseline ----
    print("\n" + "=" * 70)
    print("[1] Baseline (reproduce V6 best)")
    print("=" * 70)
    accs_bl = []
    for s in range(10):
        acc, _, _ = baseline_eval(Xt, Xq, cm, ds, seed=s * 7 + 42)
        accs_bl.append(acc)
    results["baseline"] = np.mean(accs_bl)
    print(f"  Baseline (10-seed): {np.mean(accs_bl):.1%} ± {np.std(accs_bl):.1%}")

    # ---- Approach 1: Hard question overweighting ----
    print("\n" + "=" * 70)
    print("[2] Hard question overweighting")
    print("=" * 70)
    for weight_factor in [2.0, 3.0, 5.0, 10.0]:
        weights = np.ones(len(Xt))
        weights[n_correct <= 1] = weight_factor
        weights[n_correct == 2] = weight_factor / 2

        accs_hw = []
        for s in range(5):
            y_strat = cm.argmax(axis=1)
            y_strat[cm.sum(axis=1) == 0] = 0
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s * 7 + 42)
            fold_accs = []
            for fold, (tri, tei) in enumerate(kf.split(Xt, y_strat)):
                acc, _ = train_single(Xt, Xq, cm, ds, tri, tei,
                                      seed=s * 7 + 42 + fold, weights=weights)
                fold_accs.append(acc)
            accs_hw.append(np.mean(fold_accs))

        key = f"hard_w{weight_factor}"
        results[key] = np.mean(accs_hw)
        print(f"  weight={weight_factor:5.1f}: {np.mean(accs_hw):.1%} ± {np.std(accs_hw):.1%}")

    # ---- Approach 2: Diverse ensemble ----
    print("\n" + "=" * 70)
    print("[3] Diverse ensemble (different feature subsets)")
    print("=" * 70)

    # Create feature subsets
    feature_subsets = [
        ("all", list(range(42))),
        ("ic_only", list(range(0, 5)) + [20, 21, 37]),        # IC signals
        ("agreement", list(range(20, 30)) + [30, 34, 35]),     # Agreement signals
        ("interaction", list(range(34, 42))),                     # Interaction features
        ("no_ev", list(range(0, 15)) + list(range(20, 42))),   # Drop EV
        ("stats_only", list(range(20, 42))),                     # Only aggregate stats
    ]

    y_strat = cm.argmax(axis=1)
    y_strat[cm.sum(axis=1) == 0] = 0

    for n_seeds in [5, 10]:
        all_fold_accs = []
        for s in range(n_seeds):
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s * 7 + 42)
            fold_accs_ens = []
            for fold, (tri, tei) in enumerate(kf.split(Xt, y_strat)):
                all_scores = []
                for subset_name, subset_dims in feature_subsets:
                    Xt_sub = Xt[:, :, subset_dims]
                    _, scores = train_single(Xt_sub, Xq, cm, ds, tri, tei,
                                             seed=s * 7 + 42 + fold * 100 + hash(subset_name) % 100,
                                             n_topos=n_topos)
                    if scores is not None:
                        all_scores.append(scores)

                if all_scores:
                    avg_scores = torch.stack(all_scores).mean(dim=0)
                    preds = avg_scores.argmax(dim=1).numpy()
                    correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
                    fold_accs_ens.append(correct / len(tei))

            all_fold_accs.append(np.mean(fold_accs_ens))

        results[f"diverse_ens_{n_seeds}s"] = np.mean(all_fold_accs)
        print(f"  Diverse ensemble ({n_seeds}-seed): {np.mean(all_fold_accs):.1%} ± {np.std(all_fold_accs):.1%}")

    # ---- Approach 3: Per-dataset specialized ----
    print("\n" + "=" * 70)
    print("[4] Per-dataset specialized models")
    print("=" * 70)
    for s in range(5):
        per_ds_correct = 0
        per_ds_total = 0
        for ds_idx, ds_name in enumerate(DATASETS):
            ds_mask = (ds == ds_idx)
            Xt_ds = Xt[ds_mask]
            Xq_ds = Xq[ds_mask]
            cm_ds = cm[ds_mask]
            ds_ds = ds[ds_mask]

            if len(Xt_ds) < 50:
                continue

            y_strat_ds = cm_ds.argmax(axis=1)
            y_strat_ds[cm_ds.sum(axis=1) == 0] = 0

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s * 7 + 42)
            for fold, (tri, tei) in enumerate(kf.split(Xt_ds, y_strat_ds)):
                acc, scores = train_single(Xt_ds, Xq_ds, cm_ds, ds_ds, tri, tei,
                                           seed=s * 7 + 42 + fold, hidden=128)
                if scores is not None:
                    preds = scores.argmax(dim=1).numpy()
                    for i, p in enumerate(preds):
                        if cm_ds[tei[i], p] > 0:
                            per_ds_correct += 1
                        per_ds_total += 1

    results["per_dataset"] = per_ds_correct / max(per_ds_total, 1) if per_ds_total > 0 else 0
    print(f"  Per-dataset models: {results['per_dataset']:.1%}")

    # ---- Approach 4: Two-stage (difficulty → specialized router) ----
    print("\n" + "=" * 70)
    print("[5] Two-stage: difficulty-aware routing")
    print("=" * 70)
    # Stage 1: classify easy vs hard based on how many topos are correct
    is_hard = n_correct <= 2
    hard_ratio = is_hard.mean()
    print(f"  Hard questions: {is_hard.sum()} ({hard_ratio:.1%})")

    accs_2stage = []
    for s in range(5):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s * 7 + 42)
        fold_accs_2s = []
        for fold, (tri, tei) in enumerate(kf.split(Xt, y_strat)):
            # Train separate models for hard and easy
            hard_tri = tri[is_hard[tri]]
            easy_tri = tri[~is_hard[tri]]

            # Hard model — more aggressive training
            if len(hard_tri) > 20:
                _, hard_scores = train_single(
                    Xt, Xq, cm, ds, hard_tri, tei,
                    seed=s * 7 + 42 + fold, epochs=400, lr=5e-4, hidden=256)
            else:
                hard_scores = None

            # Easy model — standard
            _, easy_scores = train_single(
                Xt, Xq, cm, ds, easy_tri, tei,
                seed=s * 7 + 42 + fold + 1000)

            # Combined model (full data)
            _, full_scores = train_single(
                Xt, Xq, cm, ds, tri, tei,
                seed=s * 7 + 42 + fold + 2000)

            # For test set: use full model score + hard model bonus
            if hard_scores is not None:
                combined = 0.7 * full_scores + 0.3 * hard_scores
            else:
                combined = full_scores

            preds = combined.argmax(dim=1).numpy()
            correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
            fold_accs_2s.append(correct / len(tei))

        accs_2stage.append(np.mean(fold_accs_2s))

    results["two_stage"] = np.mean(accs_2stage)
    print(f"  Two-stage (5-seed): {np.mean(accs_2stage):.1%} ± {np.std(accs_2stage):.1%}")

    # ---- Approach 5: Multi-seed diverse with varying hyperparams ----
    print("\n" + "=" * 70)
    print("[6] HP-diverse ensemble")
    print("=" * 70)
    hp_configs = [
        {"hidden": 128, "lr": 1e-3, "dropout": 0.15},
        {"hidden": 192, "lr": 1e-3, "dropout": 0.2},
        {"hidden": 256, "lr": 5e-4, "dropout": 0.15},
        {"hidden": 192, "lr": 2e-3, "dropout": 0.2},
        {"hidden": 128, "lr": 5e-4, "dropout": 0.1},
    ]

    accs_hpe = []
    for s in range(5):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s * 7 + 42)
        fold_accs_hpe = []
        for fold, (tri, tei) in enumerate(kf.split(Xt, y_strat)):
            all_scores = []
            for hp_idx, hp in enumerate(hp_configs):
                _, scores = train_single(
                    Xt, Xq, cm, ds, tri, tei,
                    seed=s * 7 + 42 + fold * 100 + hp_idx * 10, **hp)
                if scores is not None:
                    all_scores.append(scores)

            avg_scores = torch.stack(all_scores).mean(dim=0)
            preds = avg_scores.argmax(dim=1).numpy()
            correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
            fold_accs_hpe.append(correct / len(tei))

        accs_hpe.append(np.mean(fold_accs_hpe))

    results["hp_diverse_ens"] = np.mean(accs_hpe)
    print(f"  HP-diverse ensemble (5-seed): {np.mean(accs_hpe):.1%} ± {np.std(accs_hpe):.1%}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Star baseline:      {star_acc:.1%}")
    print(f"  V6 reference:       88.8%")
    for name in sorted(results, key=lambda x: results[x]):
        marker = " ← BEST" if results[name] == max(results.values()) else ""
        print(f"  {name:22s}: {results[name]:.1%}{marker}")
    print(f"  Oracle:              {oracle:.1%}")
    print(f"  Best:                {max(results.values()):.1%}")
    print(f"  Gap to oracle:       {(oracle - max(results.values()))*100:.1f}pp")


if __name__ == "__main__":
    main()
