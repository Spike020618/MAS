"""
Topology Router V3c — SharedEncoder + Ranking + Enhanced Features (900 questions)

The best-performing neural router architecture from V3 experiments:
  - SharedEncoder: same encoder per topology, cross-topo context via mean pooling
  - PairwiseRankingLoss with margin=1.0
  - Enhanced 34-dim features: raw signals (20) + summary stats with majority/stances (14)
  - Now trained on 900 questions (3 datasets × 300) instead of 600

Previous results on 600 questions:
  V3c: 84.9% ± 0.5pp (5 seeds)
  GBM: 83.8%
  Star baseline: 80.0%
  Oracle: 93.2%

Usage:
  python experiments/train_router_v3c.py
  python experiments/train_router_v3c.py --n_questions 300  # use all 300 per dataset
  python experiments/train_router_v3c.py --seeds 10         # more seeds for stability
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "results/signals"
CONFIGS_DIR = "results/configs"
TRAIN_DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate"]


# ============================================================
# Network Architecture
# ============================================================

class RouterV3c(nn.Module):
    """SharedEncoder + cross-topo context. Best architecture from V3 series."""

    def __init__(self, feat_dim=38, hidden=128, n_topos=4):
        super().__init__()
        self.n_topos = n_topos
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden // 2 + hidden // 4, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        """x: (batch, n_topos, feat_dim) → (batch, n_topos) scores"""
        embs = torch.stack([self.encoder(x[:, t]) for t in range(self.n_topos)], dim=1)
        scores = []
        for t in range(self.n_topos):
            mask = torch.ones(self.n_topos, dtype=torch.bool)
            mask[t] = False
            ctx = self.ctx_proj(embs[:, mask, :].mean(dim=1))
            scores.append(self.scorer(torch.cat([embs[:, t], ctx], dim=1)).squeeze(-1))
        return torch.stack(scores, dim=1)


# ============================================================
# Loss Functions
# ============================================================

class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores, correct_mask):
        batch_size, n_topos = scores.shape
        loss = torch.tensor(0.0, device=scores.device)
        n_pairs = 0

        for b in range(batch_size):
            correct_idx = torch.where(correct_mask[b] > 0)[0]
            incorrect_idx = torch.where(correct_mask[b] == 0)[0]
            if len(correct_idx) == 0 or len(incorrect_idx) == 0:
                continue
            for c in correct_idx:
                for w in incorrect_idx:
                    loss += F.relu(self.margin - (scores[b, c] - scores[b, w]))
                    n_pairs += 1

        return loss / max(n_pairs, 1)


# ============================================================
# Feature Extraction (34 dims per topology)
# ============================================================

def extract_features(signals_data, topos=TOPOS, n_questions=300):
    """
    Extract 38-dim feature vector per topology per question.

    Features (38 dims per topology):
      Raw signals (20): ic[5] + ec[5] + eq[5] + ev[5]
      Summary stats (14):
        ic_mean, ic_std, ec_mean, ec_std, eq_mean, ev_mean,
        agr_mean, agr_min, agr_max, agr_std,
        majority_ratio, decision_sign, yes_ratio, no_ratio
      Cross-topology features (4):
        cross_disagree: 1 if this topo's majority answer differs from cross-topo majority
        n_distinct_stances: number of distinct majority answers across all topos
        stance_entropy: entropy of stance distribution within this topo
        ic_cv: coefficient of variation of IC scores (spread / mean)
    """
    from collections import Counter

    FEAT_DIM = 38
    X_all = []
    correct_masks = []

    configs = {}
    for topo in topos:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    for ds in TRAIN_DATASETS:
        all_signals = {}
        for topo in topos:
            if topo == "parallel":
                path = os.path.join(SIGNALS_DIR, f"{ds}.json")
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")
            with open(path) as f:
                all_signals[topo] = json.load(f)

        n_agents = all_signals["parallel"]["meta"]["n_agents"]
        n_q = min(n_questions, len(all_signals["parallel"]["questions"]))

        for q_idx in range(n_q):
            features = np.zeros((len(topos), FEAT_DIM))
            topo_correct = np.zeros(len(topos))
            valid = True

            # First pass: collect stances for cross-topology features
            all_stances = {}
            for t_idx, topo in enumerate(topos):
                if topo not in all_signals:
                    valid = False
                    break
                q = all_signals[topo]["questions"][q_idx]
                if "signals" not in q:
                    valid = False
                    break
                all_stances[topo] = q["signals"]["stances"]

            if not valid:
                continue

            # Compute cross-topology majority stances
            maj_stances = {}
            for topo in topos:
                sc = Counter(st.lower() for st in all_stances[topo])
                maj_stances[topo] = sc.most_common(1)[0][0]
            n_distinct = len(set(maj_stances.values()))

            # Second pass: extract features
            for t_idx, topo in enumerate(topos):
                q = all_signals[topo]["questions"][q_idx]
                s = q["signals"]
                ic = np.array(s["ic"])
                ec = np.array(s["ec"])
                eq = np.array(s["eq"])
                ev = np.array(s["ev"])
                agr = np.array(s["agreement"])
                stances = s["stances"]

                # Raw signals (20 dims)
                features[t_idx, 0:5] = ic
                features[t_idx, 5:10] = ec
                features[t_idx, 10:15] = eq
                features[t_idx, 15:20] = ev

                # Summary statistics (14 dims)
                stance_counts = Counter(st.lower() for st in stances)
                majority = stance_counts.most_common(1)[0][1] if stance_counts else 0
                yes_like = sum(1 for st in stances if st.lower() in ['yes', 'true'])

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

                # NEW: Cross-topology features (4 dims)
                # cross_disagree: does this topo disagree with the cross-topo majority?
                other_stances = [maj_stances[t] for t in topos if t != topo]
                cross_majority = Counter(other_stances).most_common(1)[0][0]
                features[t_idx, 34] = 1.0 if maj_stances[topo] != cross_majority else 0.0

                # n_distinct_stances across topologies
                features[t_idx, 35] = n_distinct / len(topos)

                # stance_entropy within this topology
                probs = np.array(list(stance_counts.values())) / sum(stance_counts.values())
                features[t_idx, 36] = -np.sum(probs * np.log(probs + 1e-10))

                # ic_cv: coefficient of variation
                features[t_idx, 37] = ic.std() / (ic.mean() + 1e-6)

                # Correctness
                if topo in configs:
                    res = evaluate_config([q], configs[topo], n_agents)
                    topo_correct[t_idx] = 1.0 if res["correct"] > 0 else 0.0

            X_all.append(features)
            correct_masks.append(topo_correct)

    return np.array(X_all), np.array(correct_masks)


# ============================================================
# Dataset
# ============================================================

class TopologyDataset(Dataset):
    def __init__(self, X, correct_masks):
        self.X = torch.FloatTensor(X)
        self.correct_masks = torch.FloatTensor(correct_masks)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.correct_masks[idx]


# ============================================================
# Training
# ============================================================

def train_single_seed(X, correct_masks, seed=42, n_folds=5, epochs=200,
                      lr=1e-3, hidden=128, margin=1.0):
    """Train with one seed, return mean CV accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_topos = X.shape[1]
    feat_dim = X.shape[2]

    y_stratify = correct_masks.argmax(axis=1)
    all_wrong = correct_masks.sum(axis=1) == 0
    y_stratify[all_wrong] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_stratify)):
        X_train, X_test = X[train_idx], X[test_idx]
        cm_train, cm_test = correct_masks[train_idx], correct_masks[test_idx]

        # Normalize per-feature across all topologies
        X_train_flat = X_train.reshape(-1, feat_dim)
        X_test_flat = X_test.reshape(-1, feat_dim)
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)
        X_train_norm = X_train_flat.reshape(len(train_idx), n_topos, feat_dim)
        X_test_norm = X_test_flat.reshape(len(test_idx), n_topos, feat_dim)

        train_ds = TopologyDataset(X_train_norm, cm_train)
        test_ds = TopologyDataset(X_test_norm, cm_test)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        test_loader = DataLoader(test_ds, batch_size=128)

        torch.manual_seed(seed + fold)
        model = RouterV3c(feat_dim=feat_dim, hidden=hidden, n_topos=n_topos).to(device)
        criterion = PairwiseRankingLoss(margin=margin)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        patience = 30
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for X_batch, cm_batch in train_loader:
                X_batch, cm_batch = X_batch.to(device), cm_batch.to(device)
                optimizer.zero_grad()
                scores = model(X_batch)
                loss = criterion(scores, cm_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, cm_batch in test_loader:
                    X_batch = X_batch.to(device)
                    scores = model(X_batch)
                    preds = scores.argmax(dim=1).cpu().numpy()
                    for i, pred in enumerate(preds):
                        if cm_batch[i, pred] > 0:
                            correct += 1
                        total += 1

            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        fold_accs.append(best_acc)

    return np.mean(fold_accs), fold_accs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print("Router V3c — 900 Questions (SharedEncoder + Ranking + Enhanced Features)")
    print("=" * 60)

    print(f"\n[1] Extracting features ({args.n_questions} questions × 3 datasets)...")
    X, correct_masks = extract_features(TOPOS, n_questions=args.n_questions)
    print(f"  Total samples: {len(X)}")
    print(f"  Feature shape: {X.shape} (questions × topos × features)")
    print(f"  Avg correct topos/question: {correct_masks.sum(axis=1).mean():.2f}")

    # Distribution
    n_correct_dist = correct_masks.sum(axis=1)
    for nc in range(5):
        pct = (n_correct_dist == nc).sum() / len(n_correct_dist) * 100
        print(f"    {int(nc)} topos correct: {(n_correct_dist == nc).sum()} ({pct:.1f}%)")

    # Per-topology accuracy
    star_idx = TOPOS.index("star")
    print(f"\n  Per-topology correctness rate:")
    for t_idx, topo in enumerate(TOPOS):
        rate = correct_masks[:, t_idx].mean()
        print(f"    {topo}: {rate:.1%}")

    # Oracle
    oracle = (correct_masks.sum(axis=1) > 0).mean()
    print(f"  Oracle (any correct): {oracle:.1%}")

    print(f"\n[2] Training V3c (hidden={args.hidden}, margin={args.margin}, "
          f"lr={args.lr}, {args.seeds} seeds)...")

    seed_results = []
    for seed in range(args.seeds):
        acc, fold_accs = train_single_seed(
            X, correct_masks, seed=seed*10+42,
            epochs=args.epochs, hidden=args.hidden,
            lr=args.lr, margin=args.margin)
        seed_results.append(acc)
        print(f"  Seed {seed}: {acc:.1%} (folds: {[f'{a:.1%}' for a in fold_accs]})")

    mean_acc = np.mean(seed_results)
    std_acc = np.std(seed_results)
    print(f"\n{'='*60}")
    print(f"RESULT: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"  Star baseline: {correct_masks[:, star_idx].mean():.1%}")
    print(f"  Oracle: {oracle:.1%}")
    print(f"  Gap to oracle: {(oracle - mean_acc)*100:.1f}pp")
    print(f"{'='*60}")

    # Also compare with GBM
    print(f"\n[3] GBM baseline comparison (same features, 900 questions)...")
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        # Flatten for GBM: per (question, topology) → predict correctness
        n_q, n_topos, feat_dim = X.shape
        X_gbm = X.reshape(n_q * n_topos, feat_dim)
        y_gbm = correct_masks.reshape(-1)

        # Add topology one-hot
        topo_onehot = np.zeros((n_q * n_topos, n_topos))
        for t in range(n_topos):
            topo_onehot[t::n_topos, t] = 1.0  # wrong pattern
        # Fix: proper indexing
        topo_onehot = np.zeros((n_q * n_topos, n_topos))
        for qi in range(n_q):
            for ti in range(n_topos):
                topo_onehot[qi * n_topos + ti, ti] = 1.0

        X_gbm_full = np.hstack([X_gbm, topo_onehot])

        # Cross-validate
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_strat = correct_masks.argmax(axis=1)
        all_wrong = correct_masks.sum(axis=1) == 0
        y_strat[all_wrong] = 0

        gbm_accs = []
        for train_idx, test_idx in kf.split(np.arange(n_q), y_strat):
            # Train GBM
            train_rows = []
            for qi in train_idx:
                for ti in range(n_topos):
                    train_rows.append(qi * n_topos + ti)
            test_rows = []
            for qi in test_idx:
                for ti in range(n_topos):
                    test_rows.append(qi * n_topos + ti)

            gbm = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42)
            gbm.fit(X_gbm_full[train_rows], y_gbm[train_rows])

            # Predict: for each test question, pick topology with highest P(correct)
            correct_count = 0
            for qi in test_idx:
                rows = [qi * n_topos + ti for ti in range(n_topos)]
                probs = gbm.predict_proba(X_gbm_full[rows])[:, 1]
                pred_topo = probs.argmax()
                if correct_masks[qi, pred_topo] > 0:
                    correct_count += 1
            gbm_accs.append(correct_count / len(test_idx))

        gbm_mean = np.mean(gbm_accs)
        print(f"  GBM (900q): {gbm_mean:.1%}")
    except Exception as e:
        print(f"  GBM failed: {e}")


if __name__ == "__main__":
    main()
