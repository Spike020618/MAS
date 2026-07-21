"""
Topology Router V4 — Systematic feature & training improvements

Baseline: V3c = 86.8% ± 0.3% on 900 questions (4 topos × 3 datasets × 300q)
Oracle: 93.9%, gap: 7.1pp

This script runs a battery of improvements:
  A. Interaction features: IC×(1-agr), EC×(1-agr), eq×ic_cv — "informed contrarian" pattern
  B. Rank features: per-question rank of each topology's signals among all 4 topos
  C. Sample weighting: upweight hard questions (fewer correct topos) during training
  D. Focal ranking loss: harder margin for harder examples
  E. Label smoothing: soft targets for partially correct topologies
  F. Combined best features with hyperparameter sweep

Usage:
  python3 experiments/train_router_v4.py
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "results/signals"
CONFIGS_DIR = "results/configs"
TRAIN_DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate"]


# ============================================================
# Feature Extraction — multiple feature sets
# ============================================================

def load_raw_data(n_questions=300):
    """Load all signals and correctness labels."""
    configs = {}
    for topo in TOPOS:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    questions = []
    for ds in TRAIN_DATASETS:
        all_signals = {}
        for topo in TOPOS:
            if topo == "parallel":
                path = os.path.join(SIGNALS_DIR, f"{ds}.json")
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")
            with open(path) as f:
                all_signals[topo] = json.load(f)

        n_agents = all_signals["parallel"]["meta"]["n_agents"]
        n_q = min(n_questions, len(all_signals["parallel"]["questions"]))

        for q_idx in range(n_q):
            q_data = {}
            topo_correct = np.zeros(4)
            valid = True

            for t_idx, topo in enumerate(TOPOS):
                if topo not in all_signals:
                    valid = False
                    break
                q = all_signals[topo]["questions"][q_idx]
                if "signals" not in q:
                    valid = False
                    break
                q_data[topo] = q["signals"]
                if topo in configs:
                    res = evaluate_config([q], configs[topo], n_agents)
                    topo_correct[t_idx] = 1.0 if res["correct"] > 0 else 0.0

            if valid:
                questions.append({"signals": q_data, "correct": topo_correct, "ds": ds})

    return questions


def extract_base_features(questions):
    """34-dim base features (V3c baseline)."""
    X = np.zeros((len(questions), 4, 34))
    correct_masks = np.zeros((len(questions), 4))

    for qi, q in enumerate(questions):
        correct_masks[qi] = q["correct"]
        for t_idx, topo in enumerate(TOPOS):
            s = q["signals"][topo]
            ic = np.array(s["ic"])
            ec = np.array(s["ec"])
            eq = np.array(s["eq"])
            ev = np.array(s["ev"])
            agr = np.array(s["agreement"])
            stances = s["stances"]
            sc = Counter(st.lower() for st in stances)
            majority = sc.most_common(1)[0][1] if sc else 0
            yes_like = sum(1 for st in stances if st.lower() in ['yes', 'true'])

            X[qi, t_idx, 0:5] = ic
            X[qi, t_idx, 5:10] = ec
            X[qi, t_idx, 10:15] = eq
            X[qi, t_idx, 15:20] = ev
            X[qi, t_idx, 20] = ic.mean()
            X[qi, t_idx, 21] = ic.std()
            X[qi, t_idx, 22] = ec.mean()
            X[qi, t_idx, 23] = ec.std()
            X[qi, t_idx, 24] = eq.mean()
            X[qi, t_idx, 25] = ev.mean()
            X[qi, t_idx, 26] = agr.mean()
            X[qi, t_idx, 27] = agr.min()
            X[qi, t_idx, 28] = agr.max()
            X[qi, t_idx, 29] = agr.std()
            X[qi, t_idx, 30] = majority / len(stances)
            X[qi, t_idx, 31] = 1.0 if yes_like > len(stances) / 2 else -1.0
            X[qi, t_idx, 32] = yes_like / len(stances)
            X[qi, t_idx, 33] = 1.0 - yes_like / len(stances)

    return X, correct_masks


def add_interaction_features(X_base, questions):
    """Add interaction features capturing 'informed contrarian' pattern.

    New features per topology (8 dims):
      ic_mean × (1 - agr_mean): high reasoning quality + low consensus
      ec_mean × (1 - agr_mean): high evidence relevance + low consensus
      eq_mean × ic_cv: evidence quality × reasoning variance
      ic_max - ic_min: IC spread (agent disagreement in reasoning quality)
      (ic_mean - agr_mean): direct gap between reasoning and consensus
      ic_mean × maj_ratio: reasoning quality × confidence
      agr_std × ic_mean: agreement variance weighted by reasoning
      ev_mean × (1 - maj_ratio): cross-agent consensus × answer uncertainty
    """
    n, n_topos, base_dim = X_base.shape
    X_new = np.zeros((n, n_topos, base_dim + 8))
    X_new[:, :, :base_dim] = X_base

    for qi in range(n):
        for ti in range(n_topos):
            ic_mean = X_base[qi, ti, 20]
            ic_std = X_base[qi, ti, 21]
            ec_mean = X_base[qi, ti, 22]
            eq_mean = X_base[qi, ti, 24]
            ev_mean = X_base[qi, ti, 25]
            agr_mean = X_base[qi, ti, 26]
            agr_std = X_base[qi, ti, 29]
            maj_ratio = X_base[qi, ti, 30]
            ic_cv = ic_std / (ic_mean + 1e-6)

            X_new[qi, ti, base_dim + 0] = ic_mean * (1 - agr_mean)
            X_new[qi, ti, base_dim + 1] = ec_mean * (1 - agr_mean)
            X_new[qi, ti, base_dim + 2] = eq_mean * ic_cv
            X_new[qi, ti, base_dim + 3] = X_base[qi, ti, 0:5].max() - X_base[qi, ti, 0:5].min()
            X_new[qi, ti, base_dim + 4] = ic_mean - agr_mean
            X_new[qi, ti, base_dim + 5] = ic_mean * maj_ratio
            X_new[qi, ti, base_dim + 6] = agr_std * ic_mean
            X_new[qi, ti, base_dim + 7] = ev_mean * (1 - maj_ratio)

    return X_new


def add_rank_features(X_base, questions):
    """Add per-question rank features: where does each topology rank on key metrics.

    New features per topology (6 dims):
      rank of ic_mean among 4 topos (0-1 normalized)
      rank of agr_mean
      rank of maj_ratio
      rank of ec_mean
      rank of eq_mean
      rank of ev_mean
    """
    n, n_topos, base_dim = X_base.shape
    X_new = np.zeros((n, n_topos, base_dim + 6))
    X_new[:, :, :base_dim] = X_base

    rank_indices = [20, 26, 30, 22, 24, 25]  # ic_mean, agr_mean, maj, ec, eq, ev

    for qi in range(n):
        for ri, feat_idx in enumerate(rank_indices):
            vals = X_base[qi, :, feat_idx]
            ranks = np.argsort(np.argsort(vals)).astype(float) / (n_topos - 1)
            X_new[qi, :, base_dim + ri] = ranks

    return X_new


def add_cross_topo_stance_features(X_base, questions):
    """Add cross-topology stance agreement features.

    New features per topology (3 dims):
      cross_disagree: 1 if this topo's majority answer differs from cross-topo majority
      n_distinct_stances: number of distinct majority answers (normalized)
      stance_entropy: entropy of this topo's stance distribution
    """
    n, n_topos, base_dim = X_base.shape
    X_new = np.zeros((n, n_topos, base_dim + 3))
    X_new[:, :, :base_dim] = X_base

    for qi, q in enumerate(questions):
        maj_stances = {}
        stance_counts_all = {}
        for t_idx, topo in enumerate(TOPOS):
            sc = Counter(st.lower() for st in q["signals"][topo]["stances"])
            maj_stances[topo] = sc.most_common(1)[0][0]
            stance_counts_all[topo] = sc

        n_distinct = len(set(maj_stances.values()))

        for t_idx, topo in enumerate(TOPOS):
            others = [maj_stances[t] for t in TOPOS if t != topo]
            cross_maj = Counter(others).most_common(1)[0][0]
            X_new[qi, t_idx, base_dim + 0] = 1.0 if maj_stances[topo] != cross_maj else 0.0
            X_new[qi, t_idx, base_dim + 1] = n_distinct / n_topos
            sc = stance_counts_all[topo]
            probs = np.array(list(sc.values())) / sum(sc.values())
            X_new[qi, t_idx, base_dim + 2] = -np.sum(probs * np.log(probs + 1e-10))

    return X_new


# ============================================================
# Model
# ============================================================

class RouterV4(nn.Module):
    def __init__(self, feat_dim, hidden=128, n_topos=4):
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
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores, correct_mask, weights=None):
        batch_size, n_topos = scores.shape
        loss = torch.tensor(0.0, device=scores.device)
        n_pairs = 0

        for b in range(batch_size):
            correct_idx = torch.where(correct_mask[b] > 0)[0]
            incorrect_idx = torch.where(correct_mask[b] == 0)[0]
            if len(correct_idx) == 0 or len(incorrect_idx) == 0:
                continue
            w = weights[b] if weights is not None else 1.0
            for c in correct_idx:
                for w_idx in incorrect_idx:
                    loss += w * F.relu(self.margin - (scores[b, c] - scores[b, w_idx]))
                    n_pairs += 1

        return loss / max(n_pairs, 1)


class WeightedRankingLoss(nn.Module):
    """Ranking loss with difficulty-aware weighting.
    Questions with fewer correct topologies get higher weight."""
    def __init__(self, margin=1.0):
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

            n_correct = len(correct_idx)
            # Weight: inverse of number of correct topos
            # 1 correct → weight 4, 2 correct → weight 2, 3 correct → weight 1.33, 4 correct → weight 1
            weight = n_topos / n_correct

            for c in correct_idx:
                for w in incorrect_idx:
                    loss += weight * F.relu(self.margin - (scores[b, c] - scores[b, w]))
                    n_pairs += 1

        return loss / max(n_pairs, 1)


class AdaptiveMarginRankingLoss(nn.Module):
    """Margin scales with difficulty: harder questions need larger separation."""
    def __init__(self, base_margin=0.5, max_margin=2.0):
        super().__init__()
        self.base_margin = base_margin
        self.max_margin = max_margin

    def forward(self, scores, correct_mask):
        batch_size, n_topos = scores.shape
        loss = torch.tensor(0.0, device=scores.device)
        n_pairs = 0

        for b in range(batch_size):
            correct_idx = torch.where(correct_mask[b] > 0)[0]
            incorrect_idx = torch.where(correct_mask[b] == 0)[0]
            if len(correct_idx) == 0 or len(incorrect_idx) == 0:
                continue

            n_correct = len(correct_idx)
            margin = self.base_margin + (self.max_margin - self.base_margin) * (n_topos - n_correct) / n_topos

            for c in correct_idx:
                for w in incorrect_idx:
                    loss += F.relu(margin - (scores[b, c] - scores[b, w]))
                    n_pairs += 1

        return loss / max(n_pairs, 1)


# ============================================================
# Dataset with sample weights
# ============================================================

class TopologyDataset(Dataset):
    def __init__(self, X, correct_masks, weights=None):
        self.X = torch.FloatTensor(X)
        self.correct_masks = torch.FloatTensor(correct_masks)
        self.weights = torch.FloatTensor(weights) if weights is not None else torch.ones(len(X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.correct_masks[idx], self.weights[idx]


# ============================================================
# Training
# ============================================================

def train_eval(X, correct_masks, seed=42, n_folds=5, epochs=200,
               lr=1e-3, hidden=128, margin=1.0, loss_type="ranking",
               use_weighted_sampling=False, sample_weight_power=1.0):
    """Train with CV, return mean accuracy."""
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

        # Normalize
        X_train_flat = X_train.reshape(-1, feat_dim)
        X_test_flat = X_test.reshape(-1, feat_dim)
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)
        X_train_norm = X_train_flat.reshape(len(train_idx), n_topos, feat_dim)
        X_test_norm = X_test_flat.reshape(len(test_idx), n_topos, feat_dim)

        # Sample weights based on difficulty
        n_correct_train = cm_train.sum(axis=1)
        if use_weighted_sampling:
            weights = np.where(n_correct_train == 0, 1.0,
                      np.where(n_correct_train == 1, 4.0,
                      np.where(n_correct_train == 2, 2.0,
                      np.where(n_correct_train == 3, 1.0, 0.5)))) ** sample_weight_power
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                           generator=torch.Generator().manual_seed(seed))
            train_ds = TopologyDataset(X_train_norm, cm_train)
            train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
        else:
            train_ds = TopologyDataset(X_train_norm, cm_train)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                      generator=torch.Generator().manual_seed(seed))

        test_ds = TopologyDataset(X_test_norm, cm_test)
        test_loader = DataLoader(test_ds, batch_size=128)

        torch.manual_seed(seed + fold)
        model = RouterV4(feat_dim=feat_dim, hidden=hidden, n_topos=n_topos).to(device)

        if loss_type == "ranking":
            criterion = PairwiseRankingLoss(margin=margin)
        elif loss_type == "weighted_ranking":
            criterion = WeightedRankingLoss(margin=margin)
        elif loss_type == "adaptive_margin":
            criterion = AdaptiveMarginRankingLoss(base_margin=margin*0.5, max_margin=margin*2)
        else:
            criterion = PairwiseRankingLoss(margin=margin)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        patience = 30
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                X_batch, cm_batch = batch[0].to(device), batch[1].to(device)
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
                for batch in test_loader:
                    X_batch = batch[0].to(device)
                    cm_batch = batch[1]
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

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


# ============================================================
# Main: Run all experiments
# ============================================================

def main():
    print("=" * 70)
    print("Router V4 — Systematic Feature & Training Improvements")
    print("=" * 70)

    print("\n[1] Loading data...")
    questions = load_raw_data(n_questions=300)
    X_base, correct_masks = extract_base_features(questions)
    print(f"  {len(questions)} questions, base features: {X_base.shape}")

    n_correct_dist = correct_masks.sum(axis=1)
    oracle = (n_correct_dist > 0).mean()
    star_acc = correct_masks[:, TOPOS.index("star")].mean()
    print(f"  Star: {star_acc:.1%}, Oracle: {oracle:.1%}")

    results = {}

    # ---- Experiment A: Baseline (reproduce V3c) ----
    print("\n" + "=" * 70)
    print("[A] Baseline: V3c (34-dim, ranking loss, margin=1.0)")
    print("=" * 70)
    accs = []
    for seed in [42, 52, 62, 72, 82]:
        acc, std, folds = train_eval(X_base, correct_masks, seed=seed, margin=1.0)
        accs.append(acc)
    results["A_baseline"] = np.mean(accs)
    print(f"  → {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # ---- Experiment B: Interaction features ----
    print("\n" + "=" * 70)
    print("[B] Interaction features (+8 dims: IC×(1-agr), EC×(1-agr), etc.)")
    print("=" * 70)
    X_interact = add_interaction_features(X_base, questions)
    print(f"  Feature dim: {X_interact.shape[2]}")
    accs = []
    for seed in [42, 52, 62, 72, 82]:
        acc, std, folds = train_eval(X_interact, correct_masks, seed=seed, margin=1.0)
        accs.append(acc)
    results["B_interaction"] = np.mean(accs)
    print(f"  → {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # ---- Experiment C: Rank features ----
    print("\n" + "=" * 70)
    print("[C] Rank features (+6 dims: per-question rank of key metrics)")
    print("=" * 70)
    X_rank = add_rank_features(X_base, questions)
    print(f"  Feature dim: {X_rank.shape[2]}")
    accs = []
    for seed in [42, 52, 62, 72, 82]:
        acc, std, folds = train_eval(X_rank, correct_masks, seed=seed, margin=1.0)
        accs.append(acc)
    results["C_rank"] = np.mean(accs)
    print(f"  → {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # ---- Experiment D: Cross-topo stance features ----
    print("\n" + "=" * 70)
    print("[D] Cross-topo stance features (+3 dims: disagree, n_distinct, entropy)")
    print("=" * 70)
    X_cross = add_cross_topo_stance_features(X_base, questions)
    print(f"  Feature dim: {X_cross.shape[2]}")
    accs = []
    for seed in [42, 52, 62, 72, 82]:
        acc, std, folds = train_eval(X_cross, correct_masks, seed=seed, margin=1.0)
        accs.append(acc)
    results["D_cross_stance"] = np.mean(accs)
    print(f"  → {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # ---- Experiment E: All features combined ----
    print("\n" + "=" * 70)
    print("[E] All features combined (34 + 8 + 6 + 3 = 51 dims)")
    print("=" * 70)
    X_all = add_interaction_features(X_base, questions)
    X_all = add_rank_features(X_all, questions)
    X_all = add_cross_topo_stance_features(X_all, questions)
    print(f"  Feature dim: {X_all.shape[2]}")
    accs = []
    for seed in [42, 52, 62, 72, 82]:
        acc, std, folds = train_eval(X_all, correct_masks, seed=seed, margin=1.0)
        accs.append(acc)
    results["E_all_features"] = np.mean(accs)
    print(f"  → {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # ---- Experiment F: Weighted ranking loss ----
    print("\n" + "=" * 70)
    print("[F] Weighted ranking loss (harder questions → higher weight)")
    print("=" * 70)
    # Use the best feature set from above
    best_feat_key = max(["A_baseline", "B_interaction", "C_rank", "D_cross_stance", "E_all_features"],
                       key=lambda k: results[k])
    feat_map = {"A_baseline": X_base, "B_interaction": X_interact,
                "C_rank": X_rank, "D_cross_stance": X_cross, "E_all_features": X_all}
    X_best = feat_map[best_feat_key]
    print(f"  Using best features from {best_feat_key}: {X_best.shape[2]} dims")

    accs = []
    for seed in [42, 52, 62, 72, 82]:
        acc, std, folds = train_eval(X_best, correct_masks, seed=seed, margin=1.0,
                                     loss_type="weighted_ranking")
        accs.append(acc)
    results["F_weighted_loss"] = np.mean(accs)
    print(f"  → {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # ---- Experiment G: Adaptive margin ----
    print("\n" + "=" * 70)
    print("[G] Adaptive margin (harder questions → larger margin)")
    print("=" * 70)
    accs = []
    for seed in [42, 52, 62, 72, 82]:
        acc, std, folds = train_eval(X_best, correct_masks, seed=seed, margin=1.0,
                                     loss_type="adaptive_margin")
        accs.append(acc)
    results["G_adaptive_margin"] = np.mean(accs)
    print(f"  → {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # ---- Experiment H: Weighted sampling ----
    print("\n" + "=" * 70)
    print("[H] Weighted sampling (oversample hard questions)")
    print("=" * 70)
    accs = []
    for seed in [42, 52, 62, 72, 82]:
        acc, std, folds = train_eval(X_best, correct_masks, seed=seed, margin=1.0,
                                     use_weighted_sampling=True, sample_weight_power=1.0)
        accs.append(acc)
    results["H_weighted_sampling"] = np.mean(accs)
    print(f"  → {np.mean(accs):.1%} ± {np.std(accs):.1%}")

    # ---- Experiment I: Best features + weighted loss + hyperparams ----
    print("\n" + "=" * 70)
    print("[I] Hyperparameter sweep on best combo")
    print("=" * 70)
    best_loss_key = max(["F_weighted_loss", "G_adaptive_margin", "A_baseline"],
                       key=lambda k: results.get(k, 0))
    loss_map = {"F_weighted_loss": "weighted_ranking",
                "G_adaptive_margin": "adaptive_margin",
                "A_baseline": "ranking"}
    best_loss = loss_map.get(best_loss_key, "ranking")
    print(f"  Best loss: {best_loss} (from {best_loss_key})")

    for hidden in [64, 128, 192]:
        for margin in [0.5, 1.0, 1.5]:
            for lr in [5e-4, 1e-3]:
                accs = []
                for seed in [42, 52, 62]:
                    acc, _, _ = train_eval(X_best, correct_masks, seed=seed,
                                           hidden=hidden, margin=margin, lr=lr,
                                           loss_type=best_loss)
                    accs.append(acc)
                mean_acc = np.mean(accs)
                if mean_acc > results.get("I_best_hyperparam", 0):
                    results["I_best_hyperparam"] = mean_acc
                    results["I_best_config"] = f"h={hidden},m={margin},lr={lr}"
                    print(f"  NEW BEST: {mean_acc:.1%} (h={hidden}, m={margin}, lr={lr})")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Star baseline:     {star_acc:.1%}")
    for key in sorted(results.keys()):
        if key.startswith("I_best_config"):
            continue
        val = results[key]
        if isinstance(val, float):
            marker = " ← BEST" if val == max(v for v in results.values() if isinstance(v, float)) else ""
            print(f"  {key:25s}: {val:.1%}{marker}")
        else:
            print(f"  {key:25s}: {val}")
    print(f"  Oracle:              {oracle:.1%}")
    print(f"  Best config:         {results.get('I_best_config', 'N/A')}")


if __name__ == "__main__":
    main()
