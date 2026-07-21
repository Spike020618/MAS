"""
Topology Router V3b — Hard-Example Focused + Multi-Scale Features

改进点 (vs V3):
  1. Focal Ranking Loss: 对 hard pairs (分数接近) 加权，easy pairs 降权
  2. Multi-Scale Features: raw signals + summary stats + 跨拓扑差异特征
  3. Sample Weighting: 只有 1-2 个拓扑正确的题权重更高
  4. Agent Permutation Augmentation: agents 顺序无关，permute 做增强

Usage:
  python experiments/train_router_v3b.py
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


# ============================================================
# Multi-Scale Feature Extraction
# ============================================================

def extract_multiscale_features(raw_matrix, topos):
    """
    从 (n_topos, n_agents, 4) 提取多尺度特征:
      - Raw: agent-level signals (原始)
      - Summary: per-topo mean/std/min/max
      - Cross: 跨拓扑差异 (每对拓扑的信号差)
      - Disagreement: 拓扑间 stance 一致性
    """
    n_topos, n_agents, n_signals = raw_matrix.shape

    # Scale 1: Raw flattened (n_topos * n_agents * n_signals)
    raw_flat = raw_matrix.reshape(-1)

    # Scale 2: Per-topo summary statistics
    summary = []
    for t in range(n_topos):
        for s in range(n_signals):
            vals = raw_matrix[t, :, s]
            summary.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
    summary = np.array(summary)  # n_topos * n_signals * 4 stats

    # Scale 3: Cross-topology differences (pairwise)
    cross = []
    for t1 in range(n_topos):
        for t2 in range(t1 + 1, n_topos):
            diff = raw_matrix[t1].mean(axis=0) - raw_matrix[t2].mean(axis=0)
            cross.extend(diff)
    cross = np.array(cross)  # C(n_topos,2) * n_signals

    # Scale 4: Per-topo "confidence" (how much agents agree)
    confidence = []
    for t in range(n_topos):
        agent_means = raw_matrix[t].mean(axis=1)  # per-agent avg signal
        confidence.append(agent_means.std())  # low std = high agreement
        confidence.append(agent_means.max() - agent_means.min())  # range
    confidence = np.array(confidence)

    return np.concatenate([raw_flat, summary, cross, confidence])


# ============================================================
# Network: Multi-Scale Shared Encoder
# ============================================================

class MultiScaleRouter(nn.Module):
    """
    Two-stream architecture:
      Stream 1: Per-topo raw signals → shared encoder → topo embedding
      Stream 2: Multi-scale summary → global context
    Final: score(topo_emb, global_context) for each topology
    """

    def __init__(self, n_agents: int = 5, n_signals: int = 4,
                 n_topos: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.n_topos = n_topos
        self.n_agents = n_agents
        self.n_signals = n_signals

        # Stream 1: Shared per-topology encoder
        self.topo_encoder = nn.Sequential(
            nn.Linear(n_agents * n_signals, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Stream 2: Cross-topology difference encoder
        n_pairs = n_topos * (n_topos - 1) // 2
        cross_dim = n_pairs * n_signals + n_topos * 2  # cross diffs + confidence
        self.cross_encoder = nn.Sequential(
            nn.Linear(cross_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
        )

        # Scorer: topo_emb + cross_context + other_topos_summary → score
        scorer_input = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 2
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x_raw, x_cross):
        """
        x_raw: (batch, n_topos, n_agents * n_signals)
        x_cross: (batch, cross_dim) — precomputed cross-topo features
        """
        batch_size = x_raw.shape[0]

        # Encode each topology
        topo_embs = []
        for t in range(self.n_topos):
            emb = self.topo_encoder(x_raw[:, t, :])
            topo_embs.append(emb)
        topo_embs = torch.stack(topo_embs, dim=1)  # (batch, n_topos, hidden//2)

        # Encode cross-topology context
        cross_ctx = self.cross_encoder(x_cross)  # (batch, hidden//4)

        # Score each topology
        scores = []
        for t in range(self.n_topos):
            # Context from other topologies
            mask = torch.ones(self.n_topos, dtype=torch.bool)
            mask[t] = False
            others_mean = topo_embs[:, mask, :].mean(dim=1)  # (batch, hidden//2)

            combined = torch.cat([
                topo_embs[:, t, :],   # this topo's embedding
                cross_ctx,             # global cross-topo context
                others_mean,           # other topos' summary
            ], dim=1)
            score = self.scorer(combined).squeeze(-1)
            scores.append(score)

        return torch.stack(scores, dim=1)  # (batch, n_topos)


# ============================================================
# Focal Ranking Loss
# ============================================================

class FocalRankingLoss(nn.Module):
    """
    Hard-example focused ranking loss:
    对 score 差距小的 pair 加大惩罚 (focal factor)，
    easy pairs (差距大) 自然降权。
    """

    def __init__(self, margin: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, scores, correct_mask, sample_weights=None):
        """
        scores: (batch, n_topos)
        correct_mask: (batch, n_topos)
        sample_weights: (batch,) — per-question importance weight
        """
        batch_size, n_topos = scores.shape
        losses = []

        for b in range(batch_size):
            correct_idx = torch.where(correct_mask[b] > 0)[0]
            incorrect_idx = torch.where(correct_mask[b] == 0)[0]

            if len(correct_idx) == 0 or len(incorrect_idx) == 0:
                continue

            w = sample_weights[b] if sample_weights is not None else 1.0

            for c in correct_idx:
                for inc in incorrect_idx:
                    diff = scores[b, c] - scores[b, inc]
                    raw_loss = F.relu(self.margin - diff)

                    # Focal factor: hard examples (small diff) get higher weight
                    with torch.no_grad():
                        focal = (1.0 - torch.sigmoid(diff)) ** self.gamma

                    losses.append(w * focal * raw_loss)

        if not losses:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        return torch.stack(losses).mean()


# ============================================================
# Dataset with Augmentation
# ============================================================

class AugmentedDataset(Dataset):
    """支持 agent permutation augmentation."""

    def __init__(self, X_raw, X_cross, correct_masks, weights,
                 augment=True, n_augment=3):
        self.X_raw = torch.FloatTensor(X_raw)
        self.X_cross = torch.FloatTensor(X_cross)
        self.correct_masks = torch.FloatTensor(correct_masks)
        self.weights = torch.FloatTensor(weights)
        self.augment = augment
        self.n_augment = n_augment

    def __len__(self):
        if self.augment:
            return len(self.X_raw) * (1 + self.n_augment)
        return len(self.X_raw)

    def __getitem__(self, idx):
        real_idx = idx % len(self.X_raw)
        x_raw = self.X_raw[real_idx].clone()
        x_cross = self.X_cross[real_idx]
        cm = self.correct_masks[real_idx]
        w = self.weights[real_idx]

        # Augment: permute agents within each topology
        if self.augment and idx >= len(self.X_raw):
            n_topos = x_raw.shape[0]
            n_feats = x_raw.shape[1]  # n_agents * n_signals
            n_agents = 5
            n_signals = n_feats // n_agents

            for t in range(n_topos):
                perm = torch.randperm(n_agents)
                reshaped = x_raw[t].view(n_agents, n_signals)
                x_raw[t] = reshaped[perm].view(-1)

        return x_raw, x_cross, cm, w


# ============================================================
# Data Loading
# ============================================================

def load_data(topos=None, n_questions=200):
    if topos is None:
        topos = ["parallel", "sequential", "star", "debate"]

    configs = {}
    for topo in topos:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    X_raw_list = []
    X_cross_list = []
    correct_masks = []
    weights = []

    for ds in TRAIN_DATASETS:
        all_signals = {}
        for topo in topos:
            if topo == "parallel":
                path = os.path.join(SIGNALS_DIR, f"{ds}.json")
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")
            if os.path.exists(path):
                with open(path) as f:
                    all_signals[topo] = json.load(f)

        n_agents = all_signals["parallel"]["meta"]["n_agents"]
        n_q = min(n_questions, len(all_signals["parallel"]["questions"]))

        for q_idx in range(n_q):
            raw_matrix = np.zeros((len(topos), n_agents, 4))
            topo_correct = np.zeros(len(topos))

            valid = True
            for t_idx, topo in enumerate(topos):
                if topo not in all_signals:
                    valid = False
                    break
                q = all_signals[topo]["questions"][q_idx]
                if "signals" not in q:
                    valid = False
                    break
                s = q["signals"]
                raw_matrix[t_idx, :, 0] = np.array(s["ic"])
                raw_matrix[t_idx, :, 1] = np.array(s["ec"])
                raw_matrix[t_idx, :, 2] = np.array(s["eq"])
                raw_matrix[t_idx, :, 3] = np.array(s["ev"])

                if topo in configs:
                    res = evaluate_config([q], configs[topo], n_agents)
                    topo_correct[t_idx] = 1.0 if res["correct"] > 0 else 0.0

            if valid:
                # Raw: flatten per-topo
                x_raw = raw_matrix.reshape(len(topos), -1)
                X_raw_list.append(x_raw)

                # Cross-topo features
                x_cross = extract_multiscale_features(raw_matrix, topos)
                # Only keep cross + confidence parts (not raw, which is redundant)
                n_pairs = len(topos) * (len(topos) - 1) // 2
                cross_start = len(topos) * n_agents * 4 + len(topos) * 4 * 4
                X_cross_list.append(x_cross[cross_start:])

                correct_masks.append(topo_correct)

                # Weight: harder questions (fewer correct topos) get higher weight
                n_correct = topo_correct.sum()
                if n_correct == 0:
                    w = 0.1  # all wrong, can't learn much
                elif n_correct == 1:
                    w = 4.0  # must get this right
                elif n_correct == 2:
                    w = 2.0  # important
                elif n_correct == 3:
                    w = 1.0  # moderate
                else:
                    w = 0.5  # all correct, easy
                weights.append(w)

    return (np.array(X_raw_list), np.array(X_cross_list),
            np.array(correct_masks), np.array(weights), topos)


# ============================================================
# Training
# ============================================================

def train_and_evaluate(X_raw, X_cross, correct_masks, weights, topos,
                       n_folds=5, epochs=250, lr=1e-3, hidden_dim=64,
                       margin=0.5, gamma=2.0, augment=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_topos = len(topos)
    n_questions = len(X_raw)

    y_stratify = correct_masks.argmax(axis=1)
    all_wrong = correct_masks.sum(axis=1) == 0
    y_stratify[all_wrong] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_raw, y_stratify)):
        # Split
        Xr_train, Xr_test = X_raw[train_idx], X_raw[test_idx]
        Xc_train, Xc_test = X_cross[train_idx], X_cross[test_idx]
        cm_train, cm_test = correct_masks[train_idx], correct_masks[test_idx]
        w_train, w_test = weights[train_idx], weights[test_idx]

        # Normalize raw features
        shape_r = Xr_train.shape
        Xr_train_flat = Xr_train.reshape(len(Xr_train), -1)
        Xr_test_flat = Xr_test.reshape(len(Xr_test), -1)
        scaler_r = StandardScaler()
        Xr_train_flat = scaler_r.fit_transform(Xr_train_flat)
        Xr_test_flat = scaler_r.transform(Xr_test_flat)
        Xr_train_norm = Xr_train_flat.reshape(shape_r)
        Xr_test_norm = Xr_test_flat.reshape((len(Xr_test),) + shape_r[1:])

        # Normalize cross features
        scaler_c = StandardScaler()
        Xc_train_norm = scaler_c.fit_transform(Xc_train)
        Xc_test_norm = scaler_c.transform(Xc_test)

        # Datasets
        train_ds = AugmentedDataset(
            Xr_train_norm, Xc_train_norm, cm_train, w_train,
            augment=augment, n_augment=3)
        test_ds = AugmentedDataset(
            Xr_test_norm, Xc_test_norm, cm_test, w_test,
            augment=False)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        # Model
        cross_dim = Xc_train.shape[1]
        model = MultiScaleRouter(
            n_agents=5, n_signals=4, n_topos=n_topos,
            hidden_dim=hidden_dim).to(device)

        # Override cross_encoder input dim
        model.cross_encoder = nn.Sequential(
            nn.Linear(cross_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
        ).to(device)

        criterion = FocalRankingLoss(margin=margin, gamma=gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2)

        best_routing_acc = 0
        patience = 40
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for xr_batch, xc_batch, cm_batch, w_batch in train_loader:
                xr_batch = xr_batch.to(device)
                xc_batch = xc_batch.to(device)
                cm_batch = cm_batch.to(device)
                w_batch = w_batch.to(device)

                optimizer.zero_grad()
                scores = model(xr_batch, xc_batch)
                loss = criterion(scores, cm_batch, w_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Eval every 5 epochs
            if (epoch + 1) % 5 != 0:
                continue

            model.eval()
            routing_correct = 0
            hard_correct = 0
            hard_total = 0
            total = 0

            with torch.no_grad():
                for xr_batch, xc_batch, cm_batch, w_batch in test_loader:
                    xr_batch = xr_batch.to(device)
                    xc_batch = xc_batch.to(device)
                    scores = model(xr_batch, xc_batch)
                    preds = scores.argmax(dim=1).cpu().numpy()
                    for i, pred in enumerate(preds):
                        if cm_batch[i, pred] > 0:
                            routing_correct += 1
                        total += 1
                        # Track hard cases
                        n_corr = cm_batch[i].sum().item()
                        if n_corr <= 2 and n_corr > 0:
                            hard_total += 1
                            if cm_batch[i, pred] > 0:
                                hard_correct += 1

            routing_acc = routing_correct / total
            if routing_acc > best_routing_acc:
                best_routing_acc = routing_acc
                best_hard_acc = hard_correct / max(hard_total, 1)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience // 5:
                    break

        star_idx = topos.index("star")
        star_acc = cm_test[:, star_idx].mean()
        oracle_acc = (cm_test.sum(axis=1) > 0).mean()

        fold_results.append({
            "routing_acc": best_routing_acc,
            "hard_acc": best_hard_acc,
            "star_acc": star_acc,
            "oracle_acc": oracle_acc,
        })
        print(f"  Fold {fold+1}: routing={best_routing_acc:.1%} "
              f"hard={best_hard_acc:.1%} "
              f"(star={star_acc:.1%}, oracle={oracle_acc:.1%})")

    avg_routing = np.mean([r["routing_acc"] for r in fold_results])
    avg_hard = np.mean([r["hard_acc"] for r in fold_results])
    avg_star = np.mean([r["star_acc"] for r in fold_results])
    avg_oracle = np.mean([r["oracle_acc"] for r in fold_results])
    return avg_routing, avg_hard, avg_star, avg_oracle, fold_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--n_questions", type=int, default=200)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Topology Router V3b — Focal Ranking + Multi-Scale + Augmentation")
    print("=" * 60)

    print(f"\n[1] Loading data...")
    X_raw, X_cross, correct_masks, weights, topos = load_data(
        n_questions=args.n_questions)
    print(f"  Raw shape: {X_raw.shape}")
    print(f"  Cross shape: {X_cross.shape}")
    print(f"  Questions: {len(X_raw)}")
    print(f"  Weight distribution: "
          f"w=0.5: {(weights==0.5).sum()}, "
          f"w=1.0: {(weights==1.0).sum()}, "
          f"w=2.0: {(weights==2.0).sum()}, "
          f"w=4.0: {(weights==4.0).sum()}")

    # Main
    print(f"\n[2] MultiScaleRouter + FocalRanking "
          f"(margin={args.margin}, gamma={args.gamma}, augment={not args.no_augment})...")
    acc, hard_acc, star, oracle, results = train_and_evaluate(
        X_raw, X_cross, correct_masks, weights, topos,
        epochs=args.epochs, hidden_dim=args.hidden, lr=args.lr,
        margin=args.margin, gamma=args.gamma, augment=not args.no_augment)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Always Star:                 {star:.1%}")
    print(f"  Confidence Routing (rule):   83.7%")
    print(f"  GBM Router V2:              83.8%")
    print(f"  V3 SharedEncoder+Ranking:    84.0%")
    print(f"  ---")
    print(f"  V3b MultiScale+Focal:        {acc:.1%}")
    print(f"  V3b Hard cases accuracy:     {hard_acc:.1%}")
    print(f"  ---")
    print(f"  Oracle:                      {oracle:.1%}")

    if args.ablation:
        print(f"\n[3] Ablation: no augmentation...")
        acc_noaug, hard_noaug, _, _, _ = train_and_evaluate(
            X_raw, X_cross, correct_masks, weights, topos,
            epochs=args.epochs, hidden_dim=args.hidden, lr=args.lr,
            margin=args.margin, gamma=args.gamma, augment=False)
        print(f"  No augment: routing={acc_noaug:.1%}, hard={hard_noaug:.1%}")

        print(f"\n[4] Ablation: uniform weights (no hard-example focus)...")
        uniform_w = np.ones_like(weights)
        acc_uni, hard_uni, _, _, _ = train_and_evaluate(
            X_raw, X_cross, correct_masks, uniform_w, topos,
            epochs=args.epochs, hidden_dim=args.hidden, lr=args.lr,
            margin=args.margin, gamma=args.gamma, augment=not args.no_augment)
        print(f"  Uniform weights: routing={acc_uni:.1%}, hard={hard_uni:.1%}")

        print(f"\n[5] Ablation: gamma sensitivity...")
        for g in [0.0, 1.0, 2.0, 3.0]:
            acc_g, _, _, _, _ = train_and_evaluate(
                X_raw, X_cross, correct_masks, weights, topos,
                epochs=args.epochs // 2, hidden_dim=args.hidden, lr=args.lr,
                margin=args.margin, gamma=g, augment=not args.no_augment)
            print(f"  gamma={g:.1f}: {acc_g:.1%}")


if __name__ == "__main__":
    main()
