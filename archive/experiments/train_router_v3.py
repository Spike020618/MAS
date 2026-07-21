"""
Topology Router V3 — Shared Encoder + Pairwise Ranking Loss

改进点:
  1. 网络结构: 共享编码器为每个拓扑独立打分（不再 flatten 所有拓扑）
  2. Loss: Pairwise margin ranking loss（一道题产生多对训练信号）
  3. 数据增强: 每道题的 correct/incorrect 拓扑对组合

对比:
  V1 (neural): 4-class CE on (4,5,4) tensor → 80.5%
  V2 (GBM): per-(q,topo) correctness prediction → 83.8%
  V3 (this): shared encoder + ranking loss → ?

Usage:
  python experiments/train_router_v3.py
  python experiments/train_router_v3.py --ablation   # 消融: 对比不同 loss/结构
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
# Network Architecture: Shared Encoder + Cross-Topology Scorer
# ============================================================

class SharedEncoderRouter(nn.Module):
    """
    每个拓扑的 signals 用同一个编码器处理，
    再结合跨拓扑上下文做打分。
    """

    def __init__(self, n_agents: int = 5, n_signals: int = 4,
                 hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.n_signals = n_signals

        # Shared encoder: 处理单个拓扑的 (n_agents × n_signals) → embedding
        self.signal_encoder = nn.Sequential(
            nn.Linear(n_agents * n_signals, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Context encoder: 处理其他拓扑的摘要信息
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
        )

        # Scorer: embedding + context → scalar score
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_single(self, x_topo):
        """Encode one topology's signals. x_topo: (batch, n_agents * n_signals)"""
        return self.signal_encoder(x_topo)

    def forward(self, x_all):
        """
        x_all: (batch, n_topos, n_agents * n_signals)
        Returns: (batch, n_topos) scores
        """
        batch_size, n_topos, feat_dim = x_all.shape

        # Encode all topologies with shared encoder
        embeddings = []
        for t in range(n_topos):
            emb = self.encode_single(x_all[:, t, :])  # (batch, hidden//2)
            embeddings.append(emb)
        embeddings = torch.stack(embeddings, dim=1)  # (batch, n_topos, hidden//2)

        # For each topology, compute context from others
        scores = []
        for t in range(n_topos):
            # Context = mean of other topologies' embeddings
            mask = torch.ones(n_topos, dtype=torch.bool)
            mask[t] = False
            context = embeddings[:, mask, :].mean(dim=1)  # (batch, hidden//2)
            context_enc = self.context_encoder(context)  # (batch, hidden//4)

            # Score = f(embedding, context)
            combined = torch.cat([embeddings[:, t, :], context_enc], dim=1)
            score = self.scorer(combined).squeeze(-1)  # (batch,)
            scores.append(score)

        return torch.stack(scores, dim=1)  # (batch, n_topos)

    def score_single(self, x_target, x_context_mean):
        """Score a single topology given precomputed context."""
        emb = self.encode_single(x_target)
        ctx = self.context_encoder(x_context_mean)
        combined = torch.cat([emb, ctx], dim=1)
        return self.scorer(combined).squeeze(-1)


class DeepSharedRouter(nn.Module):
    """
    更深的版本: 加入 agent-level attention。
    先对 agents 做 attention pooling，再跨拓扑对比。
    """

    def __init__(self, n_agents: int = 5, n_signals: int = 4,
                 hidden_dim: int = 64):
        super().__init__()

        # Agent-level encoder
        self.agent_proj = nn.Sequential(
            nn.Linear(n_signals, hidden_dim // 2),
            nn.GELU(),
        )

        # Attention pooling over agents
        self.attn_query = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.attn_key = nn.Linear(hidden_dim // 2, hidden_dim // 4)

        # Post-attention projection
        self.post_attn = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Cross-topology context
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
        )

        # Final scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.n_agents = n_agents
        self.n_signals = n_signals

    def encode_topo(self, x):
        """x: (batch, n_agents, n_signals) → (batch, hidden//2)"""
        # Per-agent projection
        agent_embs = self.agent_proj(x)  # (batch, n_agents, hidden//2)

        # Attention pooling
        Q = self.attn_query(agent_embs)  # (batch, n_agents, hidden//4)
        K = self.attn_key(agent_embs)
        attn = torch.bmm(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        pooled = torch.bmm(attn, agent_embs).mean(dim=1)  # (batch, hidden//2)

        return self.post_attn(pooled)

    def forward(self, x_all):
        """x_all: (batch, n_topos, n_agents, n_signals) → (batch, n_topos) scores"""
        batch_size, n_topos = x_all.shape[0], x_all.shape[1]

        embeddings = []
        for t in range(n_topos):
            emb = self.encode_topo(x_all[:, t, :, :])
            embeddings.append(emb)
        embeddings = torch.stack(embeddings, dim=1)  # (batch, n_topos, hidden//2)

        scores = []
        for t in range(n_topos):
            mask = torch.ones(n_topos, dtype=torch.bool)
            mask[t] = False
            context = embeddings[:, mask, :].mean(dim=1)
            context_enc = self.context_proj(context)
            combined = torch.cat([embeddings[:, t, :], context_enc], dim=1)
            score = self.scorer(combined).squeeze(-1)
            scores.append(score)

        return torch.stack(scores, dim=1)


# ============================================================
# Pairwise Ranking Loss
# ============================================================

class PairwiseRankingLoss(nn.Module):
    """
    对每道题，构造 (correct_topo, incorrect_topo) pair，
    要求 score(correct) > score(incorrect) + margin。
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores, correct_mask):
        """
        scores: (batch, n_topos) — 每个拓扑的预测分数
        correct_mask: (batch, n_topos) — 1=该拓扑答对了, 0=答错了
        """
        batch_size, n_topos = scores.shape
        loss = torch.tensor(0.0, device=scores.device)
        n_pairs = 0

        for b in range(batch_size):
            correct_idx = torch.where(correct_mask[b] > 0)[0]
            incorrect_idx = torch.where(correct_mask[b] == 0)[0]

            if len(correct_idx) == 0 or len(incorrect_idx) == 0:
                continue

            # All pairs of (correct, incorrect)
            for c in correct_idx:
                for w in incorrect_idx:
                    # margin ranking: score_c - score_w > margin
                    loss += F.relu(self.margin - (scores[b, c] - scores[b, w]))
                    n_pairs += 1

        return loss / max(n_pairs, 1)


class ListwiseLoss(nn.Module):
    """
    Listwise ranking loss (ListNet style):
    用 correct_mask 作为 target distribution，最小化 KL divergence。
    """

    def forward(self, scores, correct_mask):
        """
        scores: (batch, n_topos)
        correct_mask: (batch, n_topos) — can be soft labels
        """
        # Target distribution: normalize correct_mask
        target = correct_mask.float()
        target_sum = target.sum(dim=1, keepdim=True).clamp(min=1)
        target_dist = target / target_sum  # (batch, n_topos)

        # Predicted distribution
        pred_dist = F.log_softmax(scores, dim=1)

        # KL divergence
        loss = F.kl_div(pred_dist, target_dist, reduction='batchmean')
        return loss


class CombinedLoss(nn.Module):
    """Ranking loss + correctness prediction auxiliary."""

    def __init__(self, margin: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.ranking = PairwiseRankingLoss(margin)
        self.alpha = alpha

    def forward(self, scores, correct_mask):
        rank_loss = self.ranking(scores, correct_mask)
        # Auxiliary: BCE for each topology's correctness
        bce_loss = F.binary_cross_entropy_with_logits(
            scores, correct_mask.float())
        return rank_loss + self.alpha * bce_loss


# ============================================================
# Dataset
# ============================================================

class TopologyRankingDataset(Dataset):
    """每个样本 = 一道题的所有拓扑 signals + 哪些拓扑正确。"""

    def __init__(self, X, correct_masks):
        """
        X: (N, n_topos, n_agents, n_signals) or (N, n_topos, n_agents*n_signals)
        correct_masks: (N, n_topos) binary
        """
        self.X = torch.FloatTensor(X)
        self.correct_masks = torch.FloatTensor(correct_masks)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.correct_masks[idx]


# ============================================================
# Data Loading
# ============================================================

def load_data(topos=None, n_questions=200):
    """Load signals and compute correctness for each (question, topology)."""
    if topos is None:
        topos = ["parallel", "sequential", "star", "debate"]

    configs = {}
    for topo in topos:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    X_raw = []       # (N, n_topos, n_agents, n_signals)
    correct_masks = []  # (N, n_topos)

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
                X_raw.append(raw_matrix)
                correct_masks.append(topo_correct)

    return np.array(X_raw), np.array(correct_masks), topos


# ============================================================
# Training
# ============================================================

def train_and_evaluate(X, correct_masks, topos, model_type="shared",
                       loss_type="ranking", n_folds=5, epochs=200,
                       lr=1e-3, hidden_dim=64, margin=1.0):
    """Cross-validated training with specified model and loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_topos = len(topos)
    n_questions = len(X)

    # Stratify by "best topology" label
    y_stratify = correct_masks.argmax(axis=1)
    # Handle all-zero rows
    all_wrong = correct_masks.sum(axis=1) == 0
    y_stratify[all_wrong] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_stratify)):
        X_train, X_test = X[train_idx], X[test_idx]
        cm_train, cm_test = correct_masks[train_idx], correct_masks[test_idx]

        # Normalize
        shape = X_train.shape
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)
        X_train_norm = X_train_flat.reshape(shape)
        X_test_norm = X_test_flat.reshape((len(X_test),) + shape[1:])

        # Prepare data
        if model_type == "shared":
            # Flatten agent dims for SharedEncoderRouter
            X_tr = X_train_norm.reshape(len(X_train_norm), n_topos, -1)
            X_te = X_test_norm.reshape(len(X_test_norm), n_topos, -1)
        else:
            X_tr = X_train_norm
            X_te = X_test_norm

        train_ds = TopologyRankingDataset(X_tr, cm_train)
        test_ds = TopologyRankingDataset(X_te, cm_test)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        # Model
        if model_type == "deep":
            model = DeepSharedRouter(
                n_agents=shape[2], n_signals=shape[3],
                hidden_dim=hidden_dim).to(device)
        else:
            model = SharedEncoderRouter(
                n_agents=shape[2], n_signals=shape[3],
                hidden_dim=hidden_dim).to(device)

        # Loss
        if loss_type == "ranking":
            criterion = PairwiseRankingLoss(margin=margin)
        elif loss_type == "listwise":
            criterion = ListwiseLoss()
        elif loss_type == "combined":
            criterion = CombinedLoss(margin=margin, alpha=0.5)
        else:
            criterion = ListwiseLoss()  # default

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_routing_acc = 0
        patience = 30
        no_improve = 0

        for epoch in range(epochs):
            # Train
            model.train()
            total_loss = 0
            for X_batch, cm_batch in train_loader:
                X_batch = X_batch.to(device)
                cm_batch = cm_batch.to(device)
                optimizer.zero_grad()
                scores = model(X_batch)
                loss = criterion(scores, cm_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            # Eval
            model.eval()
            routing_correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, cm_batch in test_loader:
                    X_batch = X_batch.to(device)
                    scores = model(X_batch)
                    preds = scores.argmax(dim=1).cpu().numpy()
                    for i, pred in enumerate(preds):
                        if cm_batch[i, pred] > 0:
                            routing_correct += 1
                        total += 1

            routing_acc = routing_correct / total
            if routing_acc > best_routing_acc:
                best_routing_acc = routing_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Baselines for this fold
        star_idx = topos.index("star") if "star" in topos else 0
        star_acc = cm_test[:, star_idx].mean()

        # Oracle
        oracle_acc = (cm_test.sum(axis=1) > 0).mean()

        fold_results.append({
            "routing_acc": best_routing_acc,
            "star_acc": star_acc,
            "oracle_acc": oracle_acc,
            "epochs_used": epoch + 1 - patience if no_improve >= patience else epoch + 1,
        })
        print(f"  Fold {fold+1}: routing={best_routing_acc:.1%} "
              f"(star={star_acc:.1%}, oracle={oracle_acc:.1%})")

    avg_routing = np.mean([r["routing_acc"] for r in fold_results])
    avg_star = np.mean([r["star_acc"] for r in fold_results])
    avg_oracle = np.mean([r["oracle_acc"] for r in fold_results])
    return avg_routing, avg_star, avg_oracle, fold_results


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--n_questions", type=int, default=200)
    args = parser.parse_args()

    print("=" * 60)
    print("Topology Router V3 — Shared Encoder + Ranking Loss")
    print("=" * 60)

    print(f"\n[1] Loading data (first {args.n_questions} questions)...")
    X, correct_masks, topos = load_data(n_questions=args.n_questions)
    print(f"  Shape: {X.shape}")
    print(f"  Questions: {len(X)}")
    print(f"  Avg correct topos per question: {correct_masks.sum(axis=1).mean():.2f}")
    print(f"  Pairs per question (avg): "
          f"{(correct_masks.sum(axis=1) * (len(topos) - correct_masks.sum(axis=1))).mean():.1f}")

    # Main experiment: SharedEncoder + PairwiseRanking
    print(f"\n[2] SharedEncoder + PairwiseRanking (margin={args.margin})...")
    acc_rank, star, oracle, _ = train_and_evaluate(
        X, correct_masks, topos,
        model_type="shared", loss_type="ranking",
        epochs=args.epochs, hidden_dim=args.hidden,
        lr=args.lr, margin=args.margin)

    print(f"\n[3] SharedEncoder + ListwiseLoss...")
    acc_list, _, _, _ = train_and_evaluate(
        X, correct_masks, topos,
        model_type="shared", loss_type="listwise",
        epochs=args.epochs, hidden_dim=args.hidden, lr=args.lr)

    print(f"\n[4] SharedEncoder + CombinedLoss...")
    acc_comb, _, _, _ = train_and_evaluate(
        X, correct_masks, topos,
        model_type="shared", loss_type="combined",
        epochs=args.epochs, hidden_dim=args.hidden,
        lr=args.lr, margin=args.margin)

    print(f"\n[5] DeepRouter (attention) + PairwiseRanking...")
    acc_deep, _, _, _ = train_and_evaluate(
        X, correct_masks, topos,
        model_type="deep", loss_type="ranking",
        epochs=args.epochs, hidden_dim=args.hidden,
        lr=args.lr, margin=args.margin)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Always Star:                    {star:.1%}")
    print(f"  Confidence Routing (rule):      83.7%")
    print(f"  GBM Router V2 (hand features):  83.8%")
    print(f"  ---")
    print(f"  SharedEncoder + Ranking:        {acc_rank:.1%}")
    print(f"  SharedEncoder + Listwise:       {acc_list:.1%}")
    print(f"  SharedEncoder + Combined:       {acc_comb:.1%}")
    print(f"  DeepRouter + Ranking:           {acc_deep:.1%}")
    print(f"  ---")
    print(f"  Oracle:                         {oracle:.1%}")

    if args.ablation:
        print(f"\n[6] Ablation: margin sensitivity...")
        for m in [0.3, 0.5, 1.0, 2.0, 3.0]:
            acc, _, _, _ = train_and_evaluate(
                X, correct_masks, topos,
                model_type="shared", loss_type="ranking",
                epochs=args.epochs, hidden_dim=args.hidden,
                lr=args.lr, margin=m)
            print(f"  margin={m:.1f}: {acc:.1%}")

        print(f"\n[7] Ablation: hidden dim...")
        for h in [32, 64, 128]:
            acc, _, _, _ = train_and_evaluate(
                X, correct_masks, topos,
                model_type="shared", loss_type="ranking",
                epochs=args.epochs, hidden_dim=h,
                lr=args.lr, margin=args.margin)
            print(f"  hidden={h}: {acc:.1%}")


if __name__ == "__main__":
    main()
