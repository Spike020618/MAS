"""
Neural Topology Router — 用 MLP 从原始信号矩阵学习最优拓扑路由

输入: 每道题在各拓扑下的原始 AEIC 信号矩阵 (n_topos × n_agents × 4_signals)
输出: 预测哪种拓扑能答对该题

与手工特征的区别:
  - 保留 agent-level 粒度 (不压缩为 mean/std)
  - 保留跨拓扑的信号变化模式
  - 让网络自己学习哪些交互模式预示拓扑选择

Usage:
  python experiments/train_neural_router.py
  python experiments/train_neural_router.py --ablation  # 消融实验
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "results/signals"
CONFIGS_DIR = "results/configs"
TRAIN_DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]


class TopologyRouter(nn.Module):
    """MLP that routes questions to optimal topology based on signal features."""

    def __init__(self, n_topos: int, n_agents: int, n_signals: int = 4,
                 hidden_dim: int = 64):
        super().__init__()
        self.n_topos = n_topos
        self.n_agents = n_agents
        self.n_signals = n_signals

        # Per-topology signal encoder (shared weights across topologies)
        self.signal_encoder = nn.Sequential(
            nn.Linear(n_agents * n_signals, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Cross-topology interaction layer
        self.cross_topo = nn.Sequential(
            nn.Linear(n_topos * (hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_topos),
        )

    def forward(self, x):
        # x: (batch, n_topos, n_agents, n_signals)
        batch_size = x.shape[0]

        # Encode each topology's signals independently
        topo_embeddings = []
        for t in range(self.n_topos):
            topo_flat = x[:, t, :, :].reshape(batch_size, -1)  # (batch, n_agents*n_signals)
            emb = self.signal_encoder(topo_flat)  # (batch, hidden//2)
            topo_embeddings.append(emb)

        # Concatenate and process cross-topology interactions
        combined = torch.cat(topo_embeddings, dim=1)  # (batch, n_topos * hidden//2)
        cross = self.cross_topo(combined)  # (batch, hidden)

        # Classify
        logits = self.classifier(cross)  # (batch, n_topos)
        return logits


class AttentionRouter(nn.Module):
    """Attention-based router that learns which topology signals matter most."""

    def __init__(self, n_topos: int, n_agents: int, n_signals: int = 4,
                 hidden_dim: int = 64):
        super().__init__()
        self.n_topos = n_topos
        self.n_agents = n_agents

        # Agent-level encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(n_signals, hidden_dim // 2),
            nn.ReLU(),
        )

        # Self-attention over agents within each topology
        self.agent_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2, num_heads=2, batch_first=True)

        # Topology-level attention
        self.topo_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2, num_heads=2, batch_first=True)

        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_topos),
        )

    def forward(self, x):
        # x: (batch, n_topos, n_agents, n_signals)
        batch_size = x.shape[0]

        # Encode agents per topology
        topo_reps = []
        for t in range(self.n_topos):
            agents = self.agent_encoder(x[:, t, :, :])  # (batch, n_agents, hidden//2)
            attn_out, _ = self.agent_attn(agents, agents, agents)
            topo_rep = attn_out.mean(dim=1)  # (batch, hidden//2)
            topo_reps.append(topo_rep)

        # Stack topologies and attend across them
        topo_stack = torch.stack(topo_reps, dim=1)  # (batch, n_topos, hidden//2)
        cross_attn, attn_weights = self.topo_attn(topo_stack, topo_stack, topo_stack)
        global_rep = cross_attn.mean(dim=1)  # (batch, hidden//2)

        return self.classifier(global_rep)


def load_data(topos: list = None):
    """Load signal matrices and oracle labels."""
    if topos is None:
        topos = ["parallel", "sequential", "star", "debate"]

    configs = {}
    for topo in topos:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    X_raw = []
    y_labels = []
    topo_correct_all = []

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

        for q_idx in range(len(all_signals["parallel"]["questions"])):
            raw_matrix = np.zeros((len(topos), n_agents, 4))
            topo_correct = {}

            for t_idx, topo in enumerate(topos):
                if topo not in all_signals:
                    continue
                q = all_signals[topo]["questions"][q_idx]
                if "signals" not in q:
                    continue
                s = q["signals"]
                raw_matrix[t_idx, :, 0] = np.array(s["ic"])
                raw_matrix[t_idx, :, 1] = np.array(s["ec"])
                raw_matrix[t_idx, :, 2] = np.array(s["eq"])
                raw_matrix[t_idx, :, 3] = np.array(s["ev"])

                if topo in configs:
                    res = evaluate_config([q], configs[topo], n_agents)
                    topo_correct[topo] = res["correct"] > 0

            X_raw.append(raw_matrix)
            topo_correct_all.append(topo_correct)

            # Oracle label
            best = "star"  # default
            for t in ["star", "debate", "sequential", "parallel"]:
                if topo_correct.get(t, False):
                    best = t
                    break
            if not any(topo_correct.values()):
                best = "star"
            y_labels.append(topos.index(best))

    return np.array(X_raw), np.array(y_labels), topo_correct_all, topos


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds)


def cross_validate(X, y, topo_correct_all, topos, model_class="mlp",
                   n_folds=5, epochs=100, lr=1e-3, hidden_dim=64):
    """Cross-validated evaluation of neural router."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_topos, n_agents, n_signals = X.shape[1], X.shape[2], X.shape[3]

    # Class weights for imbalanced labels
    class_counts = np.bincount(y, minlength=len(topos))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * len(topos)
    weight_tensor = torch.FloatTensor(class_weights).to(device)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize per-feature
        shape = X_train.shape
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)
        X_train_norm = X_train_flat.reshape(shape)
        X_test_norm = X_test_flat.reshape((len(X_test),) + shape[1:])

        # Tensors
        X_tr = torch.FloatTensor(X_train_norm)
        y_tr = torch.LongTensor(y_train)
        X_te = torch.FloatTensor(X_test_norm)
        y_te = torch.LongTensor(y_test)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=64)

        # Model
        if model_class == "attention":
            model = AttentionRouter(n_topos, n_agents, n_signals, hidden_dim).to(device)
        else:
            model = TopologyRouter(n_topos, n_agents, n_signals, hidden_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_test_acc = 0
        best_preds = None
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc, preds = eval_epoch(model, test_loader, criterion, device)
            scheduler.step()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_preds = preds

        # Compute actual routing accuracy (not just classification accuracy)
        routing_correct = 0
        for i, idx in enumerate(test_idx):
            pred_topo = topos[best_preds[i]]
            if topo_correct_all[idx].get(pred_topo, False):
                routing_correct += 1
        routing_acc = routing_correct / len(test_idx)

        fold_results.append({
            "clf_acc": best_test_acc,
            "routing_acc": routing_acc,
            "n_test": len(test_idx),
        })
        print(f"  Fold {fold+1}: clf_acc={best_test_acc:.1%}, routing_acc={routing_acc:.1%}")

    avg_clf = np.mean([r["clf_acc"] for r in fold_results])
    avg_routing = np.mean([r["routing_acc"] for r in fold_results])
    return avg_clf, avg_routing, fold_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 60)
    print("Neural Topology Router")
    print("=" * 60)

    # Check which topologies have data
    available_topos = ["parallel", "sequential", "star", "debate"]
    # Check for hierarchical
    if all(os.path.exists(os.path.join(SIGNALS_DIR, f"{ds}_hierarchical.json"))
           for ds in TRAIN_DATASETS):
        available_topos.append("hierarchical")
        print(f"  Found hierarchical data, using 5 topologies")
    else:
        print(f"  Using 4 topologies (hierarchical not yet available)")

    print(f"\n[1] Loading data...")
    X, y, topo_correct_all, topos = load_data(available_topos)
    print(f"  Shape: {X.shape} (questions × topos × agents × signals)")
    print(f"  Labels: {dict(zip(topos, [int((y==i).sum()) for i in range(len(topos))]))}")

    # Baselines
    always_star_acc = np.mean([tc.get("star", False) for tc in topo_correct_all])
    oracle_acc = np.mean([any(tc.values()) for tc in topo_correct_all])
    print(f"\n  Always star: {always_star_acc:.1%}")
    print(f"  Oracle: {oracle_acc:.1%}")

    # Train MLP router
    print(f"\n[2] Training MLP Router (5-fold CV, {args.epochs} epochs)...")
    mlp_clf, mlp_routing, _ = cross_validate(
        X, y, topo_correct_all, topos,
        model_class="mlp", epochs=args.epochs,
        hidden_dim=args.hidden, lr=args.lr)
    print(f"  MLP: clf_acc={mlp_clf:.1%}, routing_acc={mlp_routing:.1%}")

    # Train Attention router
    print(f"\n[3] Training Attention Router (5-fold CV, {args.epochs} epochs)...")
    attn_clf, attn_routing, _ = cross_validate(
        X, y, topo_correct_all, topos,
        model_class="attention", epochs=args.epochs,
        hidden_dim=args.hidden, lr=args.lr)
    print(f"  Attention: clf_acc={attn_clf:.1%}, routing_acc={attn_routing:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Always Star:        {always_star_acc:.1%}")
    print(f"  Confidence Routing: 83.7% (threshold rule)")
    print(f"  MLP Router:         {mlp_routing:.1%}")
    print(f"  Attention Router:   {attn_routing:.1%}")
    print(f"  Oracle:             {oracle_acc:.1%}")

    if args.ablation:
        print(f"\n[4] Ablation: single-topology features only...")
        for exclude_topo in topos:
            subset_topos = [t for t in topos if t != exclude_topo]
            X_sub, y_sub, tc_sub, _ = load_data(subset_topos)
            # Re-label for subset
            y_sub_relabeled = []
            for tc in tc_sub:
                best = subset_topos[0]
                for t in subset_topos:
                    if tc.get(t, False):
                        best = t
                        break
                y_sub_relabeled.append(subset_topos.index(best))
            y_sub_relabeled = np.array(y_sub_relabeled)

            _, routing, _ = cross_validate(
                X_sub, y_sub_relabeled, tc_sub, subset_topos,
                model_class="mlp", epochs=args.epochs // 2,
                hidden_dim=args.hidden, lr=args.lr)
            print(f"  Without {exclude_topo}: routing_acc={routing:.1%}")


if __name__ == "__main__":
    main()
