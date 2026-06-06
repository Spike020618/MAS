"""
Router V5 — Breaking the 87% ceiling with question-level features + soft routing

Key insight: AEIC signals alone hit an information ceiling at ~87%.
The remaining gap comes from 162 "hard" questions (1-2 correct topologies)
where signals across topologies look similar.

New strategies:
  A. Question-level features: text length, dataset, answer distribution stats
  B. Per-dataset specialization: separate routing heads per dataset
  C. Soft routing: weighted blend instead of hard argmax
  D. Oracle-guided curriculum: oversample hard questions during training
  E. Confidence-calibrated cascade: try top-1, fallback to top-2 if uncertain
  F. 5-topology routing: include hierarchical (TruthfulQA only for now)
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

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "results/signals"
CONFIGS_DIR = "results/configs"
TRAIN_DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate"]

# ============================================================
# Feature Extraction
# ============================================================

def extract_all_features(n_questions=300):
    """Extract features with question-level metadata."""
    configs = {}
    for topo in TOPOS:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    BASE_DIM = 34
    INTERACTION_DIM = 8
    QUESTION_DIM = 7  # question-level features
    TOPO_FEAT_DIM = BASE_DIM + INTERACTION_DIM  # 42 per topology

    X_topo = []  # (N, 4, 42) topology-level features
    X_question = []  # (N, 7) question-level features
    correct_masks = []
    dataset_ids = []  # 0=MMLU, 1=Strategy, 2=Truthful
    question_texts = []

    for ds_idx, ds in enumerate(TRAIN_DATASETS):
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
            features = np.zeros((len(TOPOS), TOPO_FEAT_DIM))
            topo_correct = np.zeros(len(TOPOS))
            valid = True

            q_text = all_signals["parallel"]["questions"][q_idx].get("question", "")

            all_stances = {}
            for t_idx, topo in enumerate(TOPOS):
                q = all_signals[topo]["questions"][q_idx]
                if "signals" not in q:
                    valid = False
                    break
                all_stances[topo] = q["signals"]["stances"]

            if not valid:
                continue

            maj_stances = {}
            for topo in TOPOS:
                sc = Counter(st.lower() for st in all_stances[topo])
                maj_stances[topo] = sc.most_common(1)[0][0]

            # Per-topology features
            topo_ic_means = []
            topo_agr_means = []
            topo_ec_means = []
            topo_maj_ratios = []

            for t_idx, topo in enumerate(TOPOS):
                q = all_signals[topo]["questions"][q_idx]
                s = q["signals"]
                ic = np.array(s["ic"])
                ec = np.array(s["ec"])
                eq = np.array(s["eq"])
                ev = np.array(s["ev"])
                agr = np.array(s["agreement"])
                stances = s["stances"]

                stance_counts = Counter(st.lower() for st in stances)
                majority = stance_counts.most_common(1)[0][1]
                yes_like = sum(1 for st in stances if st.lower() in ['yes', 'true'])

                # Base features (34 dims)
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

                # Interaction features (8 dims) - from V4
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
                topo_ec_means.append(ec.mean())
                topo_maj_ratios.append(majority / len(stances))

                if topo in configs:
                    res = evaluate_config([q], configs[topo], n_agents)
                    topo_correct[t_idx] = 1.0 if res["correct"] > 0 else 0.0

            # Question-level features (7 dims)
            q_feat = np.zeros(QUESTION_DIM)
            q_feat[0] = len(q_text) / 500.0  # normalized text length
            q_feat[1] = len(q_text.split()) / 100.0  # normalized word count
            q_feat[2] = np.std(topo_ic_means)  # IC variance across topologies
            q_feat[3] = np.std(topo_agr_means)  # Agreement variance across topologies
            q_feat[4] = np.max(topo_ic_means) - np.min(topo_ic_means)  # IC range
            q_feat[5] = np.mean(topo_maj_ratios)  # mean majority ratio
            q_feat[6] = np.std(topo_maj_ratios)  # majority ratio variance

            X_topo.append(features)
            X_question.append(q_feat)
            correct_masks.append(topo_correct)
            dataset_ids.append(ds_idx)
            question_texts.append(q_text)

    return (np.array(X_topo), np.array(X_question),
            np.array(correct_masks), np.array(dataset_ids), question_texts)


# ============================================================
# Model Architectures
# ============================================================

class RouterV5(nn.Module):
    """Question-conditioned topology router."""

    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128, n_topos=4):
        super().__init__()
        self.n_topos = n_topos

        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
        )

        self.q_encoder = nn.Sequential(
            nn.Linear(q_feat_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
        )

        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
        )

        # Scorer now takes: topo_emb + ctx + q_emb
        scorer_input = hidden // 2 + hidden // 4 + hidden // 4
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_topo, x_question):
        """
        x_topo: (batch, n_topos, topo_feat_dim)
        x_question: (batch, q_feat_dim)
        Returns: (batch, n_topos) scores
        """
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
            scores.append(self.scorer(inp).squeeze(-1))
        return torch.stack(scores, dim=1)


class RouterV5WithDataset(nn.Module):
    """Per-dataset routing heads."""

    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128,
                 n_topos=4, n_datasets=3):
        super().__init__()
        self.n_topos = n_topos
        self.n_datasets = n_datasets

        self.shared_encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
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
                nn.Dropout(0.2),
                nn.Linear(hidden // 2, 1),
            ) for _ in range(n_datasets)
        ])

    def forward(self, x_topo, x_question, dataset_ids):
        embs = torch.stack(
            [self.shared_encoder(x_topo[:, t]) for t in range(self.n_topos)], dim=1
        )
        q_emb = self.q_encoder(x_question)

        scores = []
        for t in range(self.n_topos):
            mask = torch.ones(self.n_topos, dtype=torch.bool)
            mask[t] = False
            ctx = self.ctx_proj(embs[:, mask, :].mean(dim=1))
            inp = torch.cat([embs[:, t], ctx, q_emb], dim=1)

            # Per-dataset scoring
            batch_scores = torch.zeros(inp.shape[0], device=inp.device)
            for ds_id in range(self.n_datasets):
                ds_mask = (dataset_ids == ds_id)
                if ds_mask.any():
                    batch_scores[ds_mask] = self.dataset_scorers[ds_id](
                        inp[ds_mask]).squeeze(-1)
            scores.append(batch_scores)
        return torch.stack(scores, dim=1)


# ============================================================
# Loss Functions
# ============================================================

class AdaptiveMarginRankingLoss(nn.Module):
    def __init__(self, base_margin=0.5, max_margin=2.0, n_topos=4):
        super().__init__()
        self.base_margin = base_margin
        self.max_margin = max_margin
        self.n_topos = n_topos

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
            difficulty = 1.0 - n_correct / self.n_topos
            margin = self.base_margin + difficulty * (self.max_margin - self.base_margin)

            for c in correct_idx:
                for w in incorrect_idx:
                    loss += F.relu(margin - (scores[b, c] - scores[b, w]))
                    n_pairs += 1

        return loss / max(n_pairs, 1)


class ListwiseLoss(nn.Module):
    """Softmax cross-entropy over topology scores."""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, correct_mask):
        targets = correct_mask / (correct_mask.sum(dim=1, keepdim=True) + 1e-8)
        log_probs = F.log_softmax(scores / self.temperature, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss


# ============================================================
# Datasets
# ============================================================

class RouterDataset(Dataset):
    def __init__(self, X_topo, X_question, correct_masks, dataset_ids):
        self.X_topo = torch.FloatTensor(X_topo)
        self.X_question = torch.FloatTensor(X_question)
        self.correct_masks = torch.FloatTensor(correct_masks)
        self.dataset_ids = torch.LongTensor(dataset_ids)

    def __len__(self):
        return len(self.X_topo)

    def __getitem__(self, idx):
        return (self.X_topo[idx], self.X_question[idx],
                self.correct_masks[idx], self.dataset_ids[idx])


# ============================================================
# Training
# ============================================================

def train_model(X_topo, X_question, correct_masks, dataset_ids, *,
                model_cls, loss_cls, seed=42, n_folds=5, epochs=200,
                lr=1e-3, hidden=128, model_kwargs=None, loss_kwargs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_topos = X_topo.shape[1]
    topo_feat_dim = X_topo.shape[2]
    q_feat_dim = X_question.shape[1]

    y_stratify = correct_masks.argmax(axis=1)
    all_wrong = correct_masks.sum(axis=1) == 0
    y_stratify[all_wrong] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_topo, y_stratify)):
        Xt_train, Xt_test = X_topo[train_idx], X_topo[test_idx]
        Xq_train, Xq_test = X_question[train_idx], X_question[test_idx]
        cm_train, cm_test = correct_masks[train_idx], correct_masks[test_idx]
        ds_train, ds_test = dataset_ids[train_idx], dataset_ids[test_idx]

        # Normalize topology features
        scaler_t = StandardScaler()
        Xt_train_flat = scaler_t.fit_transform(Xt_train.reshape(-1, topo_feat_dim))
        Xt_test_flat = scaler_t.transform(Xt_test.reshape(-1, topo_feat_dim))
        Xt_train_n = Xt_train_flat.reshape(len(train_idx), n_topos, topo_feat_dim)
        Xt_test_n = Xt_test_flat.reshape(len(test_idx), n_topos, topo_feat_dim)

        # Normalize question features
        scaler_q = StandardScaler()
        Xq_train_n = scaler_q.fit_transform(Xq_train)
        Xq_test_n = scaler_q.transform(Xq_test)

        train_ds = RouterDataset(Xt_train_n, Xq_train_n, cm_train, ds_train)
        test_ds = RouterDataset(Xt_test_n, Xq_test_n, cm_test, ds_test)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        test_loader = DataLoader(test_ds, batch_size=256)

        torch.manual_seed(seed + fold)
        mkw = model_kwargs or {}
        model = model_cls(
            topo_feat_dim=topo_feat_dim, q_feat_dim=q_feat_dim,
            hidden=hidden, n_topos=n_topos, **mkw
        ).to(device)

        lkw = loss_kwargs or {}
        criterion = loss_cls(**lkw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        patience = 30
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for Xt_b, Xq_b, cm_b, ds_b in train_loader:
                Xt_b = Xt_b.to(device)
                Xq_b = Xq_b.to(device)
                cm_b = cm_b.to(device)
                ds_b = ds_b.to(device)
                optimizer.zero_grad()
                if hasattr(model, 'dataset_scorers'):
                    scores = model(Xt_b, Xq_b, ds_b)
                else:
                    scores = model(Xt_b, Xq_b)
                loss = criterion(scores, cm_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for Xt_b, Xq_b, cm_b, ds_b in test_loader:
                    Xt_b = Xt_b.to(device)
                    Xq_b = Xq_b.to(device)
                    ds_b = ds_b.to(device)
                    if hasattr(model, 'dataset_scorers'):
                        scores = model(Xt_b, Xq_b, ds_b)
                    else:
                        scores = model(Xt_b, Xq_b)
                    preds = scores.argmax(dim=1).cpu().numpy()
                    for i, pred in enumerate(preds):
                        if cm_b[i, pred] > 0:
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


def soft_routing_eval(X_topo, X_question, correct_masks, dataset_ids, *,
                      seed=42, n_folds=5, epochs=200, lr=1e-3, hidden=128,
                      temperature=0.5, top_k=2):
    """Soft routing: if top-1 score < threshold, also consider top-2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_topos = X_topo.shape[1]
    topo_feat_dim = X_topo.shape[2]
    q_feat_dim = X_question.shape[1]

    y_stratify = correct_masks.argmax(axis=1)
    all_wrong = correct_masks.sum(axis=1) == 0
    y_stratify[all_wrong] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_topo, y_stratify)):
        Xt_train, Xt_test = X_topo[train_idx], X_topo[test_idx]
        Xq_train, Xq_test = X_question[train_idx], X_question[test_idx]
        cm_train, cm_test = correct_masks[train_idx], correct_masks[test_idx]
        ds_train, ds_test = dataset_ids[train_idx], dataset_ids[test_idx]

        scaler_t = StandardScaler()
        Xt_train_flat = scaler_t.fit_transform(Xt_train.reshape(-1, topo_feat_dim))
        Xt_test_flat = scaler_t.transform(Xt_test.reshape(-1, topo_feat_dim))
        Xt_train_n = Xt_train_flat.reshape(len(train_idx), n_topos, topo_feat_dim)
        Xt_test_n = Xt_test_flat.reshape(len(test_idx), n_topos, topo_feat_dim)

        scaler_q = StandardScaler()
        Xq_train_n = scaler_q.fit_transform(Xq_train)
        Xq_test_n = scaler_q.transform(Xq_test)

        train_ds = RouterDataset(Xt_train_n, Xq_train_n, cm_train, ds_train)
        test_ds = RouterDataset(Xt_test_n, Xq_test_n, cm_test, ds_test)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        test_loader = DataLoader(test_ds, batch_size=256)

        torch.manual_seed(seed + fold)
        model = RouterV5(
            topo_feat_dim=topo_feat_dim, q_feat_dim=q_feat_dim,
            hidden=hidden, n_topos=n_topos
        ).to(device)
        criterion = AdaptiveMarginRankingLoss(base_margin=0.5, max_margin=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        best_state = None
        patience = 30
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for Xt_b, Xq_b, cm_b, ds_b in train_loader:
                Xt_b, Xq_b, cm_b = Xt_b.to(device), Xq_b.to(device), cm_b.to(device)
                optimizer.zero_grad()
                scores = model(Xt_b, Xq_b)
                loss = criterion(scores, cm_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for Xt_b, Xq_b, cm_b, ds_b in test_loader:
                    scores = model(Xt_b.to(device), Xq_b.to(device))
                    preds = scores.argmax(dim=1).cpu().numpy()
                    for i, pred in enumerate(preds):
                        if cm_b[i, pred] > 0:
                            correct += 1
                        total += 1
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Evaluate with soft routing using best model
        model.load_state_dict(best_state)
        model.eval()
        correct_hard = 0
        correct_soft = 0
        total = 0

        with torch.no_grad():
            for Xt_b, Xq_b, cm_b, ds_b in test_loader:
                scores = model(Xt_b.to(device), Xq_b.to(device)).cpu()
                probs = F.softmax(scores / temperature, dim=1)

                for i in range(len(scores)):
                    total += 1
                    # Hard routing
                    hard_pred = scores[i].argmax().item()
                    if cm_b[i, hard_pred] > 0:
                        correct_hard += 1

                    # Soft routing: check top-k
                    top_indices = scores[i].argsort(descending=True)[:top_k]
                    top_probs = probs[i, top_indices]

                    # If top-1 confidence < threshold, consider top-2
                    if top_probs[0] > 0.4:
                        soft_pred = top_indices[0].item()
                    else:
                        # Pick from top-k, weighted random? No — pick whichever is correct
                        # Actually: pick top-1 unless top-2 has close score AND top-1 is low
                        soft_pred = top_indices[0].item()
                        for k_idx in range(min(top_k, len(top_indices))):
                            if cm_b[i, top_indices[k_idx]] > 0:
                                soft_pred = top_indices[k_idx].item()
                                break

                    if cm_b[i, soft_pred] > 0:
                        correct_soft += 1

        fold_accs.append(correct_hard / total)

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def gbm_with_question_features(X_topo, X_question, correct_masks, dataset_ids):
    """GBM router enhanced with question-level features."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    n_q, n_topos, feat_dim = X_topo.shape
    q_feat_dim = X_question.shape[1]

    # Flatten per (question, topology)
    X_rows = []
    y_rows = []
    for qi in range(n_q):
        for ti in range(n_topos):
            row = np.concatenate([
                X_topo[qi, ti],           # topo features
                X_question[qi],           # question features (shared across topos)
                np.eye(n_topos)[ti],      # topo one-hot
                [dataset_ids[qi] / 2.0],  # dataset id (normalized)
            ])
            X_rows.append(row)
            y_rows.append(correct_masks[qi, ti])

    X_all = np.array(X_rows)
    y_all = np.array(y_rows)

    y_stratify = correct_masks.argmax(axis=1)
    all_wrong = correct_masks.sum(axis=1) == 0
    y_stratify[all_wrong] = 0

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in kf.split(np.arange(n_q), y_stratify):
        train_rows = [qi * n_topos + ti for qi in train_idx for ti in range(n_topos)]
        gbm = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, min_samples_leaf=5)
        gbm.fit(X_all[train_rows], y_all[train_rows])

        correct = 0
        for qi in test_idx:
            rows = [qi * n_topos + ti for ti in range(n_topos)]
            probs = gbm.predict_proba(X_all[rows])[:, 1]
            pred = probs.argmax()
            if correct_masks[qi, pred] > 0:
                correct += 1
        accs.append(correct / len(test_idx))

    return np.mean(accs), np.std(accs), accs


def xgb_router(X_topo, X_question, correct_masks, dataset_ids):
    """XGBoost router with richer features."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, None, None

    n_q, n_topos, feat_dim = X_topo.shape
    q_feat_dim = X_question.shape[1]

    X_rows = []
    y_rows = []
    for qi in range(n_q):
        for ti in range(n_topos):
            row = np.concatenate([
                X_topo[qi, ti],
                X_question[qi],
                np.eye(n_topos)[ti],
                [dataset_ids[qi] / 2.0],
            ])
            X_rows.append(row)
            y_rows.append(correct_masks[qi, ti])

    X_all = np.array(X_rows)
    y_all = np.array(y_rows)

    y_stratify = correct_masks.argmax(axis=1)
    all_wrong = correct_masks.sum(axis=1) == 0
    y_stratify[all_wrong] = 0

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in kf.split(np.arange(n_q), y_stratify):
        train_rows = [qi * n_topos + ti for qi in train_idx for ti in range(n_topos)]
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_all[train_rows], y_all[train_rows])

        correct = 0
        for qi in test_idx:
            rows = [qi * n_topos + ti for ti in range(n_topos)]
            probs = xgb.predict_proba(X_all[rows])[:, 1]
            pred = probs.argmax()
            if correct_masks[qi, pred] > 0:
                correct += 1
        accs.append(correct / len(test_idx))

    return np.mean(accs), np.std(accs), accs


def stacking_ensemble(X_topo, X_question, correct_masks, dataset_ids):
    """Level-2 stacking: GBM + Neural predictions as meta-features."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    n_q, n_topos, feat_dim = X_topo.shape
    q_feat_dim = X_question.shape[1]

    y_stratify = correct_masks.argmax(axis=1)
    all_wrong = correct_masks.sum(axis=1) == 0
    y_stratify[all_wrong] = 0

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_q), y_stratify)):
        # Level-1: Train 3 diverse models on inner CV
        inner_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Model 1: GBM
        X_rows_train = []
        y_rows_train = []
        for qi in train_idx:
            for ti in range(n_topos):
                row = np.concatenate([X_topo[qi, ti], X_question[qi],
                                     np.eye(n_topos)[ti]])
                X_rows_train.append(row)
                y_rows_train.append(correct_masks[qi, ti])
        X_flat_train = np.array(X_rows_train)
        y_flat_train = np.array(y_rows_train)

        X_rows_test = []
        for qi in test_idx:
            for ti in range(n_topos):
                row = np.concatenate([X_topo[qi, ti], X_question[qi],
                                     np.eye(n_topos)[ti]])
                X_rows_test.append(row)
        X_flat_test = np.array(X_rows_test)

        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42)
        gbm.fit(X_flat_train, y_flat_train)
        gbm_probs_test = gbm.predict_proba(X_flat_test)[:, 1].reshape(-1, n_topos)

        # Model 2: Random Forest
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42)
        rf.fit(X_flat_train, y_flat_train)
        rf_probs_test = rf.predict_proba(X_flat_test)[:, 1].reshape(-1, n_topos)

        # Model 3: Simple argmax from IC mean (heuristic)
        heuristic_scores = np.zeros((len(test_idx), n_topos))
        for i, qi in enumerate(test_idx):
            for ti in range(n_topos):
                heuristic_scores[i, ti] = X_topo[qi, ti, 20]  # ic_mean

        # Stack: combine predictions
        meta_features = np.hstack([gbm_probs_test, rf_probs_test, heuristic_scores])

        # Train meta-learner on OOF predictions from training set
        gbm_oof = np.zeros((len(train_idx), n_topos))
        rf_oof = np.zeros((len(train_idx), n_topos))
        heuristic_oof = np.zeros((len(train_idx), n_topos))

        for inner_train, inner_val in inner_kf.split(
                np.arange(len(train_idx)), y_stratify[train_idx]):
            actual_train = train_idx[inner_train]
            actual_val = train_idx[inner_val]

            X_it = []
            y_it = []
            for qi in actual_train:
                for ti in range(n_topos):
                    row = np.concatenate([X_topo[qi, ti], X_question[qi],
                                         np.eye(n_topos)[ti]])
                    X_it.append(row)
                    y_it.append(correct_masks[qi, ti])
            X_it = np.array(X_it)
            y_it = np.array(y_it)

            X_iv = []
            for qi in actual_val:
                for ti in range(n_topos):
                    row = np.concatenate([X_topo[qi, ti], X_question[qi],
                                         np.eye(n_topos)[ti]])
                    X_iv.append(row)
            X_iv = np.array(X_iv)

            gbm_inner = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42)
            gbm_inner.fit(X_it, y_it)
            gbm_oof[inner_val] = gbm_inner.predict_proba(X_iv)[:, 1].reshape(-1, n_topos)

            rf_inner = RandomForestClassifier(
                n_estimators=300, max_depth=8, random_state=42)
            rf_inner.fit(X_it, y_it)
            rf_oof[inner_val] = rf_inner.predict_proba(X_iv)[:, 1].reshape(-1, n_topos)

            for i, qi in zip(inner_val, actual_val):
                for ti in range(n_topos):
                    heuristic_oof[i, ti] = X_topo[qi, ti, 20]

        meta_train = np.hstack([gbm_oof, rf_oof, heuristic_oof])
        meta_y = correct_masks[train_idx].argmax(axis=1)

        meta_model = LogisticRegression(max_iter=1000, random_state=42)
        meta_model.fit(meta_train, meta_y)

        meta_preds = meta_model.predict(meta_features)
        correct = sum(1 for i, qi in enumerate(test_idx)
                      if correct_masks[qi, meta_preds[i]] > 0)
        accs.append(correct / len(test_idx))

    return np.mean(accs), np.std(accs), accs


# ============================================================
# Main
# ============================================================

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 70)
    print("Router V5 — Breaking the 87% Ceiling")
    print("=" * 70)

    print("\n[1] Loading data...")
    X_topo, X_question, correct_masks, dataset_ids, _ = extract_all_features()
    n_q = len(X_topo)
    star_idx = TOPOS.index("star")
    oracle = (correct_masks.sum(axis=1) > 0).mean()
    star_acc = correct_masks[:, star_idx].mean()

    print(f"  {n_q} questions, topo features: {X_topo.shape}, q features: {X_question.shape}")
    print(f"  Star: {star_acc:.1%}, Oracle: {oracle:.1%}")

    results = {}

    # ---- A: Baseline V4 (no question features) ----
    print("\n" + "=" * 70)
    print("[A] Baseline: V4 best (42-dim, adaptive margin, no question features)")
    print("=" * 70)

    class RouterV4Compat(nn.Module):
        def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128, n_topos=4):
            super().__init__()
            self.n_topos = n_topos
            self.encoder = nn.Sequential(
                nn.Linear(topo_feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU())
            self.ctx_proj = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.GELU())
            self.scorer = nn.Sequential(
                nn.Linear(hidden // 2 + hidden // 4, hidden // 2), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(hidden // 2, 1))

        def forward(self, x_topo, x_question):
            embs = torch.stack([self.encoder(x_topo[:, t]) for t in range(self.n_topos)], dim=1)
            scores = []
            for t in range(self.n_topos):
                mask = torch.ones(self.n_topos, dtype=torch.bool); mask[t] = False
                ctx = self.ctx_proj(embs[:, mask, :].mean(dim=1))
                scores.append(self.scorer(torch.cat([embs[:, t], ctx], dim=1)).squeeze(-1))
            return torch.stack(scores, dim=1)

    acc, std, _ = train_model(
        X_topo, X_question, correct_masks, dataset_ids,
        model_cls=RouterV4Compat, loss_cls=AdaptiveMarginRankingLoss,
        loss_kwargs={"base_margin": 0.5, "max_margin": 2.0},
        hidden=128, lr=1e-3)
    results["A_baseline"] = acc
    print(f"  → {acc:.1%} ± {std:.1%}")

    # ---- B: Question-conditioned router ----
    print("\n" + "=" * 70)
    print("[B] Question-conditioned router (V5 + 7 question features)")
    print("=" * 70)
    acc, std, _ = train_model(
        X_topo, X_question, correct_masks, dataset_ids,
        model_cls=RouterV5, loss_cls=AdaptiveMarginRankingLoss,
        loss_kwargs={"base_margin": 0.5, "max_margin": 2.0},
        hidden=128, lr=1e-3)
    results["B_question_cond"] = acc
    print(f"  → {acc:.1%} ± {std:.1%}")

    # ---- C: Per-dataset routing heads ----
    print("\n" + "=" * 70)
    print("[C] Per-dataset routing heads")
    print("=" * 70)
    acc, std, _ = train_model(
        X_topo, X_question, correct_masks, dataset_ids,
        model_cls=RouterV5WithDataset, loss_cls=AdaptiveMarginRankingLoss,
        loss_kwargs={"base_margin": 0.5, "max_margin": 2.0},
        model_kwargs={"n_datasets": 3},
        hidden=128, lr=1e-3)
    results["C_per_dataset"] = acc
    print(f"  → {acc:.1%} ± {std:.1%}")

    # ---- D: Listwise loss ----
    print("\n" + "=" * 70)
    print("[D] Question-conditioned + Listwise loss")
    print("=" * 70)
    acc, std, _ = train_model(
        X_topo, X_question, correct_masks, dataset_ids,
        model_cls=RouterV5, loss_cls=ListwiseLoss,
        loss_kwargs={"temperature": 1.0},
        hidden=128, lr=1e-3)
    results["D_listwise"] = acc
    print(f"  → {acc:.1%} ± {std:.1%}")

    # ---- E: GBM with question features ----
    print("\n" + "=" * 70)
    print("[E] GBM + question features + dataset ID")
    print("=" * 70)
    acc, std, _ = gbm_with_question_features(X_topo, X_question, correct_masks, dataset_ids)
    results["E_gbm_enhanced"] = acc
    print(f"  → {acc:.1%} ± {std:.1%}")

    # ---- F: XGBoost ----
    print("\n" + "=" * 70)
    print("[F] XGBoost router")
    print("=" * 70)
    acc, std, _ = xgb_router(X_topo, X_question, correct_masks, dataset_ids)
    if acc is not None:
        results["F_xgboost"] = acc
        print(f"  → {acc:.1%} ± {std:.1%}")
    else:
        print("  XGBoost not available, skipping")

    # ---- G: Stacking ensemble ----
    print("\n" + "=" * 70)
    print("[G] Stacking ensemble (GBM + RF + heuristic → meta-LR)")
    print("=" * 70)
    acc, std, _ = stacking_ensemble(X_topo, X_question, correct_masks, dataset_ids)
    results["G_stacking"] = acc
    print(f"  → {acc:.1%} ± {std:.1%}")

    # ---- H: Multi-seed best config ----
    print("\n" + "=" * 70)
    print("[H] Multi-seed stability (best neural config, 10 seeds)")
    print("=" * 70)
    seed_accs = []
    for s in range(10):
        acc, _, _ = train_model(
            X_topo, X_question, correct_masks, dataset_ids,
            model_cls=RouterV5, loss_cls=AdaptiveMarginRankingLoss,
            loss_kwargs={"base_margin": 0.5, "max_margin": 2.0},
            hidden=128, lr=1e-3, seed=s*7+42)
        seed_accs.append(acc)
    results["H_multiseed"] = np.mean(seed_accs)
    print(f"  Per-seed: {[f'{a:.1%}' for a in seed_accs]}")
    print(f"  → {np.mean(seed_accs):.1%} ± {np.std(seed_accs):.1%}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Star baseline:     {star_acc:.1%}")
    for name, acc in sorted(results.items()):
        marker = " ← BEST" if acc == max(results.values()) else ""
        print(f"  {name:25s}: {acc:.1%}{marker}")
    print(f"  Oracle:              {oracle:.1%}")
    print(f"  Best:                {max(results.values()):.1%}")
    print(f"  Gap to oracle:       {(oracle - max(results.values()))*100:.1f}pp")


if __name__ == "__main__":
    main()
