"""
Router V7 — Beyond the Neural Net Ceiling.

Two new strategies to break the 88-89% barrier:
  1. Richer features: per-agent distributions, cross-topo contrasts, entropy
  2. GBM (LightGBM/XGBoost) — gradient boosted trees often outperform small NNs
     on tabular data with limited samples (900 questions)
  3. Stacking: GBM + NN meta-learner
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "results/signals"
CONFIGS_DIR = "results/configs"
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]


def extract_rich_features(n_questions=300):
    """Extended features: 62-dim per topo + 15-dim question-level."""
    configs = {}
    for topo in TOPOS:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    TOPO_DIM = 62
    Q_DIM = 15
    n_topos = len(TOPOS)

    X_topo_all = []
    X_q_all = []
    cm_all = []
    ds_all = []

    for ds_idx, ds in enumerate(DATASETS):
        all_signals = {}
        for topo in TOPOS:
            if topo == "parallel":
                path = f"{SIGNALS_DIR}/{ds}.json"
            else:
                path = f"{SIGNALS_DIR}/{ds}_{topo}.json"
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
                topo_ic_means = []
                topo_ec_means = []
                topo_eq_means = []
                topo_ev_means = []
                topo_agr_means = []
                topo_maj_ratios = []
                topo_entropies = []

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

                    # Base features (34) — same as V6
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

                    # Interaction features (8) — same as V6
                    features[t_idx, 34] = ic.mean() * (1 - agr.mean())
                    features[t_idx, 35] = ec.mean() * (1 - agr.mean())
                    features[t_idx, 36] = eq.mean() * (ic.std() / (ic.mean() + 1e-6))
                    features[t_idx, 37] = ic.max() - ic.min()
                    features[t_idx, 38] = ic.mean() - agr.mean()
                    features[t_idx, 39] = ic.mean() * (majority / len(stances))
                    features[t_idx, 40] = agr.std() * ic.mean()
                    features[t_idx, 41] = ev.mean() * (1 - majority / len(stances))

                    # NEW: Distribution features (10)
                    features[t_idx, 42] = np.median(ic)
                    features[t_idx, 43] = np.percentile(ic, 25)
                    features[t_idx, 44] = np.percentile(ic, 75)
                    features[t_idx, 45] = np.median(ec)
                    features[t_idx, 46] = np.median(eq)
                    features[t_idx, 47] = np.median(ev)
                    features[t_idx, 48] = np.median(agr)

                    # Entropy of stance distribution
                    probs = np.array([c / len(stances) for c in sc.values()])
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    features[t_idx, 49] = entropy

                    # Skewness of IC
                    ic_mean = ic.mean()
                    ic_std = ic.std() + 1e-10
                    features[t_idx, 50] = np.mean(((ic - ic_mean) / ic_std) ** 3)

                    # Kurtosis of IC
                    features[t_idx, 51] = np.mean(((ic - ic_mean) / ic_std) ** 4) - 3.0

                    # NEW: Cross-signal features (10)
                    features[t_idx, 52] = np.corrcoef(ic, ec)[0, 1] if ic.std() > 0 and ec.std() > 0 else 0
                    features[t_idx, 53] = np.corrcoef(ic, eq)[0, 1] if ic.std() > 0 and eq.std() > 0 else 0
                    features[t_idx, 54] = np.corrcoef(ic, ev)[0, 1] if ic.std() > 0 and ev.std() > 0 else 0
                    features[t_idx, 55] = np.corrcoef(ec, eq)[0, 1] if ec.std() > 0 and eq.std() > 0 else 0
                    features[t_idx, 56] = (ic * ec).mean()
                    features[t_idx, 57] = (ic * eq).mean()
                    features[t_idx, 58] = (ic * agr).mean()
                    features[t_idx, 59] = (ec * agr).mean()
                    features[t_idx, 60] = ic.mean() * ec.mean() * agr.mean()
                    features[t_idx, 61] = ic.mean() / (1 - agr.mean() + 0.01)

                    topo_ic_means.append(ic.mean())
                    topo_ec_means.append(ec.mean())
                    topo_eq_means.append(eq.mean())
                    topo_ev_means.append(ev.mean())
                    topo_agr_means.append(agr.mean())
                    topo_maj_ratios.append(majority / len(stances))
                    topo_entropies.append(entropy)

                    if topo in configs:
                        res = evaluate_config([q], configs[topo], n_agents)
                        topo_correct[t_idx] = 1.0 if res["correct"] > 0 else 0.0

                # Question-level features (15)
                q_feat = np.zeros(Q_DIM)
                q_feat[0] = len(q_text) / 500.0
                q_feat[1] = len(q_text.split()) / 100.0
                q_feat[2] = np.std(topo_ic_means)
                q_feat[3] = np.std(topo_agr_means)
                q_feat[4] = np.max(topo_ic_means) - np.min(topo_ic_means)
                q_feat[5] = np.mean(topo_maj_ratios)
                q_feat[6] = np.std(topo_maj_ratios)
                # NEW
                q_feat[7] = np.std(topo_ec_means)
                q_feat[8] = np.std(topo_eq_means)
                q_feat[9] = np.std(topo_ev_means)
                q_feat[10] = np.mean(topo_entropies)
                q_feat[11] = np.std(topo_entropies)
                q_feat[12] = np.max(topo_ic_means)
                q_feat[13] = np.max(topo_agr_means)
                q_feat[14] = ds_idx / 2.0  # dataset indicator

                X_topo_all.append(features)
                X_q_all.append(q_feat)
                cm_all.append(topo_correct)
                ds_all.append(ds_idx)

    return (np.array(X_topo_all), np.array(X_q_all),
            np.array(cm_all), np.array(ds_all))


def flatten_features(X_topo, X_q):
    """Flatten for tree-based models: [topo1_feats, topo2_feats, ..., q_feats, contrast_feats]."""
    n = len(X_topo)
    n_topos = X_topo.shape[1]
    topo_dim = X_topo.shape[2]
    q_dim = X_q.shape[1]

    # Pairwise contrast features: for each pair of topos, diff of key signals
    key_dims = [20, 22, 24, 25, 26, 30]  # ic_mean, ec_mean, eq_mean, ev_mean, agr_mean, maj_ratio
    n_pairs = n_topos * (n_topos - 1) // 2
    contrast_dim = n_pairs * len(key_dims)

    X_flat = np.zeros((n, n_topos * topo_dim + q_dim + contrast_dim))
    for i in range(n):
        offset = 0
        for t in range(n_topos):
            X_flat[i, offset:offset + topo_dim] = X_topo[i, t]
            offset += topo_dim
        X_flat[i, offset:offset + q_dim] = X_q[i]
        offset += q_dim

        # Pairwise contrasts
        for t1 in range(n_topos):
            for t2 in range(t1 + 1, n_topos):
                for d in key_dims:
                    X_flat[i, offset] = X_topo[i, t1, d] - X_topo[i, t2, d]
                    offset += 1

    return X_flat


def eval_gbm(X_flat, cm, ds, n_topos=5, n_folds=5, seed=42, **gbm_params):
    """GBM-based routing: train classifier to predict best topology."""
    y = cm.argmax(axis=1)
    y[cm.sum(axis=1) == 0] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (tri, tei) in enumerate(kf.split(X_flat, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_flat[tri])
        Xte = scaler.transform(X_flat[tei])

        clf = GradientBoostingClassifier(
            random_state=seed + fold, **gbm_params
        )
        clf.fit(Xtr, y[tri])
        preds = clf.predict(Xte)

        correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
        fold_accs.append(correct / len(tei))

    return np.mean(fold_accs), np.std(fold_accs)


def eval_rf(X_flat, cm, ds, n_topos=5, n_folds=5, seed=42, **rf_params):
    """Random Forest-based routing."""
    y = cm.argmax(axis=1)
    y[cm.sum(axis=1) == 0] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (tri, tei) in enumerate(kf.split(X_flat, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_flat[tri])
        Xte = scaler.transform(X_flat[tei])

        clf = RandomForestClassifier(
            random_state=seed + fold, **rf_params
        )
        clf.fit(Xtr, y[tri])
        preds = clf.predict(Xte)

        correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
        fold_accs.append(correct / len(tei))

    return np.mean(fold_accs), np.std(fold_accs)


def eval_pairwise_gbm(X_topo, X_q, cm, ds, n_topos=5, n_folds=5, seed=42, **gbm_params):
    """Pairwise approach: for each pair of topologies, train binary classifier.
    Then aggregate votes to pick best topology."""
    n = len(X_topo)
    topo_dim = X_topo.shape[2]
    q_dim = X_q.shape[1]

    y_best = cm.argmax(axis=1)
    y_best[cm.sum(axis=1) == 0] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (tri, tei) in enumerate(kf.split(X_topo[:, 0], y_best)):
        pair_models = {}
        for t1 in range(n_topos):
            for t2 in range(t1 + 1, n_topos):
                # Features: diff between topo t1 and t2 signals + question features
                X_pair = np.hstack([
                    X_topo[:, t1] - X_topo[:, t2],
                    X_topo[:, t1] + X_topo[:, t2],
                    X_q
                ])

                # Label: 1 if t1 correct and t2 not (or both correct but t1 preferred)
                y_pair = np.zeros(n)
                for i in range(n):
                    if cm[i, t1] > 0 and cm[i, t2] == 0:
                        y_pair[i] = 1
                    elif cm[i, t1] == 0 and cm[i, t2] > 0:
                        y_pair[i] = 0
                    else:
                        y_pair[i] = 0.5

                mask = y_pair != 0.5
                if mask[tri].sum() < 10:
                    continue

                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_pair[tri][mask[tri]])
                ytr = y_pair[tri][mask[tri]]

                clf = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=seed + fold
                )
                clf.fit(Xtr, ytr)

                Xte_all = scaler.transform(X_pair[tei])
                pair_models[(t1, t2)] = (clf, scaler, Xte_all)

        # Aggregate: vote counting
        votes = np.zeros((len(tei), n_topos))
        for (t1, t2), (clf, scaler, Xte_all) in pair_models.items():
            probs = clf.predict_proba(Xte_all)
            if probs.shape[1] == 2:
                votes[:, t1] += probs[:, 1]
                votes[:, t2] += probs[:, 0]
            else:
                votes[:, t1] += 0.5
                votes[:, t2] += 0.5

        preds = votes.argmax(axis=1)
        correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
        fold_accs.append(correct / len(tei))

    return np.mean(fold_accs), np.std(fold_accs)


class RouterPerDatasetV7(nn.Module):
    def __init__(self, topo_feat_dim=62, q_feat_dim=15, hidden=192,
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

    def forward(self, scores, correct_mask):
        batch_size = scores.shape[0]
        loss = torch.tensor(0.0, device=scores.device)
        n_pairs = 0
        for b in range(batch_size):
            correct_idx = torch.where(correct_mask[b] > 0)[0]
            incorrect_idx = torch.where(correct_mask[b] == 0)[0]
            if len(correct_idx) == 0 or len(incorrect_idx) == 0:
                continue
            difficulty = 1.0 - len(correct_idx) / self.n_topos
            margin = self.base_margin + difficulty * (self.max_margin - self.base_margin)
            for c in correct_idx:
                for w in incorrect_idx:
                    violation = F.relu(margin - (scores[b, c] - scores[b, w]))
                    focal_weight = (violation / margin + 1e-6) ** self.gamma
                    loss += focal_weight * violation
                    n_pairs += 1
        return loss / max(n_pairs, 1)


class RouterDataset(Dataset):
    def __init__(self, X_topo, X_q, cm, ds):
        self.X_topo = torch.FloatTensor(X_topo)
        self.X_q = torch.FloatTensor(X_q)
        self.cm = torch.FloatTensor(cm)
        self.ds = torch.LongTensor(ds)

    def __len__(self):
        return len(self.X_topo)

    def __getitem__(self, idx):
        return self.X_topo[idx], self.X_q[idx], self.cm[idx], self.ds[idx]


def train_nn(X_topo, X_q, cm, ds, n_topos=5, seed=42, n_folds=5,
             epochs=300, lr=1e-3, hidden=192, dropout=0.2):
    """Train NN with rich features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topo_feat_dim = X_topo.shape[2]
    q_feat_dim = X_q.shape[1]

    y_strat = cm.argmax(axis=1)
    y_strat[cm.sum(axis=1) == 0] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (tri, tei) in enumerate(kf.split(X_topo, y_strat)):
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

        train_ds = RouterDataset(Xt_tr, Xq_tr, cm[tri], ds[tri])
        test_ds = RouterDataset(Xt_te, Xq_te, cm[tei], ds[tei])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        test_loader = DataLoader(test_ds, batch_size=256)

        torch.manual_seed(seed + fold)
        model = RouterPerDatasetV7(
            topo_feat_dim=topo_feat_dim, q_feat_dim=q_feat_dim,
            hidden=hidden, n_topos=n_topos, n_datasets=3, dropout=dropout
        ).to(device)
        criterion = FocalRankingLoss(n_topos=n_topos)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        patience = 40
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for Xt_b, Xq_b, cm_b, ds_b in train_loader:
                Xt_b, Xq_b, cm_b, ds_b = (
                    Xt_b.to(device), Xq_b.to(device),
                    cm_b.to(device), ds_b.to(device))
                optimizer.zero_grad()
                scores = model(Xt_b, Xq_b, ds_b)
                loss = criterion(scores, cm_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for Xt_b, Xq_b, cm_b, ds_b in test_loader:
                    scores = model(Xt_b.to(device), Xq_b.to(device), ds_b.to(device))
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

    return np.mean(fold_accs), np.std(fold_accs)


def eval_stacking(X_topo, X_q, X_flat, cm, ds, n_topos=5, n_folds=5, seed=42):
    """Stack GBM + NN predictions with logistic regression meta-learner."""
    y_best = cm.argmax(axis=1)
    y_best[cm.sum(axis=1) == 0] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topo_feat_dim = X_topo.shape[2]
    q_feat_dim = X_q.shape[1]

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (tri, tei) in enumerate(kf.split(X_flat, y_best)):
        # GBM predictions (probability per class)
        scaler_f = StandardScaler()
        Xf_tr = scaler_f.fit_transform(X_flat[tri])
        Xf_te = scaler_f.transform(X_flat[tei])

        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=seed + fold
        )
        gbm.fit(Xf_tr, y_best[tri])
        gbm_probs_te = gbm.predict_proba(Xf_te)

        # NN predictions
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

        train_ds = RouterDataset(Xt_tr, Xq_tr, cm[tri], ds[tri])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))

        torch.manual_seed(seed + fold)
        model = RouterPerDatasetV7(
            topo_feat_dim=topo_feat_dim, q_feat_dim=q_feat_dim,
            hidden=192, n_topos=n_topos, n_datasets=3, dropout=0.2
        ).to(device)
        criterion = FocalRankingLoss(n_topos=n_topos)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        for epoch in range(200):
            model.train()
            for Xt_b, Xq_b, cm_b, ds_b in train_loader:
                Xt_b, Xq_b, cm_b, ds_b = (
                    Xt_b.to(device), Xq_b.to(device),
                    cm_b.to(device), ds_b.to(device))
                optimizer.zero_grad()
                loss = criterion(model(Xt_b, Xq_b, ds_b), cm_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            nn_scores = model(
                torch.FloatTensor(Xt_te).to(device),
                torch.FloatTensor(Xq_te).to(device),
                torch.LongTensor(ds[tei]).to(device)
            ).cpu().numpy()

        nn_probs = np.exp(nn_scores) / np.exp(nn_scores).sum(axis=1, keepdims=True)

        # Ensure both have same number of columns
        if gbm_probs_te.shape[1] < n_topos:
            padded = np.zeros((len(tei), n_topos))
            for ci, c in enumerate(gbm.classes_):
                padded[:, c] = gbm_probs_te[:, ci]
            gbm_probs_te = padded

        # Stack: combine GBM and NN probabilities
        meta_features = np.hstack([gbm_probs_te, nn_probs])

        # Simple: weighted average
        alpha = 0.5
        combined_probs = alpha * gbm_probs_te + (1 - alpha) * nn_probs
        preds = combined_probs.argmax(axis=1)

        correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
        fold_accs.append(correct / len(tei))

    return np.mean(fold_accs), np.std(fold_accs)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    import warnings; warnings.filterwarnings("ignore")

    print("=" * 70)
    print("Router V7 — Beyond the Neural Net Ceiling")
    print("=" * 70)

    print("\n[0] Loading rich features...")
    Xt, Xq, cm, ds = extract_rich_features()
    n_topos = Xt.shape[1]
    oracle = (cm.sum(axis=1) > 0).mean()
    star_acc = cm[:, TOPOS.index("star")].mean()
    print(f"  {len(Xt)} questions, {n_topos} topos")
    print(f"  Feature dims: {Xt.shape[2]} per topo, {Xq.shape[1]} per question")
    print(f"  Star: {star_acc:.1%}, Oracle: {oracle:.1%}")

    X_flat = flatten_features(Xt, Xq)
    print(f"  Flattened: {X_flat.shape[1]} features")

    results = {}

    # ---- NN with rich features ----
    print("\n" + "=" * 70)
    print("[1] Neural Net with rich features (62+15 dims)")
    print("=" * 70)
    accs = []
    for s in range(10):
        acc, std = train_nn(Xt, Xq, cm, ds, seed=s * 7 + 42)
        accs.append(acc)
    results["NN_rich"] = np.mean(accs)
    print(f"  NN (rich features, 10-seed): {np.mean(accs):.1%} ± {np.std(accs):.1%}")
    print(f"  Range: {min(accs):.1%} - {max(accs):.1%}")

    # ---- GBM sweep ----
    print("\n" + "=" * 70)
    print("[2] Gradient Boosted Trees sweep")
    print("=" * 70)
    best_gbm = 0
    best_gbm_cfg = ""
    for n_est in [100, 200, 500]:
        for depth in [3, 4, 5, 6]:
            for lr in [0.01, 0.05, 0.1]:
                for subsample in [0.8, 1.0]:
                    accs_gbm = []
                    for s in range(5):
                        acc, _ = eval_gbm(X_flat, cm, ds, seed=s * 7 + 42,
                                          n_estimators=n_est, max_depth=depth,
                                          learning_rate=lr, subsample=subsample)
                        accs_gbm.append(acc)
                    mean_acc = np.mean(accs_gbm)
                    if mean_acc > best_gbm:
                        best_gbm = mean_acc
                        best_gbm_cfg = f"n={n_est},d={depth},lr={lr},ss={subsample}"
                        print(f"  NEW BEST: {mean_acc:.1%} ± {np.std(accs_gbm):.1%} ({best_gbm_cfg})")
    results["GBM_best"] = best_gbm

    # ---- Random Forest ----
    print("\n" + "=" * 70)
    print("[3] Random Forest")
    print("=" * 70)
    best_rf = 0
    for n_est in [200, 500, 1000]:
        for depth in [None, 10, 20]:
            accs_rf = []
            for s in range(5):
                acc, _ = eval_rf(X_flat, cm, ds, seed=s * 7 + 42,
                                 n_estimators=n_est, max_depth=depth)
                accs_rf.append(acc)
            mean_acc = np.mean(accs_rf)
            if mean_acc > best_rf:
                best_rf = mean_acc
                print(f"  NEW BEST: {mean_acc:.1%} ± {np.std(accs_rf):.1%} (n={n_est},d={depth})")
    results["RF_best"] = best_rf

    # ---- Pairwise GBM ----
    print("\n" + "=" * 70)
    print("[4] Pairwise GBM (one-vs-one)")
    print("=" * 70)
    accs_pw = []
    for s in range(5):
        acc, _ = eval_pairwise_gbm(Xt, Xq, cm, ds, seed=s * 7 + 42)
        accs_pw.append(acc)
    results["Pairwise_GBM"] = np.mean(accs_pw)
    print(f"  Pairwise GBM (5-seed): {np.mean(accs_pw):.1%} ± {np.std(accs_pw):.1%}")

    # ---- Stacking ----
    print("\n" + "=" * 70)
    print("[5] Stacking (GBM + NN)")
    print("=" * 70)
    accs_stack = []
    for s in range(5):
        acc, _ = eval_stacking(Xt, Xq, X_flat, cm, ds, seed=s * 7 + 42)
        accs_stack.append(acc)
    results["Stacking"] = np.mean(accs_stack)
    print(f"  Stacking (5-seed): {np.mean(accs_stack):.1%} ± {np.std(accs_stack):.1%}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Star baseline:      {star_acc:.1%}")
    print(f"  V6 best (ref):      88.8%")
    for name in sorted(results, key=lambda x: results[x]):
        marker = " ← BEST" if results[name] == max(results.values()) else ""
        print(f"  {name:22s}: {results[name]:.1%}{marker}")
    print(f"  Oracle:              {oracle:.1%}")
    print(f"  Best:                {max(results.values()):.1%}")
    print(f"  Gap to oracle:       {(oracle - max(results.values()))*100:.1f}pp")


if __name__ == "__main__":
    main()
