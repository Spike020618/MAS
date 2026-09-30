"""
Router V6 — Exhaustive search for best accuracy on 5-topology routing.

Tries everything:
  A. Architecture variants (deeper, wider, attention, transformer)
  B. Loss functions (ranking, listwise, focal, contrastive)
  C. Training strategies (curriculum, mixup, label smoothing)
  D. Ensemble methods (multi-seed, bagging, snapshot)
  E. Hyperparameter sweep on best combo
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

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "experiments/results/signals"
CONFIGS_DIR = "experiments/results/configs"
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]

# ============================================================
# Feature Extraction (same as 5-topo)
# ============================================================

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
            if topo == "parallel":
                path = f"{SIGNALS_DIR}/{ds}.json"
            else:
                path = f"{SIGNALS_DIR}/{ds}_{topo}.json"
            with open(path) as f:
                all_signals[topo] = json.load(f)

        n_agents = all_signals["parallel"]["meta"]["n_agents"]
        n_q = min(n_questions, len(all_signals["parallel"]["questions"]))

        for q_idx in range(n_q):
            features = np.zeros((n_topos, TOPO_DIM))
            topo_correct = np.zeros(n_topos)
            valid = True

            for t_idx, topo in enumerate(TOPOS):
                if q_idx >= len(all_signals[topo]["questions"]):
                    valid = False; break
                q = all_signals[topo]["questions"][q_idx]
                if "signals" not in q:
                    valid = False; break
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


# ============================================================
# Architectures
# ============================================================

class RouterShared(nn.Module):
    """Standard SharedEncoder + cross-topo context."""
    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128,
                 n_topos=5, n_datasets=3, dropout=0.15):
        super().__init__()
        self.n_topos = n_topos
        self.encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU())
        self.q_enc = nn.Sequential(
            nn.Linear(q_feat_dim, hidden // 4), nn.GELU())
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4), nn.GELU())
        scorer_in = hidden // 2 + hidden // 4 + hidden // 4
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden // 2), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(hidden // 2, 1))

    def forward(self, x_topo, x_q, ds_ids=None):
        embs = torch.stack([self.encoder(x_topo[:, t]) for t in range(self.n_topos)], dim=1)
        q_emb = self.q_enc(x_q)
        scores = []
        for t in range(self.n_topos):
            mask = torch.ones(self.n_topos, dtype=torch.bool); mask[t] = False
            ctx = self.ctx_proj(embs[:, mask, :].mean(dim=1))
            scores.append(self.scorer(torch.cat([embs[:, t], ctx, q_emb], dim=1)).squeeze(-1))
        return torch.stack(scores, dim=1)


class RouterDeep(nn.Module):
    """Deeper encoder (3 layers)."""
    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128,
                 n_topos=5, n_datasets=3, dropout=0.15):
        super().__init__()
        self.n_topos = n_topos
        self.encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU())
        self.q_enc = nn.Sequential(
            nn.Linear(q_feat_dim, hidden // 4), nn.GELU())
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4), nn.GELU())
        scorer_in = hidden // 2 + hidden // 4 + hidden // 4
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden // 2), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(hidden // 2, 1))

    def forward(self, x_topo, x_q, ds_ids=None):
        embs = torch.stack([self.encoder(x_topo[:, t]) for t in range(self.n_topos)], dim=1)
        q_emb = self.q_enc(x_q)
        scores = []
        for t in range(self.n_topos):
            mask = torch.ones(self.n_topos, dtype=torch.bool); mask[t] = False
            ctx = self.ctx_proj(embs[:, mask, :].mean(dim=1))
            scores.append(self.scorer(torch.cat([embs[:, t], ctx, q_emb], dim=1)).squeeze(-1))
        return torch.stack(scores, dim=1)


class RouterAttention(nn.Module):
    """Attention-based cross-topology context."""
    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128,
                 n_topos=5, n_datasets=3, dropout=0.15):
        super().__init__()
        self.n_topos = n_topos
        self.encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU())
        self.q_enc = nn.Sequential(
            nn.Linear(q_feat_dim, hidden // 4), nn.GELU())
        # Attention for cross-topo context
        self.attn_q = nn.Linear(hidden // 2, hidden // 4)
        self.attn_k = nn.Linear(hidden // 2, hidden // 4)
        self.attn_v = nn.Linear(hidden // 2, hidden // 4)
        scorer_in = hidden // 2 + hidden // 4 + hidden // 4
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden // 2), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(hidden // 2, 1))

    def forward(self, x_topo, x_q, ds_ids=None):
        embs = torch.stack([self.encoder(x_topo[:, t]) for t in range(self.n_topos)], dim=1)
        q_emb = self.q_enc(x_q)
        Q = self.attn_q(embs)
        K = self.attn_k(embs)
        V = self.attn_v(embs)
        d_k = Q.shape[-1]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
        # Mask self-attention
        mask = torch.eye(self.n_topos, device=attn_scores.device).bool().unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        ctx = torch.bmm(attn_weights, V)
        scores = []
        for t in range(self.n_topos):
            scores.append(self.scorer(
                torch.cat([embs[:, t], ctx[:, t], q_emb], dim=1)).squeeze(-1))
        return torch.stack(scores, dim=1)


class RouterPerDataset(nn.Module):
    """Per-dataset scoring heads."""
    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128,
                 n_topos=5, n_datasets=3, dropout=0.15):
        super().__init__()
        self.n_topos = n_topos
        self.n_datasets = n_datasets
        self.encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU())
        self.q_enc = nn.Sequential(
            nn.Linear(q_feat_dim, hidden // 4), nn.GELU())
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4), nn.GELU())
        scorer_in = hidden // 2 + hidden // 4 + hidden // 4
        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scorer_in, hidden // 2), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(hidden // 2, 1))
            for _ in range(n_datasets)])

    def forward(self, x_topo, x_q, ds_ids=None):
        embs = torch.stack([self.encoder(x_topo[:, t]) for t in range(self.n_topos)], dim=1)
        q_emb = self.q_enc(x_q)
        scores = []
        for t in range(self.n_topos):
            mask = torch.ones(self.n_topos, dtype=torch.bool); mask[t] = False
            ctx = self.ctx_proj(embs[:, mask, :].mean(dim=1))
            inp = torch.cat([embs[:, t], ctx, q_emb], dim=1)
            if ds_ids is not None:
                batch_scores = torch.zeros(inp.shape[0], device=inp.device)
                for d in range(self.n_datasets):
                    dm = (ds_ids == d)
                    if dm.any():
                        batch_scores[dm] = self.scorers[d](inp[dm]).squeeze(-1)
                scores.append(batch_scores)
            else:
                scores.append(self.scorers[0](inp).squeeze(-1))
        return torch.stack(scores, dim=1)


class RouterResidual(nn.Module):
    """Residual connections in encoder."""
    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128,
                 n_topos=5, n_datasets=3, dropout=0.15):
        super().__init__()
        self.n_topos = n_topos
        self.proj = nn.Linear(topo_feat_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ff1 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(hidden)
        self.ff2 = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU())
        self.q_enc = nn.Sequential(nn.Linear(q_feat_dim, hidden // 4), nn.GELU())
        self.ctx_proj = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.GELU())
        scorer_in = hidden // 2 + hidden // 4 + hidden // 4
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden // 2), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(hidden // 2, 1))

    def forward(self, x_topo, x_q, ds_ids=None):
        all_embs = []
        for t in range(self.n_topos):
            h = self.proj(x_topo[:, t])
            h = self.ln1(h + self.ff1(h))
            h = self.ff2(self.ln2(h))
            all_embs.append(h)
        embs = torch.stack(all_embs, dim=1)
        q_emb = self.q_enc(x_q)
        scores = []
        for t in range(self.n_topos):
            mask = torch.ones(self.n_topos, dtype=torch.bool); mask[t] = False
            ctx = self.ctx_proj(embs[:, mask, :].mean(dim=1))
            scores.append(self.scorer(torch.cat([embs[:, t], ctx, q_emb], dim=1)).squeeze(-1))
        return torch.stack(scores, dim=1)


# ============================================================
# Loss Functions
# ============================================================

class AdaptiveMarginLoss(nn.Module):
    def __init__(self, base_margin=0.5, max_margin=2.0, n_topos=5):
        super().__init__()
        self.bm, self.mm, self.nt = base_margin, max_margin, n_topos

    def forward(self, scores, cm):
        loss = torch.tensor(0.0, device=scores.device)
        n = 0
        for b in range(scores.shape[0]):
            ci = torch.where(cm[b] > 0)[0]
            wi = torch.where(cm[b] == 0)[0]
            if len(ci) == 0 or len(wi) == 0: continue
            d = 1.0 - len(ci) / self.nt
            m = self.bm + d * (self.mm - self.bm)
            for c in ci:
                for w in wi:
                    loss += F.relu(m - (scores[b, c] - scores[b, w]))
                    n += 1
        return loss / max(n, 1)


class FocalRankingLoss(nn.Module):
    """Focal-style: upweight samples where model is currently wrong."""
    def __init__(self, margin=1.0, gamma=2.0, n_topos=5):
        super().__init__()
        self.margin, self.gamma, self.nt = margin, gamma, n_topos

    def forward(self, scores, cm):
        loss = torch.tensor(0.0, device=scores.device)
        n = 0
        for b in range(scores.shape[0]):
            ci = torch.where(cm[b] > 0)[0]
            wi = torch.where(cm[b] == 0)[0]
            if len(ci) == 0 or len(wi) == 0: continue
            pred = scores[b].argmax()
            is_wrong = float(cm[b, pred] == 0)
            focal_weight = 1.0 + self.gamma * is_wrong
            for c in ci:
                for w in wi:
                    loss += focal_weight * F.relu(self.margin - (scores[b, c] - scores[b, w]))
                    n += 1
        return loss / max(n, 1)


class SoftCrossEntropyLoss(nn.Module):
    """Soft CE with label smoothing."""
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, scores, cm):
        n_topos = scores.shape[1]
        targets = cm / (cm.sum(dim=1, keepdim=True) + 1e-8)
        targets = (1 - self.smoothing) * targets + self.smoothing / n_topos
        log_probs = F.log_softmax(scores / self.temperature, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


class CombinedLoss(nn.Module):
    """Ranking + CE combined."""
    def __init__(self, alpha=0.7, margin=1.0, n_topos=5):
        super().__init__()
        self.alpha = alpha
        self.ranking = AdaptiveMarginLoss(n_topos=n_topos)
        self.ce = SoftCrossEntropyLoss(smoothing=0.05)

    def forward(self, scores, cm):
        return self.alpha * self.ranking(scores, cm) + (1 - self.alpha) * self.ce(scores, cm)


# ============================================================
# Dataset & Training
# ============================================================

class RouterDS(Dataset):
    def __init__(self, Xt, Xq, cm, ds):
        self.Xt = torch.FloatTensor(Xt)
        self.Xq = torch.FloatTensor(Xq)
        self.cm = torch.FloatTensor(cm)
        self.ds = torch.LongTensor(ds)
    def __len__(self): return len(self.Xt)
    def __getitem__(self, i): return self.Xt[i], self.Xq[i], self.cm[i], self.ds[i]


def train_eval(Xt, Xq, cm, ds, model_cls, loss_cls, seed=42, n_folds=5,
               epochs=200, lr=1e-3, hidden=128, loss_kw=None, model_kw=None,
               use_weighted_sampler=False, patience=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nt, tf, qf = Xt.shape[1], Xt.shape[2], Xq.shape[1]
    ys = cm.argmax(axis=1); ys[cm.sum(axis=1)==0] = 0
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (tri, tei) in enumerate(kf.split(Xt, ys)):
        sc_t = StandardScaler()
        Xt_tr = sc_t.fit_transform(Xt[tri].reshape(-1, tf)).reshape(len(tri), nt, tf)
        Xt_te = sc_t.transform(Xt[tei].reshape(-1, tf)).reshape(len(tei), nt, tf)
        sc_q = StandardScaler()
        Xq_tr = sc_q.fit_transform(Xq[tri])
        Xq_te = sc_q.transform(Xq[tei])

        train_ds = RouterDS(Xt_tr, Xq_tr, cm[tri], ds[tri])
        test_ds = RouterDS(Xt_te, Xq_te, cm[tei], ds[tei])

        if use_weighted_sampler:
            n_correct = cm[tri].sum(axis=1)
            weights = np.where(n_correct <= 2, 3.0, 1.0)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                            generator=torch.Generator().manual_seed(seed))
            train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
        else:
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                      generator=torch.Generator().manual_seed(seed))
        test_loader = DataLoader(test_ds, batch_size=256)

        torch.manual_seed(seed + fold)
        mkw = model_kw or {}
        model = model_cls(topo_feat_dim=tf, q_feat_dim=qf, hidden=hidden,
                          n_topos=nt, **mkw).to(device)
        lkw = loss_kw or {}
        criterion = loss_cls(**lkw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0; no_imp = 0
        for epoch in range(epochs):
            model.train()
            for Xt_b, Xq_b, cm_b, ds_b in train_loader:
                Xt_b, Xq_b, cm_b, ds_b = Xt_b.to(device), Xq_b.to(device), cm_b.to(device), ds_b.to(device)
                optimizer.zero_grad()
                scores = model(Xt_b, Xq_b, ds_b)
                loss = criterion(scores, cm_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            c = t = 0
            with torch.no_grad():
                for Xt_b, Xq_b, cm_b, ds_b in test_loader:
                    scores = model(Xt_b.to(device), Xq_b.to(device), ds_b.to(device))
                    preds = scores.argmax(dim=1).cpu().numpy()
                    for i, p in enumerate(preds):
                        if cm_b[i, p] > 0: c += 1
                        t += 1
            acc = c / t
            if acc > best_acc: best_acc = acc; no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience: break

        fold_accs.append(best_acc)

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def multi_seed_eval(Xt, Xq, cm, ds, model_cls, loss_cls, n_seeds=10,
                    hidden=128, lr=1e-3, loss_kw=None, model_kw=None, **kwargs):
    accs = []
    for s in range(n_seeds):
        acc, _, _ = train_eval(Xt, Xq, cm, ds, model_cls, loss_cls,
                               seed=s*7+42, hidden=hidden, lr=lr,
                               loss_kw=loss_kw, model_kw=model_kw, **kwargs)
        accs.append(acc)
    return np.mean(accs), np.std(accs), accs


def ensemble_eval(Xt, Xq, cm, ds, model_cls, loss_cls, n_models=20,
                  hidden=128, lr=1e-3, loss_kw=None, model_kw=None):
    """Train N models, ensemble their predictions by averaging scores."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nt, tf, qf = Xt.shape[1], Xt.shape[2], Xq.shape[1]
    ys = cm.argmax(axis=1); ys[cm.sum(axis=1)==0] = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (tri, tei) in enumerate(kf.split(Xt, ys)):
        sc_t = StandardScaler()
        Xt_tr = sc_t.fit_transform(Xt[tri].reshape(-1, tf)).reshape(len(tri), nt, tf)
        Xt_te = sc_t.transform(Xt[tei].reshape(-1, tf)).reshape(len(tei), nt, tf)
        sc_q = StandardScaler()
        Xq_tr = sc_q.fit_transform(Xq[tri])
        Xq_te = sc_q.transform(Xq[tei])

        train_ds = RouterDS(Xt_tr, Xq_tr, cm[tri], ds[tri])
        test_ds = RouterDS(Xt_te, Xq_te, cm[tei], ds[tei])
        test_loader = DataLoader(test_ds, batch_size=256)

        all_scores = []
        for m_idx in range(n_models):
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                      generator=torch.Generator().manual_seed(m_idx*13+7))
            torch.manual_seed(m_idx*13+7 + fold)
            mkw = model_kw or {}
            model = model_cls(topo_feat_dim=tf, q_feat_dim=qf, hidden=hidden,
                              n_topos=nt, **mkw).to(device)
            lkw = loss_kw or {}
            criterion = loss_cls(**lkw)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            best_acc = 0; best_state = None; no_imp = 0
            for epoch in range(200):
                model.train()
                for Xt_b, Xq_b, cm_b, ds_b in train_loader:
                    Xt_b, Xq_b, cm_b, ds_b = Xt_b.to(device), Xq_b.to(device), cm_b.to(device), ds_b.to(device)
                    optimizer.zero_grad()
                    scores = model(Xt_b, Xq_b, ds_b)
                    loss = criterion(scores, cm_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                model.eval()
                c = t = 0
                with torch.no_grad():
                    for Xt_b, Xq_b, cm_b, ds_b in test_loader:
                        preds = model(Xt_b.to(device), Xq_b.to(device), ds_b.to(device)).argmax(dim=1).cpu().numpy()
                        for i, p in enumerate(preds):
                            if cm_b[i, p] > 0: c += 1
                            t += 1
                acc = c / t
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= 30: break

            model.load_state_dict(best_state)
            model.eval()
            model_scores = []
            with torch.no_grad():
                for Xt_b, Xq_b, cm_b, ds_b in test_loader:
                    s = model(Xt_b.to(device), Xq_b.to(device), ds_b.to(device)).cpu()
                    model_scores.append(s)
            all_scores.append(torch.cat(model_scores, dim=0))

        # Ensemble: average scores
        avg_scores = torch.stack(all_scores).mean(dim=0)
        preds = avg_scores.argmax(dim=1).numpy()
        cm_test = cm[tei]
        correct = sum(1 for i, p in enumerate(preds) if cm_test[i, p] > 0)
        fold_accs.append(correct / len(tei))

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


# ============================================================
# Main
# ============================================================

def main():
    sys.stdout.reconfigure(line_buffering=True)
    import warnings; warnings.filterwarnings("ignore")

    print("=" * 70)
    print("Router V6 — Exhaustive 5-Topology Training")
    print("=" * 70)

    print("\n[0] Loading data...")
    Xt, Xq, cm, ds = extract_features()
    nt = Xt.shape[1]
    oracle = (cm.sum(axis=1) > 0).mean()
    star_acc = cm[:, TOPOS.index("star")].mean()
    print(f"  {len(Xt)} questions, {nt} topos, Star: {star_acc:.1%}, Oracle: {oracle:.1%}")

    results = {}

    # ---- Architectures ----
    configs = [
        ("A1_shared",       RouterShared,    AdaptiveMarginLoss, {}, {"n_topos": nt}),
        ("A2_deep",         RouterDeep,      AdaptiveMarginLoss, {}, {"n_topos": nt}),
        ("A3_attention",    RouterAttention,  AdaptiveMarginLoss, {}, {"n_topos": nt}),
        ("A4_per_dataset",  RouterPerDataset, AdaptiveMarginLoss, {"n_datasets": 3}, {"n_topos": nt}),
        ("A5_residual",     RouterResidual,   AdaptiveMarginLoss, {}, {"n_topos": nt}),
    ]

    print("\n" + "=" * 70)
    print("[1] Architecture comparison (h=128, lr=1e-3, adaptive margin)")
    print("=" * 70)
    for name, mcls, lcls, mkw, lkw in configs:
        acc, std, _ = train_eval(Xt, Xq, cm, ds, mcls, lcls,
                                 hidden=128, lr=1e-3, model_kw=mkw, loss_kw=lkw)
        results[name] = acc
        print(f"  {name:20s}: {acc:.1%} ± {std:.1%}")

    # ---- Loss functions (with best arch) ----
    print("\n" + "=" * 70)
    print("[2] Loss function comparison (best arch)")
    print("=" * 70)
    best_arch = max(configs, key=lambda x: results[x[0]])
    ba_name, ba_cls, _, ba_mkw, _ = best_arch
    print(f"  Best arch: {ba_name}")

    losses = [
        ("L1_adaptive_margin", AdaptiveMarginLoss, {"n_topos": nt}),
        ("L2_focal_ranking",   FocalRankingLoss,   {"n_topos": nt, "gamma": 2.0}),
        ("L3_focal_g3",        FocalRankingLoss,   {"n_topos": nt, "gamma": 3.0}),
        ("L4_soft_ce",         SoftCrossEntropyLoss, {"smoothing": 0.1}),
        ("L5_soft_ce_05",      SoftCrossEntropyLoss, {"smoothing": 0.05}),
        ("L6_combined",        CombinedLoss,       {"n_topos": nt}),
    ]

    for name, lcls, lkw in losses:
        acc, std, _ = train_eval(Xt, Xq, cm, ds, ba_cls, lcls,
                                 hidden=128, lr=1e-3, model_kw=ba_mkw, loss_kw=lkw)
        results[name] = acc
        print(f"  {name:20s}: {acc:.1%} ± {std:.1%}")

    # ---- Training strategies ----
    print("\n" + "=" * 70)
    print("[3] Training strategies")
    print("=" * 70)

    best_loss = max(losses, key=lambda x: results[x[0]])
    bl_name, bl_cls, bl_kw = best_loss
    print(f"  Best loss: {bl_name}")

    # Weighted sampler
    acc, std, _ = train_eval(Xt, Xq, cm, ds, ba_cls, bl_cls,
                             hidden=128, lr=1e-3, model_kw=ba_mkw, loss_kw=bl_kw,
                             use_weighted_sampler=True)
    results["S1_weighted_sampler"] = acc
    print(f"  S1_weighted_sampler: {acc:.1%} ± {std:.1%}")

    # More patience
    acc, std, _ = train_eval(Xt, Xq, cm, ds, ba_cls, bl_cls,
                             hidden=128, lr=1e-3, model_kw=ba_mkw, loss_kw=bl_kw,
                             epochs=400, patience=50)
    results["S2_long_train"] = acc
    print(f"  S2_long_train:      {acc:.1%} ± {std:.1%}")

    # ---- Hyperparameter sweep on best combo ----
    print("\n" + "=" * 70)
    print("[4] Hyperparameter sweep (best arch + loss)")
    print("=" * 70)

    best_hp_acc = 0
    best_hp_cfg = ""
    for h in [128, 192, 256]:
        for lr in [3e-4, 5e-4, 1e-3, 2e-3]:
            for drop in [0.1, 0.15, 0.2]:
                mkw2 = {**ba_mkw, "dropout": drop}
                acc, std, _ = train_eval(Xt, Xq, cm, ds, ba_cls, bl_cls,
                                         hidden=h, lr=lr, model_kw=mkw2, loss_kw=bl_kw)
                if acc > best_hp_acc:
                    best_hp_acc = acc
                    best_hp_cfg = f"h={h},lr={lr},drop={drop}"
                    print(f"  NEW BEST: {acc:.1%} ± {std:.1%} ({best_hp_cfg})")
    results["H_best_hp"] = best_hp_acc

    # ---- Multi-seed stability ----
    print("\n" + "=" * 70)
    print("[5] Multi-seed stability (20 seeds, best config)")
    print("=" * 70)
    # Parse best config
    h_best = int(best_hp_cfg.split("h=")[1].split(",")[0])
    lr_best = float(best_hp_cfg.split("lr=")[1].split(",")[0])
    drop_best = float(best_hp_cfg.split("drop=")[1])
    mkw_best = {**ba_mkw, "dropout": drop_best}

    acc, std, seed_accs = multi_seed_eval(
        Xt, Xq, cm, ds, ba_cls, bl_cls, n_seeds=20,
        hidden=h_best, lr=lr_best, model_kw=mkw_best, loss_kw=bl_kw)
    results["M_20seed"] = acc
    print(f"  20-seed: {acc:.1%} ± {std:.1%}")
    print(f"  Range: {min(seed_accs):.1%} - {max(seed_accs):.1%}")

    # ---- Ensemble ----
    print("\n" + "=" * 70)
    print("[6] Ensemble (10 models, score averaging)")
    print("=" * 70)
    acc, std, _ = ensemble_eval(Xt, Xq, cm, ds, ba_cls, bl_cls, n_models=10,
                                hidden=h_best, lr=lr_best,
                                model_kw=mkw_best, loss_kw=bl_kw)
    results["E_ensemble_10"] = acc
    print(f"  Ensemble (10 models): {acc:.1%} ± {std:.1%}")

    # ---- 20 model ensemble ----
    print("\n" + "=" * 70)
    print("[7] Large ensemble (20 models)")
    print("=" * 70)
    acc, std, _ = ensemble_eval(Xt, Xq, cm, ds, ba_cls, bl_cls, n_models=20,
                                hidden=h_best, lr=lr_best,
                                model_kw=mkw_best, loss_kw=bl_kw)
    results["E_ensemble_20"] = acc
    print(f"  Ensemble (20 models): {acc:.1%} ± {std:.1%}")

    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Star baseline:     {star_acc:.1%}")
    for name in sorted(results, key=lambda x: results[x]):
        marker = " ← BEST" if results[name] == max(results.values()) else ""
        print(f"  {name:22s}: {results[name]:.1%}{marker}")
    print(f"  Oracle:              {oracle:.1%}")
    print(f"  Best:                {max(results.values()):.1%}")
    print(f"  Gap to oracle:       {(oracle - max(results.values()))*100:.1f}pp")


if __name__ == "__main__":
    main()
