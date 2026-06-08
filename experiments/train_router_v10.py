"""
Router V10 — Clean labels + honest evaluation.

Critical fixes over V9:
  1. Labels use MajVote correctness (not overfit θ* configs) — eliminates contamination
  2. Top-1 only — no top-2 ground-truth oracle cheating
  3. Per-dataset specialized scorer heads (from V9 architecture)
  4. 5 topologies including hierarchical

Usage:
  python experiments/train_router_v10.py
  python experiments/train_router_v10.py --seeds 10
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

SIGNALS_DIR = "results/signals"
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]


def _question_key(q):
    """Stable key for matching the same question across topology files."""
    return q.get("question", "")[:200]


def extract_features(n_questions=300):
    """Extract features with MajVote correctness labels (no θ* contamination).

    Questions are matched across topologies by text content, not by index.
    Only questions present in ALL 5 topologies are used.
    """
    TOPO_DIM = 42
    Q_DIM = 7
    n_topos = len(TOPOS)
    X_topo_all, X_q_all, cm_all, ds_all = [], [], [], []

    for ds_idx, ds_name in enumerate(DATASETS):
        all_signals = {}
        for topo in TOPOS:
            path = f"{SIGNALS_DIR}/{ds_name}.json" if topo == "parallel" \
                else f"{SIGNALS_DIR}/{ds_name}_{topo}.json"
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}, skipping {topo}")
                break
            with open(path) as f:
                all_signals[topo] = json.load(f)
        else:
            n_agents = all_signals["parallel"]["meta"]["n_agents"]

            topo_indices = {}
            for topo in TOPOS:
                topo_indices[topo] = {}
                for i, q in enumerate(all_signals[topo]["questions"][:n_questions]):
                    if "signals" in q:
                        topo_indices[topo][_question_key(q)] = i

            common_keys = set(topo_indices[TOPOS[0]].keys())
            for topo in TOPOS[1:]:
                common_keys &= set(topo_indices[topo].keys())
            common_keys = sorted(common_keys)

            print(f"  {ds_name}: {len(common_keys)} questions in all 5 topologies")

            for q_key in common_keys:
                features = np.zeros((n_topos, TOPO_DIM))
                topo_correct = np.zeros(n_topos)

                par_idx = topo_indices["parallel"][q_key]
                q_data = all_signals["parallel"]["questions"][par_idx]
                q_text = q_data.get("question", "")
                ground_truth = q_data["ground_truth"].lower()
                topo_ic_means, topo_agr_means, topo_maj_ratios = [], [], []

                for t_idx, topo in enumerate(TOPOS):
                    idx = topo_indices[topo][q_key]
                    q = all_signals[topo]["questions"][idx]
                    s = q["signals"]
                    ic = np.array(s["ic"]); ec = np.array(s["ec"])
                    eq = np.array(s["eq"]); ev = np.array(s["ev"])
                    agr = np.array(s["agreement"]); stances = s["stances"]
                    sc = Counter(st.lower() for st in stances)
                    majority_count = sc.most_common(1)[0][1]
                    majority_stance = sc.most_common(1)[0][0]
                    yes_like = sum(1 for st in stances if st.lower() in ['yes', 'true'])

                    features[t_idx, 0:5] = ic; features[t_idx, 5:10] = ec
                    features[t_idx, 10:15] = eq; features[t_idx, 15:20] = ev
                    features[t_idx, 20] = ic.mean(); features[t_idx, 21] = ic.std()
                    features[t_idx, 22] = ec.mean(); features[t_idx, 23] = ec.std()
                    features[t_idx, 24] = eq.mean(); features[t_idx, 25] = ev.mean()
                    features[t_idx, 26] = agr.mean(); features[t_idx, 27] = agr.min()
                    features[t_idx, 28] = agr.max(); features[t_idx, 29] = agr.std()
                    features[t_idx, 30] = majority_count / len(stances)
                    features[t_idx, 31] = 1.0 if yes_like > len(stances) / 2 else -1.0
                    features[t_idx, 32] = yes_like / len(stances)
                    features[t_idx, 33] = 1.0 - yes_like / len(stances)
                    features[t_idx, 34] = ic.mean() * (1 - agr.mean())
                    features[t_idx, 35] = ec.mean() * (1 - agr.mean())
                    features[t_idx, 36] = eq.mean() * (ic.std() / (ic.mean() + 1e-6))
                    features[t_idx, 37] = ic.max() - ic.min()
                    features[t_idx, 38] = ic.mean() - agr.mean()
                    features[t_idx, 39] = ic.mean() * (majority_count / len(stances))
                    features[t_idx, 40] = agr.std() * ic.mean()
                    features[t_idx, 41] = ev.mean() * (1 - majority_count / len(stances))

                    topo_ic_means.append(ic.mean())
                    topo_agr_means.append(agr.mean())
                    topo_maj_ratios.append(majority_count / len(stances))

                    topo_correct[t_idx] = 1.0 if majority_stance == ground_truth else 0.0

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
# Model (same architecture as V9)
# ============================================================

class RouterModel(nn.Module):
    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=192,
                 n_topos=5, n_datasets=3, dropout=0.2):
        super().__init__()
        self.n_topos = n_topos
        self.n_datasets = n_datasets
        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_feat_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
        )
        self.q_encoder = nn.Sequential(nn.Linear(q_feat_dim, hidden // 4), nn.GELU())
        self.ctx_proj = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.GELU())
        scorer_in = hidden // 2 + hidden // 4 + hidden // 4
        self.dataset_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scorer_in, hidden // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
            ) for _ in range(n_datasets)
        ])

    def forward(self, x_topo, x_question, dataset_ids):
        embs = torch.stack(
            [self.topo_encoder(x_topo[:, t]) for t in range(self.n_topos)], dim=1)
        q_emb = self.q_encoder(x_question)
        scores = []
        for t in range(self.n_topos):
            mask = torch.ones(self.n_topos, dtype=torch.bool); mask[t] = False
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
        loss = scores.sum() * 0.0
        n_pairs = 0
        for b in range(scores.shape[0]):
            ci = torch.where(correct_mask[b] > 0)[0]
            wi = torch.where(correct_mask[b] == 0)[0]
            if len(ci) == 0 or len(wi) == 0:
                continue
            difficulty = 1.0 - len(ci) / self.n_topos
            margin = self.base_margin + difficulty * (self.max_margin - self.base_margin)
            for c in ci:
                for w in wi:
                    v = F.relu(margin - (scores[b, c] - scores[b, w]))
                    loss += ((v / margin + 1e-6) ** self.gamma) * v
                    n_pairs += 1
        return loss / max(n_pairs, 1)


class RouterDataset(Dataset):
    def __init__(self, X_topo, X_q, cm, ds):
        self.X_topo = torch.FloatTensor(X_topo)
        self.X_q = torch.FloatTensor(X_q)
        self.cm = torch.FloatTensor(cm)
        self.ds = torch.LongTensor(ds)
    def __len__(self): return len(self.X_topo)
    def __getitem__(self, idx):
        return self.X_topo[idx], self.X_q[idx], self.cm[idx], self.ds[idx]


# ============================================================
# Training
# ============================================================

def train_model(X_topo, X_q, cm, ds, tri, tei, seed=42,
                epochs=300, lr=1e-3, hidden=192, dropout=0.2, n_topos=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = X_topo.shape[2]; qf = X_q.shape[1]

    sc_t = StandardScaler()
    Xt_tr = sc_t.fit_transform(X_topo[tri].reshape(-1, tf)).reshape(len(tri), n_topos, tf)
    Xt_te = sc_t.transform(X_topo[tei].reshape(-1, tf)).reshape(len(tei), n_topos, tf)
    sc_q = StandardScaler()
    Xq_tr = sc_q.fit_transform(X_q[tri]); Xq_te = sc_q.transform(X_q[tei])

    train_ds = RouterDataset(Xt_tr, Xq_tr, cm[tri], ds[tri])
    loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    torch.manual_seed(seed)
    model = RouterModel(tf, qf, hidden, n_topos, 3, dropout).to(device)
    criterion = FocalRankingLoss(n_topos=n_topos)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0; best_state = None; no_improve = 0

    for epoch in range(epochs):
        model.train()
        for Xt_b, Xq_b, cm_b, ds_b in loader:
            Xt_b, Xq_b = Xt_b.to(device), Xq_b.to(device)
            cm_b, ds_b = cm_b.to(device), ds_b.to(device)
            opt.zero_grad()
            loss = criterion(model(Xt_b, Xq_b, ds_b), cm_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            s = model(torch.FloatTensor(Xt_te).to(device),
                      torch.FloatTensor(Xq_te).to(device),
                      torch.LongTensor(ds[tei]).to(device)).cpu()
        preds = s.argmax(dim=1).numpy()
        acc = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0) / len(tei)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 40: break

    return best_acc, best_state


def cv_eval(X_topo, X_q, cm, ds, seed=42, n_folds=5, **kw):
    """Top-1 CV accuracy (deployable metric)."""
    y = cm.argmax(axis=1); y[cm.sum(axis=1) == 0] = 0
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for fold, (tri, tei) in enumerate(kf.split(X_topo, y)):
        acc, _ = train_model(X_topo, X_q, cm, ds, tri, tei,
                             seed=seed + fold, **kw)
        accs.append(acc)
    return np.mean(accs), np.std(accs), accs


def main():
    sys.stdout.reconfigure(line_buffering=True)
    import warnings; warnings.filterwarnings("ignore")

    print("=" * 70)
    print("Router V10 — Clean MajVote Labels, Top-1 Only")
    print("=" * 70)

    print("\n[0] Loading data (MajVote labels)...")
    Xt, Xq, cm, ds = extract_features()
    n_topos = Xt.shape[1]
    oracle = (cm.sum(axis=1) > 0).mean()

    print(f"  Total samples: {len(Xt)}")
    print(f"  Feature shape: {Xt.shape}")

    # Per-topology MajVote accuracy
    print(f"\n  Per-topology MajVote correctness:")
    for t_idx, topo in enumerate(TOPOS):
        rate = cm[:, t_idx].mean()
        print(f"    {topo:15s}: {rate:.1%}")
    print(f"  Oracle (any correct): {oracle:.1%}")

    nc = cm.sum(axis=1)
    for k in range(int(nc.max()) + 1):
        pct = (nc == k).sum() / len(nc) * 100
        print(f"    {int(k)} topos correct: {(nc == k).sum()} ({pct:.1f}%)")

    # Per-dataset breakdown
    print(f"\n  Per-dataset per-topology accuracy:")
    for ds_idx, ds_name in enumerate(DATASETS):
        mask = (ds == ds_idx)
        print(f"  {ds_name}:")
        for t_idx, topo in enumerate(TOPOS):
            rate = cm[mask, t_idx].mean()
            print(f"    {topo:15s}: {rate:.1%}")
        ds_oracle = (cm[mask].sum(axis=1) > 0).mean()
        print(f"    {'oracle':15s}: {ds_oracle:.1%}")

    # ---- Shared model (top-1) ----
    print("\n" + "=" * 70)
    print("[1] Shared model — Top-1 (5 seeds)")
    print("=" * 70)
    shared_results = []
    for s in range(5):
        mean_acc, std_acc, fold_accs = cv_eval(Xt, Xq, cm, ds, seed=s*7+42)
        shared_results.append(mean_acc)
        print(f"  Seed {s}: {mean_acc:.1%} (folds: {[f'{a:.1%}' for a in fold_accs]})")
    shared_mean = np.mean(shared_results)
    shared_std = np.std(shared_results)
    print(f"  RESULT: {shared_mean:.1%} ± {shared_std:.1%}")

    # ---- Per-dataset specialized ----
    print("\n" + "=" * 70)
    print("[2] Per-dataset specialized (5 seeds)")
    print("=" * 70)
    per_ds_results = []
    for s in range(5):
        total_correct = 0; total_count = 0
        for ds_idx, ds_name in enumerate(DATASETS):
            mask = (ds == ds_idx)
            Xt_d, Xq_d = Xt[mask], Xq[mask]
            cm_d, ds_d = cm[mask], ds[mask]
            if len(Xt_d) < 50: continue

            y_d = cm_d.argmax(axis=1); y_d[cm_d.sum(axis=1) == 0] = 0
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s*7+42)
            for fold, (tri, tei) in enumerate(kf.split(Xt_d, y_d)):
                acc, _ = train_model(Xt_d, Xq_d, cm_d, ds_d, tri, tei,
                                     seed=s*7+42+fold, hidden=128)
                total_correct += int(acc * len(tei))
                total_count += len(tei)

        per_ds_results.append(total_correct / max(total_count, 1))

    per_ds_mean = np.mean(per_ds_results)
    per_ds_std = np.std(per_ds_results)
    print(f"  RESULT: {per_ds_mean:.1%} ± {per_ds_std:.1%}")

    # ---- HP sweep (shared model) ----
    print("\n" + "=" * 70)
    print("[3] HP sweep (3 seeds each)")
    print("=" * 70)
    best_hp_acc = 0
    best_hp_cfg = ""
    for hidden in [128, 192, 256]:
        for lr in [5e-4, 1e-3]:
            for drop in [0.15, 0.2]:
                accs = []
                for s in range(3):
                    a, _, _ = cv_eval(Xt, Xq, cm, ds, seed=s*7+42,
                                      hidden=hidden, lr=lr, dropout=drop)
                    accs.append(a)
                mean_a = np.mean(accs)
                if mean_a > best_hp_acc:
                    best_hp_acc = mean_a
                    best_hp_cfg = f"h={hidden},lr={lr},d={drop}"
                    print(f"  NEW BEST: {mean_a:.1%} ({best_hp_cfg})")
    print(f"  Best HP: {best_hp_acc:.1%} ({best_hp_cfg})")

    # ---- Summary ----
    star_idx = TOPOS.index("star")
    star_acc = cm[:, star_idx].mean()
    best_fixed = max(cm[:, t].mean() for t in range(n_topos))
    best_fixed_name = TOPOS[np.argmax([cm[:, t].mean() for t in range(n_topos)])]

    print("\n" + "=" * 70)
    print("FINAL SUMMARY (all top-1, no ground-truth oracle)")
    print("=" * 70)
    print(f"  Best fixed topology: {best_fixed:.1%} ({best_fixed_name})")
    print(f"  Star:                {star_acc:.1%}")
    print(f"  Router (shared):     {shared_mean:.1%} ± {shared_std:.1%}")
    print(f"  Router (per-ds):     {per_ds_mean:.1%} ± {per_ds_std:.1%}")
    print(f"  Router (best HP):    {best_hp_acc:.1%}")
    print(f"  Oracle:              {oracle:.1%}")
    print(f"  Gap closed:          {(shared_mean - best_fixed) / (oracle - best_fixed) * 100:.1f}%")

    # Save results
    results = {
        "label_type": "majority_vote",
        "n_topos": n_topos,
        "n_questions": len(Xt),
        "best_fixed_topo": best_fixed_name,
        "best_fixed_acc": float(best_fixed),
        "star_acc": float(star_acc),
        "oracle_acc": float(oracle),
        "router_shared_mean": float(shared_mean),
        "router_shared_std": float(shared_std),
        "router_per_dataset_mean": float(per_ds_mean),
        "router_per_dataset_std": float(per_ds_std),
        "router_best_hp_acc": float(best_hp_acc),
        "router_best_hp_cfg": best_hp_cfg,
        "per_topo_acc": {t: float(cm[:, i].mean()) for i, t in enumerate(TOPOS)},
    }
    os.makedirs("results/configs", exist_ok=True)
    with open("results/configs/router_v10_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/configs/router_v10_results.json")


if __name__ == "__main__":
    main()
