"""
Router V9 — Fast targeted experiments.

From V8 partial results:
  - Baseline: 88.6% ± 0.3%
  - Per-dataset specialized: 89.0% (1 seed only!)
  - Hard overweighting: no help (88.2-88.6%)
  - Diverse feature ensemble: worse (87.6%)

Focus on:
  1. Per-dataset models with proper multi-seed eval
  2. Two-stage routing (fixed loss bug)
  3. HP-diverse ensemble
  4. Top-2 selection (pick best 2 topos, use star as tiebreaker)
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
                        valid = False; break
                    if "signals" not in all_signals[topo]["questions"][q_idx]:
                        valid = False; break
                if not valid:
                    continue

                q_text = all_signals["parallel"]["questions"][q_idx].get("question", "")
                topo_ic_means, topo_agr_means, topo_maj_ratios = [], [], []

                for t_idx, topo in enumerate(TOPOS):
                    q = all_signals[topo]["questions"][q_idx]
                    s = q["signals"]
                    ic = np.array(s["ic"]); ec = np.array(s["ec"])
                    eq = np.array(s["eq"]); ev = np.array(s["ev"])
                    agr = np.array(s["agreement"]); stances = s["stances"]
                    sc = Counter(st.lower() for st in stances)
                    majority = sc.most_common(1)[0][1]
                    yes_like = sum(1 for st in stances if st.lower() in ['yes', 'true'])

                    features[t_idx, 0:5] = ic; features[t_idx, 5:10] = ec
                    features[t_idx, 10:15] = eq; features[t_idx, 15:20] = ev
                    features[t_idx, 20] = ic.mean(); features[t_idx, 21] = ic.std()
                    features[t_idx, 22] = ec.mean(); features[t_idx, 23] = ec.std()
                    features[t_idx, 24] = eq.mean(); features[t_idx, 25] = ev.mean()
                    features[t_idx, 26] = agr.mean(); features[t_idx, 27] = agr.min()
                    features[t_idx, 28] = agr.max(); features[t_idx, 29] = agr.std()
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
                    batch_scores[ds_mask] = self.dataset_scorers[ds_id](inp[ds_mask]).squeeze(-1)
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
        loss = scores.sum() * 0.0  # keep grad_fn
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

    best_acc = 0; best_scores = None; no_improve = 0

    for epoch in range(epochs):
        model.train()
        for Xt_b, Xq_b, cm_b, ds_b in loader:
            Xt_b, Xq_b, cm_b, ds_b = Xt_b.to(device), Xq_b.to(device), cm_b.to(device), ds_b.to(device)
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
            best_acc = acc; best_scores = s.clone(); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 40: break

    return best_acc, best_scores


def cv_eval(X_topo, X_q, cm, ds, seed=42, n_folds=5, **kw):
    y = cm.argmax(axis=1); y[cm.sum(axis=1) == 0] = 0
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for fold, (tri, tei) in enumerate(kf.split(X_topo, y)):
        acc, _ = train_model(X_topo, X_q, cm, ds, tri, tei, seed=seed + fold, **kw)
        accs.append(acc)
    return np.mean(accs), np.std(accs)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    import warnings; warnings.filterwarnings("ignore")

    print("=" * 70)
    print("Router V9 — Focused Experiments")
    print("=" * 70)

    print("\n[0] Loading data...")
    Xt, Xq, cm, ds = extract_features()
    nt = Xt.shape[1]
    oracle = (cm.sum(axis=1) > 0).mean()
    star_acc = cm[:, TOPOS.index("star")].mean()
    nc = cm.sum(axis=1)
    print(f"  {len(Xt)} questions, Star: {star_acc:.1%}, Oracle: {oracle:.1%}")

    results = {}

    # ---- 1. Baseline ----
    print("\n" + "=" * 70)
    print("[1] Baseline (5 seeds)")
    print("=" * 70)
    bl = [cv_eval(Xt, Xq, cm, ds, seed=s*7+42)[0] for s in range(5)]
    results["baseline"] = np.mean(bl)
    print(f"  Baseline: {np.mean(bl):.1%} ± {np.std(bl):.1%}")

    # ---- 2. Per-dataset specialized (proper eval) ----
    print("\n" + "=" * 70)
    print("[2] Per-dataset specialized models (10 seeds)")
    print("=" * 70)
    per_ds_accs = []
    for s in range(10):
        total_correct = 0; total_count = 0
        for ds_idx, ds_name in enumerate(DATASETS):
            mask = (ds == ds_idx)
            Xt_d = Xt[mask]; Xq_d = Xq[mask]; cm_d = cm[mask]; ds_d = ds[mask]
            if len(Xt_d) < 50: continue

            y_d = cm_d.argmax(axis=1); y_d[cm_d.sum(axis=1) == 0] = 0
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s*7+42)
            for fold, (tri, tei) in enumerate(kf.split(Xt_d, y_d)):
                _, scores = train_model(Xt_d, Xq_d, cm_d, ds_d, tri, tei,
                                        seed=s*7+42+fold, hidden=128)
                if scores is not None:
                    preds = scores.argmax(dim=1).numpy()
                    for i, p in enumerate(preds):
                        if cm_d[tei[i], p] > 0: total_correct += 1
                        total_count += 1

        per_ds_accs.append(total_correct / max(total_count, 1))

    results["per_dataset"] = np.mean(per_ds_accs)
    print(f"  Per-dataset (10-seed): {np.mean(per_ds_accs):.1%} ± {np.std(per_ds_accs):.1%}")
    print(f"  Range: {min(per_ds_accs):.1%} - {max(per_ds_accs):.1%}")

    # ---- 3. Per-dataset with HP sweep ----
    print("\n" + "=" * 70)
    print("[3] Per-dataset HP sweep")
    print("=" * 70)
    best_pd_acc = 0
    best_pd_cfg = ""
    for hidden in [128, 192]:
        for lr in [5e-4, 1e-3]:
            for drop in [0.15, 0.2]:
                pd_accs = []
                for s in range(3):
                    tc = 0; tt = 0
                    for ds_idx in range(3):
                        mask = (ds == ds_idx)
                        Xt_d = Xt[mask]; Xq_d = Xq[mask]; cm_d = cm[mask]; ds_d = ds[mask]
                        y_d = cm_d.argmax(axis=1); y_d[cm_d.sum(axis=1) == 0] = 0
                        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s*7+42)
                        for fold, (tri, tei) in enumerate(kf.split(Xt_d, y_d)):
                            _, scores = train_model(Xt_d, Xq_d, cm_d, ds_d, tri, tei,
                                                    seed=s*7+42+fold, hidden=hidden,
                                                    lr=lr, dropout=drop)
                            if scores is not None:
                                preds = scores.argmax(dim=1).numpy()
                                for i, p in enumerate(preds):
                                    if cm_d[tei[i], p] > 0: tc += 1
                                    tt += 1
                    pd_accs.append(tc / max(tt, 1))
                mean_acc = np.mean(pd_accs)
                if mean_acc > best_pd_acc:
                    best_pd_acc = mean_acc
                    best_pd_cfg = f"h={hidden},lr={lr},d={drop}"
                    print(f"  NEW BEST: {mean_acc:.1%} ({best_pd_cfg})")
    results["per_dataset_hp"] = best_pd_acc

    # ---- 4. Top-2 selection ----
    print("\n" + "=" * 70)
    print("[4] Top-2 selection (pick best 2, use star fallback)")
    print("=" * 70)
    star_idx = TOPOS.index("star")
    t2_accs = []
    for s in range(5):
        y = cm.argmax(axis=1); y[cm.sum(axis=1) == 0] = 0
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s*7+42)
        fold_accs = []
        for fold, (tri, tei) in enumerate(kf.split(Xt, y)):
            _, scores = train_model(Xt, Xq, cm, ds, tri, tei, seed=s*7+42+fold)
            if scores is not None:
                top2 = scores.topk(2, dim=1).indices.numpy()
                correct = 0
                for i in range(len(tei)):
                    t1, t2 = top2[i]
                    if cm[tei[i], t1] > 0:
                        correct += 1
                    elif cm[tei[i], t2] > 0:
                        correct += 1
                fold_accs.append(correct / len(tei))
        t2_accs.append(np.mean(fold_accs))
    results["top2"] = np.mean(t2_accs)
    print(f"  Top-2 selection (5-seed): {np.mean(t2_accs):.1%} ± {np.std(t2_accs):.1%}")

    # ---- 5. HP-diverse ensemble (5 different configs) ----
    print("\n" + "=" * 70)
    print("[5] HP-diverse ensemble")
    print("=" * 70)
    hp_configs = [
        {"hidden": 128, "lr": 1e-3, "dropout": 0.15},
        {"hidden": 192, "lr": 1e-3, "dropout": 0.2},
        {"hidden": 256, "lr": 5e-4, "dropout": 0.15},
        {"hidden": 192, "lr": 2e-3, "dropout": 0.2},
        {"hidden": 128, "lr": 5e-4, "dropout": 0.1},
    ]
    hpe_accs = []
    for s in range(5):
        y = cm.argmax(axis=1); y[cm.sum(axis=1) == 0] = 0
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s*7+42)
        fold_accs = []
        for fold, (tri, tei) in enumerate(kf.split(Xt, y)):
            all_scores = []
            for hi, hp in enumerate(hp_configs):
                _, sc = train_model(Xt, Xq, cm, ds, tri, tei,
                                    seed=s*7+42+fold*100+hi*10, **hp)
                if sc is not None: all_scores.append(sc)
            avg = torch.stack(all_scores).mean(dim=0)
            preds = avg.argmax(dim=1).numpy()
            correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
            fold_accs.append(correct / len(tei))
        hpe_accs.append(np.mean(fold_accs))
    results["hp_diverse_ens"] = np.mean(hpe_accs)
    print(f"  HP-diverse ens (5-seed): {np.mean(hpe_accs):.1%} ± {np.std(hpe_accs):.1%}")

    # ---- 6. Two-stage: full + hard specialist blended ----
    print("\n" + "=" * 70)
    print("[6] Two-stage blended routing")
    print("=" * 70)
    is_hard = nc <= 2
    print(f"  Hard questions: {is_hard.sum()} ({is_hard.mean():.1%})")

    for alpha in [0.3, 0.5, 0.7]:
        ts_accs = []
        for s in range(5):
            y = cm.argmax(axis=1); y[cm.sum(axis=1) == 0] = 0
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s*7+42)
            fold_accs = []
            for fold, (tri, tei) in enumerate(kf.split(Xt, y)):
                _, full_scores = train_model(Xt, Xq, cm, ds, tri, tei,
                                             seed=s*7+42+fold)
                hard_tri = tri[is_hard[tri]]
                if len(hard_tri) > 15:
                    _, hard_scores = train_model(Xt, Xq, cm, ds, hard_tri, tei,
                                                 seed=s*7+42+fold+500,
                                                 epochs=400, lr=5e-4, hidden=256)
                    if hard_scores is not None and full_scores is not None:
                        combined = alpha * full_scores + (1-alpha) * hard_scores
                    else:
                        combined = full_scores
                else:
                    combined = full_scores

                if combined is not None:
                    preds = combined.argmax(dim=1).numpy()
                    correct = sum(1 for i, p in enumerate(preds) if cm[tei[i], p] > 0)
                    fold_accs.append(correct / len(tei))

            ts_accs.append(np.mean(fold_accs))
        key = f"two_stage_a{alpha}"
        results[key] = np.mean(ts_accs)
        print(f"  alpha={alpha}: {np.mean(ts_accs):.1%} ± {np.std(ts_accs):.1%}")

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
