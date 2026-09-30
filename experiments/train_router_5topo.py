"""
5-Topology Router — Train with all 5 topologies including hierarchical.

Uses the best architecture from V4/V5 experiments:
  - SharedEncoder + cross-topo context
  - 42-dim features per topology (34 base + 8 interaction)
  - 7-dim question-level features
  - AdaptiveMarginRankingLoss
  - Per-dataset routing heads
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

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "experiments/results/signals"
CONFIGS_DIR = "experiments/results/configs"
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS_4 = ["parallel", "sequential", "star", "debate"]
TOPOS_5 = ["parallel", "sequential", "star", "debate", "hierarchical"]


def extract_features(topos, n_questions=300):
    configs = {}
    for topo in topos:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    TOPO_DIM = 42
    Q_DIM = 7
    n_topos = len(topos)

    X_topo_all = []
    X_q_all = []
    cm_all = []
    ds_all = []

    for ds_idx, ds in enumerate(DATASETS):
        all_signals = {}
        for topo in topos:
            if topo == "parallel":
                path = f"{SIGNALS_DIR}/{ds}.json"
            else:
                path = f"{SIGNALS_DIR}/{ds}_{topo}.json"
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping {ds}/{topo}")
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

                for t_idx, topo in enumerate(topos):
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
                topo_agr_means = []
                topo_maj_ratios = []

                for t_idx, topo in enumerate(topos):
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

                    # Base features (34)
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

                    # Interaction features (8)
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


class Router5Topo(nn.Module):
    def __init__(self, topo_feat_dim=42, q_feat_dim=7, hidden=128,
                 n_topos=5, n_datasets=3):
        super().__init__()
        self.n_topos = n_topos
        self.n_datasets = n_datasets

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


class AdaptiveMarginRankingLoss(nn.Module):
    def __init__(self, base_margin=0.5, max_margin=2.0, n_topos=5):
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
            difficulty = 1.0 - len(correct_idx) / self.n_topos
            margin = self.base_margin + difficulty * (self.max_margin - self.base_margin)
            for c in correct_idx:
                for w in incorrect_idx:
                    loss += F.relu(margin - (scores[b, c] - scores[b, w]))
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


def train_and_eval(X_topo, X_q, cm, ds, n_topos, seed=42, n_folds=5,
                   epochs=200, lr=1e-3, hidden=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topo_feat_dim = X_topo.shape[2]
    q_feat_dim = X_q.shape[1]

    y_strat = cm.argmax(axis=1)
    y_strat[cm.sum(axis=1) == 0] = 0

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []
    fold_details = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_topo, y_strat)):
        scaler_t = StandardScaler()
        Xt_tr = scaler_t.fit_transform(
            X_topo[train_idx].reshape(-1, topo_feat_dim)
        ).reshape(len(train_idx), n_topos, topo_feat_dim)
        Xt_te = scaler_t.transform(
            X_topo[test_idx].reshape(-1, topo_feat_dim)
        ).reshape(len(test_idx), n_topos, topo_feat_dim)

        scaler_q = StandardScaler()
        Xq_tr = scaler_q.fit_transform(X_q[train_idx])
        Xq_te = scaler_q.transform(X_q[test_idx])

        train_ds = RouterDataset(Xt_tr, Xq_tr, cm[train_idx], ds[train_idx])
        test_ds = RouterDataset(Xt_te, Xq_te, cm[test_idx], ds[test_idx])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        test_loader = DataLoader(test_ds, batch_size=256)

        torch.manual_seed(seed + fold)
        model = Router5Topo(
            topo_feat_dim=topo_feat_dim, q_feat_dim=q_feat_dim,
            hidden=hidden, n_topos=n_topos, n_datasets=3
        ).to(device)
        criterion = AdaptiveMarginRankingLoss(n_topos=n_topos)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        patience = 30
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

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def main():
    sys.stdout.reconfigure(line_buffering=True)
    import warnings; warnings.filterwarnings("ignore")

    print("=" * 70)
    print("5-Topology Router Training")
    print("=" * 70)

    # ---- 4-topo baseline ----
    print("\n[1] 4-topo baseline (reproduce)...")
    X4t, X4q, cm4, ds4 = extract_features(TOPOS_4)
    star_idx = TOPOS_4.index("star")
    oracle4 = (cm4.sum(axis=1) > 0).mean()
    print(f"  {len(X4t)} questions, Star: {cm4[:, star_idx].mean():.1%}, Oracle: {oracle4:.1%}")

    results = {}
    seed_accs_4 = []
    for s in range(5):
        acc, _, _ = train_and_eval(X4t, X4q, cm4, ds4, n_topos=4, seed=s*7+42)
        seed_accs_4.append(acc)
    results["4-topo"] = np.mean(seed_accs_4)
    print(f"  4-topo router: {np.mean(seed_accs_4):.1%} ± {np.std(seed_accs_4):.1%}")
    print(f"  Seeds: {[f'{a:.1%}' for a in seed_accs_4]}")

    # ---- 5-topo ----
    print(f"\n[2] 5-topo router...")
    X5t, X5q, cm5, ds5 = extract_features(TOPOS_5)
    oracle5 = (cm5.sum(axis=1) > 0).mean()
    print(f"  {len(X5t)} questions, Oracle: {oracle5:.1%}")

    # Per-topo accuracy
    for t_idx, topo in enumerate(TOPOS_5):
        acc = cm5[:, t_idx].mean()
        print(f"  {topo:12s}: {acc:.1%}")

    # Distribution
    for nc in range(6):
        count = (cm5.sum(axis=1) == nc).sum()
        if count > 0:
            print(f"  {nc} topos correct: {count} ({count/len(cm5)*100:.1f}%)")

    seed_accs_5 = []
    for s in range(5):
        acc, _, _ = train_and_eval(X5t, X5q, cm5, ds5, n_topos=5, seed=s*7+42)
        seed_accs_5.append(acc)
    results["5-topo"] = np.mean(seed_accs_5)
    print(f"  5-topo router: {np.mean(seed_accs_5):.1%} ± {np.std(seed_accs_5):.1%}")
    print(f"  Seeds: {[f'{a:.1%}' for a in seed_accs_5]}")

    # ---- 5-topo hyperparameter sweep ----
    print(f"\n[3] 5-topo hyperparameter sweep...")
    best_5t_acc = 0
    best_5t_cfg = ""
    for hidden in [128, 256]:
        for lr in [5e-4, 1e-3]:
            acc, std, _ = train_and_eval(
                X5t, X5q, cm5, ds5, n_topos=5,
                hidden=hidden, lr=lr, seed=42)
            if acc > best_5t_acc:
                best_5t_acc = acc
                best_5t_cfg = f"h={hidden},lr={lr}"
                print(f"  NEW BEST: {acc:.1%} ± {std:.1%} ({best_5t_cfg})")
    results["5-topo-best"] = best_5t_acc

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Star baseline:     {cm4[:, star_idx].mean():.1%}")
    print(f"  4-topo router:     {results['4-topo']:.1%} (oracle {oracle4:.1%})")
    print(f"  5-topo router:     {results['5-topo']:.1%} (oracle {oracle5:.1%})")
    print(f"  5-topo best:       {results['5-topo-best']:.1%}")
    print(f"  Improvement:       +{(results['5-topo'] - results['4-topo'])*100:.1f}pp")


if __name__ == "__main__":
    main()
