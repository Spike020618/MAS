"""
Topology Router V2 — 2400+ 样本正确性预测框架

思路: 每个 (题, 拓扑) 对作为独立样本，模型学习 P(correct | signals, topology)
路由时: 对每种拓扑预测 P(correct)，选最高的

自动检测可用拓扑 (4-mode 或 5-mode with hierarchical)

Usage:
  python experiments/train_router_v2.py           # 训练并评估
  python experiments/train_router_v2.py --compare # 与 confidence routing 对比
"""
from __future__ import annotations

import json
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "results/signals"
CONFIGS_DIR = "results/configs"
TRAIN_DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]


def detect_available_topos() -> list:
    """检测哪些拓扑有完整数据。"""
    base_topos = ["parallel", "sequential", "star", "debate"]
    extra = ["hierarchical"]

    available = list(base_topos)
    for topo in extra:
        all_exist = all(
            os.path.exists(os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json"))
            for ds in TRAIN_DATASETS
        )
        if all_exist:
            # Check if all 200 questions have signals
            complete = True
            for ds in TRAIN_DATASETS:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")
                with open(path) as f:
                    data = json.load(f)
                n_with_signals = sum(1 for q in data["questions"] if "signals" in q)
                if n_with_signals < 200:
                    complete = False
                    break
            if complete:
                available.append(topo)

    return available


def build_features(q_signals: dict, target_topo: str, all_q_data: dict,
                   topos: list) -> np.ndarray:
    """为 (题, 拓扑) 对构建特征向量。"""
    s = q_signals
    ic = np.array(s["ic"])
    ec = np.array(s["ec"])
    eq = np.array(s["eq"])
    ev = np.array(s["ev"])
    agr = np.array(s["agreement"])
    stances = s["stances"]
    n = len(stances)
    majority = max(stances.count("yes"), stances.count("no")) / n if n > 0 else 0

    # Primary: target topology's own signals (38 dims)
    primary = np.concatenate([
        ic, ec, eq, ev, agr,  # 25
        np.array([1 if st == "yes" else -1 if st == "no" else 0 for st in stances]),  # 5
        np.array([ic.mean(), ic.std(), ec.mean(), eq.mean(),
                  ev.mean(), agr.mean(), agr.min(), majority]),  # 8
    ])

    # Cross-topology context (6 per topo)
    cross = []
    target_decision = 1 if stances.count("yes") > stances.count("no") else -1
    for other_topo in topos:
        if other_topo not in all_q_data or other_topo == target_topo:
            cross.extend([0] * 6)
            continue
        o = all_q_data[other_topo]
        o_stances = o["stances"]
        o_n = len(o_stances)
        o_majority = max(o_stances.count("yes"), o_stances.count("no")) / o_n if o_n > 0 else 0
        o_decision = 1 if o_stances.count("yes") > o_stances.count("no") else -1
        cross.extend([
            np.array(o["ic"]).mean(),
            np.array(o["ec"]).mean(),
            np.array(o["ev"]).mean(),
            np.array(o["agreement"]).mean(),
            o_majority,
            1.0 if o_decision == target_decision else 0.0,
        ])

    # Cross-topo meta
    decisions = []
    for topo in topos:
        if topo in all_q_data:
            s_t = all_q_data[topo]["stances"]
            decisions.append(1 if s_t.count("yes") > s_t.count("no") else -1)
        else:
            decisions.append(0)
    cross_std = np.std(decisions)
    n_agree = sum(1 for d in decisions if d == target_decision)
    meta = np.array([cross_std, n_agree / len(topos)])

    # Topology one-hot
    topo_onehot = np.zeros(len(topos))
    topo_onehot[topos.index(target_topo)] = 1.0

    return np.concatenate([primary, np.array(cross), meta, topo_onehot])


def load_dataset(topos: list):
    """加载完整数据集。"""
    configs = {}
    for topo in topos:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)

    all_questions = []
    X_all = []
    y_all = []
    q_ids = []
    sample_topos = []

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

        for q_idx in range(200):
            q_data = {}
            q_correct = {}

            for topo in topos:
                if topo not in all_signals:
                    continue
                q = all_signals[topo]["questions"][q_idx]
                if "signals" not in q:
                    continue
                q_data[topo] = q["signals"]
                if topo in configs:
                    res = evaluate_config([q], configs[topo], n_agents)
                    q_correct[topo] = res["correct"] > 0

            all_questions.append({"signals": q_data, "correct": q_correct})
            q_id = len(all_questions) - 1

            for topo in topos:
                if topo not in q_data:
                    continue
                feat = build_features(q_data[topo], topo, q_data, topos)
                X_all.append(feat)
                y_all.append(1 if q_correct.get(topo, False) else 0)
                q_ids.append(q_id)
                sample_topos.append(topo)

    return (np.array(X_all), np.array(y_all), np.array(q_ids),
            sample_topos, all_questions, configs, topos)


def train_and_evaluate(X, y, q_ids, sample_topos, all_questions, topos,
                       n_folds=5):
    """训练 + 交叉验证。"""
    unique_q = np.unique(q_ids)
    q_star_correct = np.array([
        all_questions[qi]["correct"].get("star", False) for qi in unique_q
    ]).astype(int)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_qi, test_qi) in enumerate(kf.split(unique_q, q_star_correct)):
        train_qs = set(unique_q[train_qi])
        test_qs = set(unique_q[test_qi])

        train_mask = np.isin(q_ids, list(train_qs))
        test_mask = np.isin(q_ids, list(test_qs))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        y_train = y[train_mask]

        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            learning_rate=0.05, subsample=0.8, random_state=42)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]

        # Group by question
        test_indices = np.where(test_mask)[0]
        q_probs = {}
        q_correct = {}
        for i, idx in enumerate(test_indices):
            qi = q_ids[idx]
            topo = sample_topos[idx]
            if qi not in q_probs:
                q_probs[qi] = {}
                q_correct[qi] = {}
            q_probs[qi][topo] = probs[i]
            q_correct[qi][topo] = y[idx]

        # Route to max-probability topology
        correct_routed = 0
        correct_star = 0
        topo_picks = {t: 0 for t in topos}

        for qi in q_probs:
            best_topo = max(q_probs[qi], key=q_probs[qi].get)
            topo_picks[best_topo] += 1
            if q_correct[qi].get(best_topo, 0) > 0:
                correct_routed += 1
            if q_correct[qi].get("star", 0) > 0:
                correct_star += 1

        n_q = len(q_probs)
        acc = correct_routed / n_q
        cv_results.append({
            "routing_acc": acc,
            "star_acc": correct_star / n_q,
            "picks": topo_picks,
        })
        print(f"  Fold {fold+1}: routing={acc:.1%} "
              f"(star={correct_star/n_q:.1%}) picks={topo_picks}")

    return cv_results, clf, scaler


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Topology Router V2 — Correctness Prediction Framework")
    print("=" * 60)

    # Detect available topologies
    topos = detect_available_topos()
    n_topos = len(topos)
    n_samples = len(TRAIN_DATASETS) * 200 * n_topos
    print(f"\n  Available topologies: {topos}")
    print(f"  Expected samples: {n_samples}")

    # Load data
    print("\n[1] Loading data...")
    X, y, q_ids, sample_topos, all_questions, configs, topos = load_dataset(topos)
    print(f"  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.1%}")

    # Train and evaluate
    print(f"\n[2] Training GBM Router ({n_topos}-topology, {X.shape[0]} samples)...")
    cv_results, clf, scaler = train_and_evaluate(
        X, y, q_ids, sample_topos, all_questions, topos)

    avg_routing = np.mean([r["routing_acc"] for r in cv_results])
    avg_star = np.mean([r["star_acc"] for r in cv_results])

    # Save model
    model_path = f"results/configs/router_v2_{n_topos}topo.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "clf": clf, "scaler": scaler,
            "topos": topos, "n_features": X.shape[1],
        }, f)
    print(f"\n  Model saved to {model_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Results ({n_topos}-topology router):")
    print(f"  Learned Router:       {avg_routing:.1%}")
    print(f"  Always Star:          {avg_star:.1%}")
    print(f"  Confidence Routing:   83.7%  (star + debate fallback)")
    print(f"  Oracle:               93.2%")

    if args.compare and n_topos > 4:
        # Also train 4-topo version for comparison
        print(f"\n[3] Training 4-topology version for comparison...")
        topos_4 = ["parallel", "sequential", "star", "debate"]
        X4, y4, q4, st4, aq4, _, _ = load_dataset(topos_4)
        cv4, _, _ = train_and_evaluate(X4, y4, q4, st4, aq4, topos_4)
        avg4 = np.mean([r["routing_acc"] for r in cv4])
        print(f"\n  4-topo router: {avg4:.1%}")
        print(f"  {n_topos}-topo router: {avg_routing:.1%}")
        print(f"  Improvement from hierarchical: +{(avg_routing-avg4)*100:.1f}pp")


if __name__ == "__main__":
    main()
