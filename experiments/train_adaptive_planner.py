"""
Train Adaptive Topology Planner — 基于信号特征的拓扑路由

两种策略:
1. Classifier-based: 从 parallel signals 统计特征预测最优拓扑 (4类分类)
2. Confidence-based (更优): star topology + 多数投票检测 → 低共识时 escalate to debate

训练集: MMLU-Hard + StrategyQA + TruthfulQA (600题)

Usage:
  python experiments/train_adaptive_planner.py
  python experiments/train_adaptive_planner.py --confidence  # 推荐: 基于共识度的路由
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

TRAIN_DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate"]
SIGNALS_DIR = "experiments/results/signals"
CONFIGS_DIR = "experiments/results/configs"


def extract_features(signals: dict) -> np.ndarray:
    """从一道题的 parallel signals 提取分类特征。"""
    ic = np.array(signals["ic"])
    ec = np.array(signals["ec"])
    eq = np.array(signals["eq"])
    ev = np.array(signals["ev"])
    agreement = np.array(signals["agreement"])
    stances = signals["stances"]

    yes_count = stances.count("yes")
    no_count = stances.count("no")
    neutral_count = stances.count("neutral")
    n = len(stances)

    majority_ratio = max(yes_count, no_count) / n if n > 0 else 0
    stance_entropy = 0
    for count in [yes_count, no_count, neutral_count]:
        if count > 0:
            p = count / n
            stance_entropy -= p * np.log2(p)

    features = [
        ic.mean(), ic.std(), ic.min(), ic.max(),
        ec.mean(), ec.std(), ec.min(), ec.max(),
        eq.mean(), eq.std(),
        ev.mean(), ev.std(),
        agreement.mean(), agreement.std(), agreement.min(),
        majority_ratio,
        stance_entropy,
        (ic * ec).mean(),  # IC-EC interaction
        (ev * agreement).mean(),  # EV-agreement interaction
        np.corrcoef(ic, ec)[0, 1] if ic.std() > 0 and ec.std() > 0 else 0,
    ]
    return np.array(features)


def get_oracle_labels(datasets: list) -> tuple:
    """为每道题计算 oracle label（哪种拓扑效果最好）。"""
    configs = {}
    for topo in TOPOS:
        configs[topo] = ConsensusConfig.load(
            os.path.join(CONFIGS_DIR, f"opt_{topo}.json"))

    X = []
    y = []
    question_info = []

    for ds in datasets:
        # Load parallel signals for features
        parallel_path = os.path.join(SIGNALS_DIR, f"{ds}.json")
        with open(parallel_path) as f:
            parallel_data = json.load(f)

        # Load all topology signals for labels
        all_signals = {}
        for topo in TOPOS:
            if topo == "parallel":
                path = parallel_path
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")
            if os.path.exists(path):
                with open(path) as f:
                    all_signals[topo] = json.load(f)

        n_agents = parallel_data["meta"]["n_agents"]

        for q_idx, q in enumerate(parallel_data["questions"]):
            if "signals" not in q:
                continue

            # Features from parallel signals
            features = extract_features(q["signals"])

            # Oracle: which topology gets it right?
            topo_correct = {}
            for topo in TOPOS:
                if topo not in all_signals:
                    continue
                topo_q = all_signals[topo]["questions"][q_idx]
                if "signals" not in topo_q:
                    continue
                res = evaluate_config([topo_q], configs[topo], n_agents)
                topo_correct[topo] = res["correct"] > 0

            # Pick best topology (prefer cheaper if tied)
            # Priority: parallel < sequential < star < debate (by cost)
            best_topo = "parallel"
            for topo in ["star", "debate", "sequential", "parallel"]:
                if topo_correct.get(topo, False):
                    best_topo = topo
                    break

            # If none got it right, label as parallel (cheapest)
            if not any(topo_correct.values()):
                best_topo = "parallel"

            X.append(features)
            y.append(TOPOS.index(best_topo))
            question_info.append({"dataset": ds, "q_idx": q_idx})

    return np.array(X), np.array(y), question_info


def train_classifier(X, y):
    """训练拓扑选择分类器。"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 尝试多种分类器
    classifiers = {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    print("\nClassifier comparison (5-fold CV):")
    print(f"{'Classifier':<22} {'Accuracy':>10} {'Std':>8}")
    print("-" * 42)

    best_name = None
    best_score = 0
    best_clf = None

    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        print(f"{name:<22} {mean_score:>9.1%} {scores.std():>7.1%}")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_clf = clf

    print(f"\nBest: {best_name} ({best_score:.1%})")

    # Train best on full data
    best_clf.fit(X_scaled, y)

    # Feature importance (if available)
    feature_names = [
        "ic_mean", "ic_std", "ic_min", "ic_max",
        "ec_mean", "ec_std", "ec_min", "ec_max",
        "eq_mean", "eq_std",
        "ev_mean", "ev_std",
        "agr_mean", "agr_std", "agr_min",
        "majority_ratio", "stance_entropy",
        "ic_ec_interaction", "ev_agr_interaction", "ic_ec_corr",
    ]

    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("\nTop features:")
        for i in range(min(8, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"  {feature_names[idx]:<20} {importances[idx]:.3f}")

    return best_clf, scaler, best_name


def evaluate_adaptive_with_classifier(clf, scaler, datasets: list) -> dict:
    """用训练好的 classifier 做 adaptive 拓扑选择。"""
    configs = {}
    for topo in TOPOS:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        configs[topo] = ConsensusConfig.load(cfg_path)

    results = {}
    for ds in datasets:
        parallel_path = os.path.join(SIGNALS_DIR, f"{ds}.json")
        if not os.path.exists(parallel_path):
            continue
        with open(parallel_path) as f:
            parallel_data = json.load(f)

        # Load all topology signals
        all_signals = {}
        for topo in TOPOS:
            if topo == "parallel":
                path = parallel_path
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")
            if os.path.exists(path):
                with open(path) as f:
                    all_signals[topo] = json.load(f)

        n_agents = parallel_data["meta"]["n_agents"]
        correct = 0
        total = 0
        topo_counts = {t: 0 for t in TOPOS}

        for q_idx, q in enumerate(parallel_data["questions"]):
            if "signals" not in q:
                continue

            features = extract_features(q["signals"]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            pred_topo_idx = clf.predict(features_scaled)[0]
            pred_topo = TOPOS[pred_topo_idx]
            topo_counts[pred_topo] += 1

            # Evaluate with predicted topology
            if pred_topo in all_signals:
                topo_q = all_signals[pred_topo]["questions"][q_idx]
                if "signals" in topo_q:
                    res = evaluate_config([topo_q], configs[pred_topo], n_agents)
                    if res["correct"] > 0:
                        correct += 1
                    total += 1
                    continue

            # Fallback to parallel
            res = evaluate_config([q], configs["parallel"], n_agents)
            if res["correct"] > 0:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        results[ds] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "topology_distribution": topo_counts,
        }
        print(f"  {ds}: {acc:.1%} ({correct}/{total}) topo={topo_counts}")

    return results


def evaluate_confidence_routing(datasets: list, threshold: float = 0.5) -> dict:
    """基于共识度的路由: star → 低共识时 escalate to debate。"""
    configs = {}
    for topo in TOPOS:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        configs[topo] = ConsensusConfig.load(cfg_path)

    results = {}
    for ds in datasets:
        star_path = os.path.join(SIGNALS_DIR, f"{ds}_star.json")
        debate_path = os.path.join(SIGNALS_DIR, f"{ds}_debate.json")

        if not os.path.exists(star_path):
            continue
        with open(star_path) as f:
            star_data = json.load(f)

        debate_data = None
        if os.path.exists(debate_path):
            with open(debate_path) as f:
                debate_data = json.load(f)

        n_agents = star_data["meta"]["n_agents"]
        correct = 0
        total = 0
        routed_to_debate = 0

        for q_idx, q in enumerate(star_data["questions"]):
            if "signals" not in q:
                continue

            stances = q["signals"]["stances"]
            n = len(stances)
            majority_ratio = max(
                stances.count("yes"), stances.count("no")
            ) / n if n > 0 else 0

            if majority_ratio < threshold and debate_data:
                debate_q = debate_data["questions"][q_idx]
                if "signals" in debate_q:
                    res = evaluate_config([debate_q], configs["debate"], n_agents)
                    routed_to_debate += 1
                else:
                    res = evaluate_config([q], configs["star"], n_agents)
            else:
                res = evaluate_config([q], configs["star"], n_agents)

            if res["correct"] > 0:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        results[ds] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "routed_to_debate": routed_to_debate,
        }
        print(f"  {ds}: {acc:.1%} ({correct}/{total}), "
              f"debate_escalation={routed_to_debate}/{total}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Also evaluate on test datasets")
    parser.add_argument("--confidence", action="store_true",
                        help="Use confidence-based routing (recommended)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Majority threshold for debate escalation")
    args = parser.parse_args()

    print("=" * 60)
    print("Training Adaptive Topology Planner")
    print("=" * 60)

    if args.confidence:
        # Confidence-based routing
        print(f"\n[Confidence-based routing] threshold={args.threshold}")
        print("Rule: if star majority_ratio < threshold → escalate to debate")
        print("\nEvaluating on training datasets...")
        results = evaluate_confidence_routing(TRAIN_DATASETS, args.threshold)
        avg = np.mean([r["accuracy"] for r in results.values()])
        print(f"\n  Avg accuracy: {avg:.1%}")

        # Save routing config
        routing_config = {
            "strategy": "confidence_routing",
            "primary_topology": "star",
            "fallback_topology": "debate",
            "threshold": args.threshold,
            "rule": "majority_ratio < threshold → debate",
            "accuracy": avg,
        }
        config_path = "experiments/results/configs/adaptive_routing.json"
        with open(config_path, "w") as f:
            json.dump(routing_config, f, indent=2)
        print(f"  Config saved to {config_path}")

        print("\n" + "=" * 60)
        print("Final comparison:")
        print(f"  ChatEval (parallel+vote):     73.0%")
        print(f"  UnguidedDebate (debate+vote): 73.0%")
        print(f"  Best baseline (MajVote_star): 78.2%")
        print(f"  AEIC+SG_star (fixed best):    80.0%")
        print(f"  Adaptive (confidence route):  {avg:.1%}")
        print(f"  Oracle (upper bound):         93.2%")
        return

    # Original classifier-based approach
    print("\n[1] Building training data (oracle labels)...")
    X, y, info = get_oracle_labels(TRAIN_DATASETS)
    print(f"  Samples: {len(X)}")
    print(f"  Label distribution: {dict(zip(TOPOS, [int((y==i).sum()) for i in range(len(TOPOS))]))}")

    print("\n[2] Training classifier...")
    clf, scaler, clf_name = train_classifier(X, y)

    print("\n[3] Evaluating on training datasets (adaptive with classifier)...")
    train_results = evaluate_adaptive_with_classifier(clf, scaler, TRAIN_DATASETS)

    avg_train = np.mean([r["accuracy"] for r in train_results.values()])
    print(f"\n  Train avg: {avg_train:.1%}")

    model_path = "experiments/results/configs/adaptive_planner.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler, "name": clf_name,
                     "topos": TOPOS, "feature_names": [
                         "ic_mean", "ic_std", "ic_min", "ic_max",
                         "ec_mean", "ec_std", "ec_min", "ec_max",
                         "eq_mean", "eq_std", "ev_mean", "ev_std",
                         "agr_mean", "agr_std", "agr_min",
                         "majority_ratio", "stance_entropy",
                         "ic_ec_interaction", "ev_agr_interaction", "ic_ec_corr",
                     ]}, f)
    print(f"\n  Model saved to {model_path}")

    print("\n" + "=" * 60)
    print("Final comparison:")
    print(f"  ChatEval (parallel+vote):     73.0%")
    print(f"  Best baseline (MajVote_star): 78.2%")
    print(f"  AEIC+SG_star (fixed best):    80.0%")
    print(f"  Adaptive (trained clf):       {avg_train:.1%}")
    print(f"  Oracle (upper bound):         93.2%")


if __name__ == "__main__":
    main()
