"""
泛化评估 — 在多个数据集上评估config + 1-agent基线 + 实验日志

Usage:
  # 评估一个config在所有已有signals上的泛化
  python experiments/eval_generalization.py \
      --config experiments/results/configs/opt_001.json

  # 完整流水线：优化+泛化+日志
  python experiments/eval_generalization.py \
      --config experiments/results/configs/opt_001.json \
      --train_datasets StrategyQA \
      --train_accuracy 0.815
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import (
    consensus_from_signals, evaluate_config, compute_ic_metric,
)


LOG_PATH = "experiments/results/optimization_log.json"


def load_log() -> Dict:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"experiments": [], "best": None}


def save_log(log: Dict):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def next_experiment_id(log: Dict) -> str:
    existing = [e["id"] for e in log["experiments"]]
    for i in range(1, 9999):
        eid = f"opt_{i:03d}"
        if eid not in existing:
            return eid
    return f"opt_{len(existing)+1:03d}"


def find_all_signals(results_dir: str = "results") -> List[str]:
    pattern = os.path.join(results_dir, "signals", "*.json")
    return sorted(glob.glob(pattern))


def extract_single_agent_baseline(signals_path: str) -> Dict:
    """从signals缓存提取agent 0（DeepSeek）的单智能体准确率。"""
    with open(signals_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    correct = 0
    total = 0
    for q in data["questions"]:
        if "signals" not in q:
            continue
        stances = q["signals"].get("stances", [])
        if not stances:
            continue
        agent0_stance = stances[0]
        if agent0_stance == q["ground_truth"]:
            correct += 1
        total += 1
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def evaluate_on_all(config: ConsensusConfig,
                    signal_files: List[str]) -> Dict[str, Dict]:
    """评估config在多个数据集上的准确率和IC。"""
    results = {}
    for sf in signal_files:
        with open(sf, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = data["meta"]["dataset"]
        n_agents = data["meta"]["n_agents"]
        result = evaluate_config(data["questions"], config, n_agents)
        result["ic"] = compute_ic_metric(data["questions"], config, n_agents)
        results[dataset] = result
    return results


def run_generalization(config_path: str,
                       train_datasets: Optional[List[str]] = None,
                       train_accuracy: Optional[float] = None,
                       results_dir: str = "results"):
    """评估一个config的泛化能力，输出对比表，记录到log。"""
    config = ConsensusConfig.load(config_path)
    default_config = ConsensusConfig()

    signal_files = find_all_signals(results_dir)
    if not signal_files:
        print("No signals files found.")
        return

    print(f"Evaluating config: {config_path}")
    print(f"Found {len(signal_files)} signal files\n")

    optimized_results = evaluate_on_all(config, signal_files)
    default_results = evaluate_on_all(default_config, signal_files)

    single_agent = {}
    for sf in signal_files:
        with open(sf, "r", encoding="utf-8") as f:
            ds = json.load(f)["meta"]["dataset"]
        single_agent[ds] = extract_single_agent_baseline(sf)

    header = (f"{'Dataset':<18} {'1-Agent':>8} {'Default':>8} {'Optimized':>10} "
              f"{'Δacc':>6} {'IC-def':>7} {'IC-opt':>7}")
    print(header)
    print("-" * len(header))

    datasets = sorted(optimized_results.keys())
    sum_1a, sum_def, sum_opt, sum_ic_def, sum_ic_opt = 0, 0, 0, 0, 0
    n_ds = 0
    for ds in datasets:
        a1 = single_agent.get(ds, {}).get("accuracy", 0)
        a_def = default_results[ds]["accuracy"]
        a_opt = optimized_results[ds]["accuracy"]
        ic_def = default_results[ds].get("ic", 0)
        ic_opt = optimized_results[ds].get("ic", 0)
        delta = a_opt - a_def
        marker = " *" if train_datasets and ds in train_datasets else ""
        print(f"{ds:<18} {a1:>7.1%} {a_def:>7.1%} {a_opt:>9.1%} "
              f"{delta:>+5.1%} {ic_def:>6.1%} {ic_opt:>6.1%}{marker}")
        sum_1a += a1
        sum_def += a_def
        sum_opt += a_opt
        sum_ic_def += ic_def
        sum_ic_opt += ic_opt
        n_ds += 1

    if n_ds > 0:
        print("-" * len(header))
        print(f"{'Average':<18} {sum_1a/n_ds:>7.1%} {sum_def/n_ds:>7.1%} "
              f"{sum_opt/n_ds:>9.1%} {(sum_opt-sum_def)/n_ds:>+5.1%} "
              f"{sum_ic_def/n_ds:>6.1%} {sum_ic_opt/n_ds:>6.1%}")

    test_results = {}
    for ds in datasets:
        if train_datasets and ds in train_datasets:
            continue
        test_results[ds] = optimized_results[ds]["accuracy"]
    avg_test = sum(test_results.values()) / max(len(test_results), 1)

    log = load_log()
    eid = next_experiment_id(log)
    entry = {
        "id": eid,
        "timestamp": datetime.now().isoformat(),
        "train_datasets": train_datasets or [],
        "config_path": config_path,
        "train_accuracy": train_accuracy,
        "default_accuracy": {ds: default_results[ds]["accuracy"] for ds in datasets},
        "default_ic": {ds: default_results[ds].get("ic", 0) for ds in datasets},
        "optimized_accuracy": {ds: optimized_results[ds]["accuracy"] for ds in datasets},
        "optimized_ic": {ds: optimized_results[ds].get("ic", 0) for ds in datasets},
        "single_agent_accuracy": {ds: single_agent.get(ds, {}).get("accuracy", 0) for ds in datasets},
        "test_results": test_results,
        "avg_test_accuracy": avg_test,
        "avg_all_accuracy": sum_opt / max(n_ds, 1),
    }
    log["experiments"].append(entry)

    if log["best"] is None or avg_test > log["best"].get("avg_test_accuracy", 0):
        log["best"] = {"id": eid, "avg_test_accuracy": avg_test, "config_path": config_path}
        print(f"\n  NEW BEST: {eid} avg_test={avg_test:.1%}")

    save_log(log)
    print(f"\n  Logged as {eid} → {LOG_PATH}")
    print(f"  Avg test accuracy (excl train): {avg_test:.1%}")

    return entry


def main():
    parser = argparse.ArgumentParser(description="Evaluate config generalization")
    parser.add_argument("--config", required=True, help="Config JSON path")
    parser.add_argument("--train_datasets", nargs="*", default=None,
                        help="Dataset names used for training (marked in output)")
    parser.add_argument("--train_accuracy", type=float, default=None,
                        help="Training accuracy to log")
    parser.add_argument("--results_dir", default="results",
                        help="Directory with signals files")
    args = parser.parse_args()

    run_generalization(args.config, args.train_datasets,
                       args.train_accuracy, args.results_dir)


if __name__ == "__main__":
    main()
