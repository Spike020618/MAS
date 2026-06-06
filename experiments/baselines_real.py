"""
Real Baseline Evaluation — 基于真实LLM生成数据的baseline对比

每种baseline使用其对应拓扑下的真实LLM输出（不再模拟）：
- ChatEval: parallel拓扑 + 多数投票
- UnguidedDebate: debate拓扑 + 最终轮多数投票
- AgentVerse: agentverse拓扑 + solver决策

Usage:
  python experiments/baselines_real.py
  python experiments/baselines_real.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
SIGNALS_DIR = "results/signals"


def _load_signals(dataset: str, topology: str) -> dict:
    """加载指定数据集+拓扑的signals数据。"""
    if topology == "parallel":
        path = os.path.join(SIGNALS_DIR, f"{dataset}.json")
    else:
        path = os.path.join(SIGNALS_DIR, f"{dataset}_{topology}.json")

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_majority_vote(questions: list) -> dict:
    """简单多数投票：统计stances中出现最多的作为决策。"""
    correct = 0
    total = 0
    for q in questions:
        if "signals" not in q:
            continue
        stances = q["signals"]["stances"]
        gt = q["ground_truth"]

        yes_count = stances.count("yes")
        no_count = stances.count("no")

        if yes_count > no_count:
            decision = "yes"
        elif no_count > yes_count:
            decision = "no"
        else:
            decision = "neutral"

        if decision == gt:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total > 0 else 0,
            "correct": correct, "total": total}


def evaluate_agentverse_solver(questions: list) -> dict:
    """AgentVerse: solver（agent 0）修正后的决策。"""
    from core.trust import extract_stance

    correct = 0
    total = 0
    for q in questions:
        agents = q.get("agents", [])
        if not agents:
            continue
        gt = q["ground_truth"]

        solver_aeic = agents[0].get("aeic", {})
        solver_conclusion = solver_aeic.get("C", "")
        decision = extract_stance(solver_conclusion)

        if decision == gt:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total > 0 else 0,
            "correct": correct, "total": total}


def evaluate_chateval(dataset: str) -> dict:
    """ChatEval: parallel拓扑 + 多数投票。"""
    data = _load_signals(dataset, "parallel")
    if not data:
        return None
    return evaluate_majority_vote(data["questions"])


def evaluate_unguided_debate(dataset: str) -> dict:
    """UnguidedDebate: debate拓扑 + 最终轮多数投票。"""
    data = _load_signals(dataset, "debate")
    if not data:
        return None
    return evaluate_majority_vote(data["questions"])


def evaluate_agentverse(dataset: str) -> dict:
    """AgentVerse: agentverse拓扑 + solver修正后决策。"""
    data = _load_signals(dataset, "agentverse")
    if not data:
        return None
    return evaluate_agentverse_solver(data["questions"])


def evaluate_ours(dataset: str, topology: str, config_path: str = None) -> dict:
    """Our method: 指定拓扑 + Stackelberg加权。"""
    data = _load_signals(dataset, topology)
    if not data:
        return None

    if config_path and os.path.exists(config_path):
        cfg = ConsensusConfig.load(config_path)
    else:
        cfg = ConsensusConfig()

    n_agents = data["meta"]["n_agents"]
    questions = [q for q in data["questions"] if "signals" in q]
    return evaluate_config(questions, cfg, n_agents)


def evaluate_majvote_on_topology(dataset: str, topology: str) -> dict:
    """MajVote baseline: 指定拓扑的数据 + 简单多数投票。"""
    data = _load_signals(dataset, topology)
    if not data:
        return None
    return evaluate_majority_vote(data["questions"])


def run_full_comparison():
    """运行完整对比实验。"""
    topologies = ["parallel", "sequential", "star", "debate"]
    configs_dir = "results/configs"

    results = {}
    for ds in DATASETS:
        results[ds] = {}

        # --- Baselines (各自用对应拓扑) ---
        chateval = evaluate_chateval(ds)
        if chateval:
            results[ds]["ChatEval"] = chateval

        debate = evaluate_unguided_debate(ds)
        if debate:
            results[ds]["UnguidedDebate"] = debate

        agentverse = evaluate_agentverse(ds)
        if agentverse:
            results[ds]["AgentVerse"] = agentverse

        # --- Per-topology: MajVote vs Ours ---
        for topo in topologies:
            majvote = evaluate_majvote_on_topology(ds, topo)
            if majvote:
                results[ds][f"MajVote_{topo}"] = majvote

            cfg_path = os.path.join(configs_dir, f"opt_{topo}.json")
            ours = evaluate_ours(ds, topo, cfg_path)
            if ours:
                results[ds][f"AEIC+SG_{topo}"] = ours

    return results


def print_results(results: dict):
    """打印结果表格。"""
    # 方法列表
    methods = ["ChatEval", "UnguidedDebate", "AgentVerse"]
    topologies = ["parallel", "sequential", "star", "debate"]
    for topo in topologies:
        methods.append(f"MajVote_{topo}")
        methods.append(f"AEIC+SG_{topo}")

    print(f"\n{'Method':<24}", end="")
    for ds in DATASETS:
        print(f" {ds:>12}", end="")
    print(f" {'Avg':>8}")
    print("-" * (24 + 13 * len(DATASETS) + 9))

    for method in methods:
        accs = []
        print(f"{method:<24}", end="")
        for ds in DATASETS:
            if ds in results and method in results[ds]:
                acc = results[ds][method]["accuracy"]
                accs.append(acc)
                print(f" {acc:>11.1%}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        avg = sum(accs) / len(accs) if accs else 0
        print(f" {avg:>7.1%}")


def verify():
    """验证：检查所有需要的signals文件是否存在。"""
    print("=== Baseline Verification ===\n")

    checks = [
        ("ChatEval", "parallel"),
        ("UnguidedDebate", "debate"),
        ("AgentVerse", "agentverse"),
    ]
    topologies = ["parallel", "sequential", "star", "debate"]

    all_ok = True
    for ds in DATASETS:
        print(f"Dataset: {ds}")
        for name, topo in checks:
            data = _load_signals(ds, topo)
            status = f"OK ({len(data['questions'])}q)" if data else "MISSING"
            if not data:
                all_ok = False
            print(f"  {name} ({topo}): {status}")

        for topo in topologies:
            data = _load_signals(ds, topo)
            status = f"OK ({len(data['questions'])}q)" if data else "MISSING"
            if not data:
                all_ok = False
            print(f"  Ours ({topo}): {status}")
        print()

    # 检查 config 文件
    print("Configs:")
    for topo in topologies:
        cfg_path = f"results/configs/opt_{topo}.json"
        exists = os.path.exists(cfg_path)
        print(f"  {cfg_path}: {'OK' if exists else 'MISSING'}")
        if not exists:
            all_ok = False

    print(f"\n{'All ready!' if all_ok else 'Some files missing.'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Real Baseline Evaluation")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", default="results/comparison_results.json")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    results = run_full_comparison()
    print_results(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
