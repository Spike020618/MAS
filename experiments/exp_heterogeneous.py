"""
Experiment: Heterogeneous Agent Quality (Plan A)

7个agent池，混合 deepseek + flash 后端。
全部7个agent参与每题，AEIC+Stackelberg共识自动识别弱agent。

对比条件:
  - 1flash: 7个agent中1个flash (agent 3)
  - 2flash: 7个agent中2个flash (agents 1,5)
  - 3flash: 7个agent中3个flash (agents 1,3,5)

Usage:
  python experiments/exp_heterogeneous.py --dataset TruthfulQA --n_samples 15
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import LLMClient, MultiLLMPool, AGENT_PERSONAS
from core.planner import TopologyPlanner, PlanResult, topological_layers
from core.adaptive_executor import AdaptiveExecutor
from core.similarity import SimilarityCalculator
from core.trust import extract_stance, majority_vote
from experiments.exp_orchestrator import load_questions, evaluate_accuracy

POOL_SIZE = 7

POOL_CONFIGS = {
    "1flash": {
        0: "deepseek", 1: "deepseek", 2: "deepseek",
        3: "zhipu_flash",
        4: "deepseek", 5: "deepseek", 6: "deepseek",
    },
    "2flash": {
        0: "deepseek", 1: "zhipu_flash", 2: "deepseek",
        3: "deepseek", 4: "deepseek",
        5: "zhipu_flash", 6: "deepseek",
    },
    "3flash": {
        0: "deepseek", 1: "zhipu_flash", 2: "deepseek",
        3: "zhipu_flash", 4: "deepseek",
        5: "zhipu_flash", 6: "deepseek",
    },
}

CSV_FIELDS = [
    "dataset", "condition", "question_id", "question", "ground_truth",
    "adaptive_decision", "majvote_decision",
    "per_agent_ic", "per_agent_ec", "per_agent_theta",
    "per_agent_stance", "per_agent_backend",
    "mechanism", "elapsed_s",
]


def _load_completed(csv_path: str) -> Set[Tuple]:
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add((row["dataset"], row["condition"], row["question_id"]))
    return completed


def _append_row(csv_path: str, row: Dict):
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_condition(
    condition: str,
    questions: List[Dict],
    dataset_name: str,
    csv_path: str = None,
) -> Tuple[List[Dict], Dict]:

    agent_map = POOL_CONFIGS[condition]
    completed = _load_completed(csv_path) if csv_path else set()

    pool = MultiLLMPool(agent_map=agent_map)
    sim = SimilarityCalculator(method="embedding")
    executor = AdaptiveExecutor(pool, sim, n_agents=POOL_SIZE)

    flash_agents = {k for k, v in agent_map.items() if "flash" in v}
    results = []
    agent_stats = {
        i: {"ic": [], "ec": [], "eq": [], "theta": [], "correct": 0, "total": 0,
            "backend": agent_map[i], "is_flash": i in flash_agents}
        for i in range(POOL_SIZE)
    }

    for q_idx, q_item in enumerate(questions):
        question = q_item["question"]
        gt = q_item["ground_truth"]

        if csv_path and (dataset_name, condition, str(q_idx)) in completed:
            continue

        print(f"    [{q_idx+1}/{len(questions)}] {question[:50]}...")
        t0 = time.time()

        # All 7 agents, W=0
        selected = list(range(POOL_SIZE))
        W = np.zeros((POOL_SIZE, POOL_SIZE))
        plan = PlanResult(
            n_agents=POOL_SIZE, W=W,
            roles=[f"分析师{i}" for i in range(POOL_SIZE)],
            rounds=1, round_mode="fixed",
            execution_order=topological_layers(W),
            reasoning="independent voting",
        )

        try:
            result = executor.execute(
                question, plan, use_stackelberg=True,
                selected_agents=selected)
            adaptive_dec = result["decision"]
        except Exception as e:
            print(f"      Exec error: {e}")
            adaptive_dec = "neutral"
            result = {"mechanism": "error", "nodes": [None]*POOL_SIZE,
                      "ic": [], "ec": [], "eq": [], "theta": [], "stances": []}

        # MajVote
        nodes = result.get("nodes", [])
        valid_nodes = [nd for nd in nodes if nd is not None]
        mv_dec = majority_vote(valid_nodes) if valid_nodes else "neutral"

        elapsed = time.time() - t0

        # Per-agent metrics
        ic_list = result.get("ic", [])
        ec_list = result.get("ec", [])
        eq_list = result.get("eq", [])
        theta_list = result.get("theta", [])
        stances = result.get("stances", [])

        for i in range(min(POOL_SIZE, len(ic_list))):
            agent_stats[i]["ic"].append(ic_list[i])
            agent_stats[i]["ec"].append(ec_list[i])
            if i < len(eq_list):
                agent_stats[i]["eq"].append(eq_list[i])
            if i < len(theta_list):
                agent_stats[i]["theta"].append(theta_list[i])
            agent_stats[i]["total"] += 1
            if i < len(stances):
                if (gt in ("yes", "Yes") and stances[i] == "yes") or \
                   (gt in ("no", "No") and stances[i] == "no"):
                    agent_stats[i]["correct"] += 1

        results.append({
            "question": question, "ground_truth": gt,
            "adaptive_decision": adaptive_dec, "majvote_decision": mv_dec,
        })

        if csv_path:
            backends = [agent_map[i] for i in range(POOL_SIZE)]
            _append_row(csv_path, {
                "dataset": dataset_name, "condition": condition,
                "question_id": str(q_idx), "question": question,
                "ground_truth": gt,
                "adaptive_decision": adaptive_dec, "majvote_decision": mv_dec,
                "per_agent_ic": json.dumps([round(x, 3) for x in ic_list]),
                "per_agent_ec": json.dumps([round(x, 3) for x in ec_list]),
                "per_agent_theta": json.dumps([round(x, 3) for x in theta_list]),
                "per_agent_stance": json.dumps(stances),
                "per_agent_backend": json.dumps(backends),
                "mechanism": result.get("mechanism", ""),
                "elapsed_s": f"{elapsed:.2f}",
            })

        flash_s = [stances[i] for i in range(min(POOL_SIZE, len(stances))) if i in flash_agents]
        ds_s = [stances[i] for i in range(min(POOL_SIZE, len(stances))) if i not in flash_agents]
        print(f"      GT={gt} Adapt={adaptive_dec} MV={mv_dec} "
              f"| ds={ds_s} flash={flash_s} | {elapsed:.1f}s")

    pool.close()
    return results, agent_stats


def print_agent_stats(agent_stats: Dict, condition: str):
    print(f"\n    Per-Agent Quality ({condition}):")
    print(f"    {'Agent':>6} {'Backend':>12} {'AvgIC':>7} {'AvgEC':>7} "
          f"{'AvgEQ':>7} {'AvgΘ':>7} {'IndAcc':>8} {'N':>4}")
    print(f"    {'-'*66}")

    flash_ics, ds_ics = [], []
    flash_ecs, ds_ecs = [], []
    flash_eqs, ds_eqs = [], []
    flash_accs, ds_accs = [], []

    for aid in sorted(agent_stats.keys()):
        s = agent_stats[aid]
        if s["total"] == 0:
            continue
        avg_ic = np.mean(s["ic"]) if s["ic"] else 0
        avg_ec = np.mean(s["ec"]) if s["ec"] else 0
        avg_eq = np.mean(s["eq"]) if s["eq"] else 0
        avg_th = np.mean(s["theta"]) if s["theta"] else 0
        acc = s["correct"] / s["total"] if s["total"] else 0
        marker = " ◄" if s["is_flash"] else ""
        print(f"    {aid:>6} {s['backend']:>12} {avg_ic:>7.3f} "
              f"{avg_ec:>7.3f} {avg_eq:>7.3f} {avg_th:>7.3f} "
              f"{acc:>7.1%} {s['total']:>4}{marker}")
        if s["is_flash"]:
            flash_ics.extend(s["ic"]); flash_ecs.extend(s["ec"])
            flash_eqs.extend(s["eq"]); flash_accs.append(acc)
        else:
            ds_ics.extend(s["ic"]); ds_ecs.extend(s["ec"])
            ds_eqs.extend(s["eq"]); ds_accs.append(acc)

    if flash_ics and ds_ics:
        print(f"\n    Group comparison:")
        print(f"      DeepSeek: IC={np.mean(ds_ics):.3f} EC={np.mean(ds_ecs):.3f} "
              f"EQ={np.mean(ds_eqs):.3f} IndAcc={np.mean(ds_accs):.1%}")
        print(f"      Flash:    IC={np.mean(flash_ics):.3f} EC={np.mean(flash_ecs):.3f} "
              f"EQ={np.mean(flash_eqs):.3f} IndAcc={np.mean(flash_accs):.1%}")
        ic_gap = np.mean(ds_ics) - np.mean(flash_ics)
        ec_gap = np.mean(ds_ecs) - np.mean(flash_ecs)
        eq_gap = np.mean(ds_eqs) - np.mean(flash_eqs)
        acc_gap = np.mean(ds_accs) - np.mean(flash_accs)
        print(f"      Gaps: IC={ic_gap:+.3f} EC={ec_gap:+.3f} "
              f"EQ={eq_gap:+.3f} IndAcc={acc_gap:+.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Heterogeneous Agent Quality Experiment (Plan A)")
    parser.add_argument("--dataset", default="StrategyQA")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--conditions", nargs="+",
                        default=["1flash", "2flash", "3flash"])
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    csv_path = args.csv or f"results/exp_hetero_{args.dataset}_{args.n_samples}q.csv"
    os.makedirs(os.path.dirname(csv_path) or "results", exist_ok=True)

    print("=" * 60)
    print("  Heterogeneous Agent Quality Experiment (Plan A)")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Pool: {POOL_SIZE} agents")
    print(f"  Conditions: {args.conditions}")
    print(f"  CSV: {csv_path}")

    questions = load_questions(args.dataset, args.n_samples)
    print(f"  Loaded {len(questions)} questions\n")

    all_results = {}

    for cond in args.conditions:
        agent_map = POOL_CONFIGS[cond]
        flash_ids = sorted(k for k, v in agent_map.items() if "flash" in v)
        ds_ids = sorted(k for k, v in agent_map.items() if v == "deepseek")
        print(f"\n  {'='*50}")
        print(f"  Condition: {cond}")
        print(f"  Flash agents: {flash_ids}")
        print(f"  DeepSeek agents: {ds_ids}")
        print(f"  {'='*50}")

        results, stats = run_condition(cond, questions, args.dataset, csv_path)
        all_results[cond] = (results, stats)

        a = evaluate_accuracy(
            [{"decision": r["adaptive_decision"], "ground_truth": r["ground_truth"]}
             for r in results])
        m = evaluate_accuracy(
            [{"decision": r["majvote_decision"], "ground_truth": r["ground_truth"]}
             for r in results])
        print(f"\n    Adaptive: {a['accuracy']:.1%} ({a['correct']}/{a['n']})")
        print(f"    MajVote:  {m['accuracy']:.1%} ({m['correct']}/{m['n']})")
        print_agent_stats(stats, cond)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY — {args.dataset}")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<12} {'Adaptive':>10} {'MajVote':>10} {'Delta':>8}")
    print(f"  {'-'*42}")
    for cond in args.conditions:
        res = all_results[cond][0]
        a = evaluate_accuracy(
            [{"decision": r["adaptive_decision"], "ground_truth": r["ground_truth"]}
             for r in res])["accuracy"]
        m = evaluate_accuracy(
            [{"decision": r["majvote_decision"], "ground_truth": r["ground_truth"]}
             for r in res])["accuracy"]
        print(f"  {cond:<12} {a:>9.1%} {m:>9.1%} {a-m:>+7.1%}")


if __name__ == "__main__":
    main()
