"""
Experiment: Adaptive Heterogeneous — 纯Stackelberg + 跨问题信任

Agent分配: id小→deepseek, id大→flash, flash不超过半数
9组条件:
  3a-0f, 3a-1f
  5a-0f, 5a-1f, 5a-2f
  7a-0f, 7a-1f, 7a-2f, 7a-3f

每组:
  全量agent运行 → TrustTracker选择子集 → 纯θ加权投票
  同时记录MajVote(全量)作为baseline

Usage:
  python experiments/exp_adaptive.py --dataset StrategyQA --n_samples 10
  python experiments/exp_adaptive.py --dataset TruthfulQA --n_samples 10 \
      --conditions 7a-0f 7a-1f 7a-2f 7a-3f
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

from core.llm import MultiLLMPool, AGENT_PERSONAS
from core.planner import PlanResult, topological_layers
from core.adaptive_executor import AdaptiveExecutor
from core.similarity import SimilarityCalculator
from core.trust import extract_stance, majority_vote
from core.trust_tracker import TrustTracker
from experiments.exp_orchestrator import load_questions, evaluate_accuracy

# ── 统一CSV格式 ──────────────────────────────────────────────────

UNIFIED_FIELDS = [
    "dataset", "condition", "question_id",
    "question", "ground_truth", "decision", "mechanism",
    "n_agents_used", "selected_agents",
    "per_agent_stance", "per_agent_ic", "per_agent_ec", "per_agent_eq",
    "per_agent_theta", "per_agent_backend",
    "trust_scores", "elapsed_s",
]


def _append_unified_row(csv_path: str, row: Dict):
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_completed(csv_path: str) -> Set[Tuple]:
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add((row["dataset"], row["condition"], row["question_id"]))
    return completed


# ── 9组条件定义 ──────────────────────────────────────────────────

def build_agent_map(pool_size: int, n_flash: int) -> Dict[int, str]:
    agent_map = {}
    n_ds = pool_size - n_flash
    for i in range(pool_size):
        agent_map[i] = "deepseek" if i < n_ds else "zhipu_flash"
    return agent_map

ALL_CONDITIONS = {
    "3a-0f": (3, 0), "3a-1f": (3, 1),
    "5a-0f": (5, 0), "5a-1f": (5, 1), "5a-2f": (5, 2),
    "7a-0f": (7, 0), "7a-1f": (7, 1), "7a-2f": (7, 2), "7a-3f": (7, 3),
}


# ── 单条件运行 ───────────────────────────────────────────────────

def run_condition(
    condition: str,
    questions: List[Dict],
    dataset_name: str,
    csv_dir: str,
) -> List[Dict]:

    pool_size, n_flash = ALL_CONDITIONS[condition]
    agent_map = build_agent_map(pool_size, n_flash)

    sg_path = os.path.join(csv_dir, f"adaptive_AEIC_SG_{condition}_{dataset_name}.csv")
    mv_path = os.path.join(csv_dir, f"adaptive_MajVote_{condition}_{dataset_name}.csv")

    completed_sg = _load_completed(sg_path)
    completed_mv = _load_completed(mv_path)

    pool = MultiLLMPool(agent_map=agent_map)
    sim = SimilarityCalculator(method="embedding")
    executor = AdaptiveExecutor(pool, sim, n_agents=pool_size)
    tracker = TrustTracker(n_agents=pool_size, warmup=3)

    results = []

    for q_idx, q_item in enumerate(questions):
        question = q_item["question"]
        gt = q_item["ground_truth"]

        already_done = (
            (dataset_name, condition, str(q_idx)) in completed_sg and
            (dataset_name, condition, str(q_idx)) in completed_mv
        )
        if already_done:
            continue

        print(f"    [{q_idx+1}/{len(questions)}] {question[:50]}...")
        t0 = time.time()

        # ① 全量agent运行 (W=0, independent)
        all_agents = list(range(pool_size))
        W_full = np.zeros((pool_size, pool_size))
        plan_full = PlanResult(
            n_agents=pool_size, W=W_full,
            roles=[f"分析师{i}" for i in range(pool_size)],
            rounds=1, round_mode="fixed",
            execution_order=topological_layers(W_full),
            reasoning="full pool independent",
        )

        try:
            result_full = executor.execute(
                question, plan_full, use_stackelberg=True,
                selected_agents=all_agents)
        except Exception as e:
            print(f"      Full exec error: {e}")
            result_full = {
                "nodes": [None] * pool_size, "mechanism": "error",
                "ic": [], "ec": [], "eq": [], "ev": [],
                "theta": [], "stances": [],
            }

        nodes_full = result_full.get("nodes", [])
        ic_full = result_full.get("ic", [])
        ec_full = result_full.get("ec", [])
        eq_full = result_full.get("eq", [])
        theta_full = result_full.get("theta", [])
        stances_full = result_full.get("stances", [])

        # ② TrustTracker选择子集
        selected = tracker.select_agents(max_n=pool_size, min_n=2)

        # ③ 子集共识
        if len(selected) == 1:
            idx = selected[0]
            adaptive_dec = stances_full[idx] if idx < len(stances_full) else "neutral"
            mechanism = "single_agent"
        elif len(selected) >= 2:
            subset_nodes = [nodes_full[i] for i in selected
                            if i < len(nodes_full) and nodes_full[i] is not None]
            if len(subset_nodes) >= 2:
                n_sub = len(subset_nodes)
                W_sub = np.zeros((n_sub, n_sub))
                plan_sub = PlanResult(
                    n_agents=n_sub, W=W_sub,
                    roles=[f"分析师{selected[j]}" for j in range(n_sub)],
                    rounds=1, round_mode="fixed",
                    execution_order=topological_layers(W_sub),
                    reasoning="trust-selected subset",
                )
                sub_result = AdaptiveExecutor(pool, sim, n_agents=n_sub) \
                    ._consensus_decision(subset_nodes, plan_sub, use_stackelberg=True)
                adaptive_dec = sub_result["decision"]
                mechanism = sub_result["mechanism"]
            elif len(subset_nodes) == 1:
                adaptive_dec = extract_stance(subset_nodes[0].get("C", ""))
                mechanism = "single_agent"
            else:
                adaptive_dec = "neutral"
                mechanism = "no_valid_nodes"
        else:
            adaptive_dec = "neutral"
            mechanism = "no_selection"

        # ④ MajVote baseline (全量)
        valid_full = [nd for nd in nodes_full if nd is not None]
        mv_dec = majority_vote(valid_full) if valid_full else "neutral"

        # ⑤ 更新TrustTracker
        for i in range(pool_size):
            theta_i = theta_full[i] if i < len(theta_full) else 0.5
            ic_i = ic_full[i] if i < len(ic_full) else 0.5
            ec_i = ec_full[i] if i < len(ec_full) else 0.5
            eq_i = eq_full[i] if i < len(eq_full) else 0.5
            stance_i = stances_full[i] if i < len(stances_full) else "neutral"
            tracker.update(i, theta_i, ic_i, ec_i, eq_i,
                           agreed_with_consensus=(stance_i == adaptive_dec))
        tracker.increment_question()

        elapsed = time.time() - t0
        backends = [agent_map[i] for i in range(pool_size)]
        trust_now = tracker.get_all_scores()

        results.append({
            "question": question, "ground_truth": gt,
            "adaptive_decision": adaptive_dec, "majvote_decision": mv_dec,
        })

        # 共享字段
        base = {
            "dataset": dataset_name, "condition": condition,
            "question_id": str(q_idx),
            "question": question, "ground_truth": gt,
            "per_agent_stance": json.dumps(stances_full),
            "per_agent_ic": json.dumps([round(x, 3) for x in ic_full]),
            "per_agent_ec": json.dumps([round(x, 3) for x in ec_full]),
            "per_agent_eq": json.dumps([round(x, 3) for x in eq_full]),
            "per_agent_theta": json.dumps([round(x, 3) for x in theta_full]),
            "per_agent_backend": json.dumps(backends),
            "trust_scores": json.dumps(trust_now),
            "elapsed_s": f"{elapsed:.2f}",
        }

        # AEIC+SG file
        _append_unified_row(sg_path, {
            **base,
            "decision": adaptive_dec, "mechanism": mechanism,
            "n_agents_used": str(len(selected)),
            "selected_agents": json.dumps(selected),
        })
        # MajVote file
        _append_unified_row(mv_path, {
            **base,
            "decision": mv_dec, "mechanism": "majority",
            "n_agents_used": str(pool_size),
            "selected_agents": json.dumps(all_agents),
        })

        sel_str = ",".join(str(s) for s in selected)
        print(f"      GT={gt} Adapt={adaptive_dec} MV={mv_dec} "
              f"| sel=[{sel_str}] n={len(selected)} | {elapsed:.1f}s")

    pool.close()
    return results


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Heterogeneous Experiment (Pure Stackelberg + Trust)")
    parser.add_argument("--dataset", default="StrategyQA")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Conditions to run (default: all 9)")
    parser.add_argument("--csv_dir", default="results",
                        help="Output directory for CSV files")
    args = parser.parse_args()

    conditions = args.conditions or sorted(ALL_CONDITIONS.keys())
    for c in conditions:
        if c not in ALL_CONDITIONS:
            print(f"Unknown condition: {c}")
            print(f"Available: {sorted(ALL_CONDITIONS.keys())}")
            sys.exit(1)

    os.makedirs(args.csv_dir, exist_ok=True)

    print("=" * 60)
    print("  Adaptive Heterogeneous Experiment")
    print("  Pure Stackelberg θ + Cross-Question Trust")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Conditions: {conditions}")
    print(f"  Output: {args.csv_dir}/adaptive_{{method}}_{{cond}}_{args.dataset}.csv")

    questions = load_questions(args.dataset, args.n_samples)
    print(f"  Loaded {len(questions)} questions\n")

    all_results = {}

    for cond in conditions:
        pool_size, n_flash = ALL_CONDITIONS[cond]
        agent_map = build_agent_map(pool_size, n_flash)
        flash_ids = sorted(k for k, v in agent_map.items() if "flash" in v)
        ds_ids = sorted(k for k, v in agent_map.items() if v == "deepseek")

        print(f"\n  {'='*50}")
        print(f"  Condition: {cond} (pool={pool_size}, flash={n_flash})")
        print(f"  DeepSeek: {ds_ids}  Flash: {flash_ids}")
        print(f"  {'='*50}")

        results = run_condition(cond, questions, args.dataset, args.csv_dir)
        all_results[cond] = results

        a = evaluate_accuracy(
            [{"decision": r["adaptive_decision"], "ground_truth": r["ground_truth"]}
             for r in results])
        m = evaluate_accuracy(
            [{"decision": r["majvote_decision"], "ground_truth": r["ground_truth"]}
             for r in results])
        print(f"\n    AEIC+SG:  {a['accuracy']:.1%} ({a['correct']}/{a['n']})")
        print(f"    MajVote:  {m['accuracy']:.1%} ({m['correct']}/{m['n']})")

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY — {args.dataset}")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<12} {'Pool':>5} {'Flash':>6} {'AEIC+SG':>10} {'MajVote':>10} {'Delta':>8}")
    print(f"  {'-'*55}")
    for cond in conditions:
        pool_size, n_flash = ALL_CONDITIONS[cond]
        res = all_results[cond]
        a = evaluate_accuracy(
            [{"decision": r["adaptive_decision"], "ground_truth": r["ground_truth"]}
             for r in res])["accuracy"]
        m = evaluate_accuracy(
            [{"decision": r["majvote_decision"], "ground_truth": r["ground_truth"]}
             for r in res])["accuracy"]
        print(f"  {cond:<12} {pool_size:>5} {n_flash:>6} {a:>9.1%} {m:>9.1%} {a-m:>+7.1%}")


if __name__ == "__main__":
    main()
