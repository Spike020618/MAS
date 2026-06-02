"""
Experiment: Adaptive Topology Planning (Set 2)

对比:
  - Adaptive: Planner自由生成拓扑 + Option B共识
  - Random-W: 随机拓扑 + Option B共识 (证明Planner有价值)
  - MajVote-Adaptive: Planner拓扑 + 简单多数投票 (证明共识机制有价值)
  - Best-Fixed: Set 1中4种固定模式的事后最优 (upper bound参考)

Usage:
  python experiments/exp_adaptive.py --dataset StrategyQA --n_samples 10
  python experiments/exp_adaptive.py --dataset StrategyQA --n_samples 20 --all_datasets
  python experiments/exp_adaptive.py --dataset StrategyQA --n_samples 20 --csv results/set2.csv
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
from core.planner import TopologyPlanner, PlanResult, random_plan
from core.adaptive_executor import AdaptiveExecutor
from core.similarity import SimilarityCalculator
from core.trust import extract_stance, majority_vote
from experiments.exp_orchestrator import load_questions, evaluate_accuracy


# ─── CSV断点续跑 ────────────────────────────────────────────────────

SET2_FIELDS = [
    "dataset", "method", "n_agents", "question_id",
    "question", "ground_truth", "decision", "mechanism",
    "plan_reasoning", "W_edges", "rounds", "execution_layers",
    "stances", "ic", "ec", "trust", "depth_bonus", "elapsed_s",
    "trace",
]


def _load_completed_set2(csv_path: str) -> Set[Tuple]:
    """加载已完成的 (dataset, method, n_agents, question_id) 组合。"""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["method"], row["n_agents"],
                   row["question_id"])
            completed.add(key)
    return completed


def _append_csv_row_set2(csv_path: str, row: Dict):
    """追加一行到CSV（首次自动写header）。"""
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SET2_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _serialize(lst) -> str:
    if lst is None:
        return ""
    if isinstance(lst, np.ndarray):
        lst = lst.tolist()
    return json.dumps(lst, ensure_ascii=False)


# ─── 实验主逻辑 ─────────────────────────────────────────────────────

def run_adaptive_experiment(
    questions: List[Dict],
    n_agents: int = 5,
    sim_method: str = "embedding",
    csv_path: str = None,
    dataset_name: str = "",
) -> Dict[str, List[Dict]]:
    """运行自适应拓扑实验。"""
    completed = _load_completed_set2(csv_path) if csv_path else set()
    if completed:
        print(f"  CSV checkpoint: {len(completed)} rows already completed")

    planner_llm = LLMClient(backend="deepseek")
    planner = TopologyPlanner(planner_llm)

    pool = MultiLLMPool()
    sim = SimilarityCalculator(method=sim_method)
    executor = AdaptiveExecutor(pool, sim, n_agents=n_agents)

    methods_results = {
        "Adaptive": [],
        "Random-W": [],
        "MajVote-Adapt": [],
    }

    for q_idx, q_item in enumerate(questions):
        question = q_item["question"]
        gt = q_item["ground_truth"]

        # 断点续跑: 检查该题所有method是否都已完成
        if csv_path:
            all_done = all(
                (dataset_name, method, str(n_agents), str(q_idx))
                in completed
                for method in ["Adaptive", "Random-W", "MajVote-Adapt"]
            )
            if all_done:
                continue

        print(f"\n  [{q_idx+1}/{len(questions)}] {question[:55]}...")
        t0 = time.time()

        # 1. Planner generates topology
        _tp0 = time.time()
        try:
            plan = planner.plan(question, n_agents=n_agents)
        except Exception as e:
            print(f"    Planner error: {e}, using fallback")
            plan = planner._fallback_plan(n_agents, question)
        planner_time = time.time() - _tp0

        plan_info = (f"n={plan.n_agents}, "
                     f"W_edges={int(plan.W.sum())}, "
                     f"rounds={plan.rounds}, "
                     f"layers={len(plan.execution_order)}")
        print(f"    Plan: {plan.reasoning[:40]} | {plan_info}")

        # 2. Adaptive: execute with planner topology + Option B
        try:
            result_adaptive = executor.execute(question, plan,
                                               use_stackelberg=True)
            adaptive_decision = result_adaptive["decision"]
        except Exception as e:
            print(f"    Adaptive error: {e}")
            adaptive_decision = "neutral"
            result_adaptive = {"mechanism": "error"}

        methods_results["Adaptive"].append({
            "question": question,
            "ground_truth": gt,
            "decision": adaptive_decision,
            "mechanism": result_adaptive.get("mechanism", ""),
            "plan_reasoning": plan.reasoning,
            "W_edges": int(plan.W.sum()),
            "rounds": plan.rounds,
        })

        # 3. Random-W: random topology with same n_agents as adaptive plan
        try:
            rplan = random_plan(plan.n_agents, seed=q_idx)
            result_random = executor.execute(question, rplan,
                                             use_stackelberg=True)
            random_decision = result_random["decision"]
        except Exception as e:
            print(f"    Random error: {e}")
            random_decision = "neutral"
            result_random = {"mechanism": "error"}

        methods_results["Random-W"].append({
            "question": question,
            "ground_truth": gt,
            "decision": random_decision,
            "mechanism": result_random.get("mechanism", ""),
        })

        # 4. MajVote-Adaptive: planner topology + simple majority vote
        nodes = result_adaptive.get("nodes", [])
        if nodes and any(nd is not None for nd in nodes):
            valid_nodes = [nd for nd in nodes if nd is not None]
            mv_decision = majority_vote(valid_nodes)
        else:
            mv_decision = "neutral"

        methods_results["MajVote-Adapt"].append({
            "question": question,
            "ground_truth": gt,
            "decision": mv_decision,
            "mechanism": "majority",
        })

        elapsed = time.time() - t0

        # CSV即时写入
        if csv_path:
            # 构建trace JSON
            agent_times = result_adaptive.get("agent_times", [0.0] * n_agents)
            consensus_time = result_adaptive.get("consensus_time_s", 0)
            backends = pool.backends if hasattr(pool, 'backends') else ["deepseek"]
            trace_adaptive = {
                "agents": [
                    {
                        "id": i,
                        "role": plan.roles[i] if i < len(plan.roles) else f"agent-{i}",
                        "persona": AGENT_PERSONAS[i % len(AGENT_PERSONAS)]["name"],
                        "model": backends[i % len(backends)] if backends else "unknown",
                        "llm_time_s": round(agent_times[i], 2) if i < len(agent_times) else 0,
                        "stance": result_adaptive.get("stances", [])[i] if i < len(result_adaptive.get("stances", [])) else "",
                        "ic": round(result_adaptive.get("ic", [])[i], 3) if i < len(result_adaptive.get("ic", [])) else 0,
                        "ec": round(result_adaptive.get("ec", [])[i], 3) if i < len(result_adaptive.get("ec", [])) else 0,
                    }
                    for i in range(n_agents)
                ],
                "W": plan.W.tolist(),
                "roles": plan.roles,
                "planner_time_s": round(planner_time, 2),
                "total_llm_time_s": round(sum(agent_times), 2),
                "consensus_time_s": round(consensus_time, 3),
                "overhead_s": round(max(0, elapsed - planner_time - sum(agent_times) - consensus_time), 2),
            }
            trace_json = json.dumps(trace_adaptive, ensure_ascii=False)

            base_row = {
                "dataset": dataset_name,
                "n_agents": str(n_agents),
                "question_id": str(q_idx),
                "question": question,
                "ground_truth": gt,
                "elapsed_s": f"{elapsed:.2f}",
            }
            # Adaptive
            _append_csv_row_set2(csv_path, {
                **base_row,
                "method": "Adaptive",
                "decision": adaptive_decision,
                "mechanism": result_adaptive.get("mechanism", ""),
                "plan_reasoning": plan.reasoning,
                "W_edges": str(int(plan.W.sum())),
                "rounds": str(plan.rounds),
                "execution_layers": str(len(plan.execution_order)),
                "stances": _serialize(result_adaptive.get("stances", [])),
                "ic": _serialize(result_adaptive.get("ic", [])),
                "ec": _serialize(result_adaptive.get("ec", [])),
                "trust": _serialize(result_adaptive.get("trust", [])),
                "depth_bonus": _serialize(result_adaptive.get("depth_bonus", [])),
                "trace": trace_json,
            })
            # Random-W
            _append_csv_row_set2(csv_path, {
                **base_row,
                "method": "Random-W",
                "decision": random_decision,
                "mechanism": result_random.get("mechanism", ""),
                "plan_reasoning": "random topology baseline",
                "W_edges": str(int(rplan.W.sum())),
                "rounds": str(rplan.rounds),
                "execution_layers": str(len(rplan.execution_order)),
                "stances": _serialize(result_random.get("stances", [])),
                "ic": _serialize(result_random.get("ic", [])),
                "ec": _serialize(result_random.get("ec", [])),
                "trust": _serialize(result_random.get("trust", [])),
                "depth_bonus": _serialize(result_random.get("depth_bonus", [])),
                "trace": "",
            })
            # MajVote-Adapt
            _append_csv_row_set2(csv_path, {
                **base_row,
                "method": "MajVote-Adapt",
                "decision": mv_decision,
                "mechanism": "majority",
                "plan_reasoning": plan.reasoning,
                "W_edges": str(int(plan.W.sum())),
                "rounds": str(plan.rounds),
                "execution_layers": str(len(plan.execution_order)),
                "stances": _serialize([extract_stance(nd.get("C", ""))
                                       for nd in nodes if nd]),
                "ic": "", "ec": "", "trust": "", "depth_bonus": "",
                "trace": "",
            })

        print(f"    GT={gt} | Adapt={adaptive_decision} "
              f"| Rand={random_decision} | MV={mv_decision} | {elapsed:.1f}s")

    pool.close()
    planner_llm.close()
    return methods_results


def print_summary(results: Dict[str, List[Dict]], dataset: str):
    """Print accuracy summary."""
    print(f"\n{'='*60}")
    print(f"  ADAPTIVE EXPERIMENT RESULTS — {dataset}")
    print(f"{'='*60}")

    methods = list(results.keys())
    header = "  " + "".join(f"{m:<16}" for m in methods)
    print(f"\n{header}")
    print(f"  {'-'*len(header)}")

    row = "  "
    for method in methods:
        metrics = evaluate_accuracy(results[method])
        row += f"{metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['n']})  "
    print(row)

    # Plan analysis
    adaptive_results = results.get("Adaptive", [])
    if adaptive_results:
        print(f"\n  Plan Analysis:")
        reasonings = [r.get("plan_reasoning", "") for r in adaptive_results]
        edges = [r.get("W_edges", 0) for r in adaptive_results]
        rounds = [r.get("rounds", 1) for r in adaptive_results]
        print(f"    Avg W edges: {np.mean(edges):.1f}")
        print(f"    Avg rounds: {np.mean(rounds):.1f}")
        print(f"    Sample plans:")
        for r in adaptive_results[:3]:
            print(f"      Q: {r['question'][:40]}...")
            print(f"        Plan: {r.get('plan_reasoning', '')[:60]}")
            print(f"        Result: {r['decision']} (GT={r['ground_truth']})")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Topology Experiment (Set 2)")
    parser.add_argument("--dataset", default="StrategyQA")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--sim", default="embedding")
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv", default=None,
                        help="CSV output path (enables checkpoint/resume)")
    parser.add_argument("--all_datasets", action="store_true",
                        help="Run on all 8 datasets")
    args = parser.parse_args()

    datasets = (
        ["StrategyQA", "TruthfulQA", "HotpotQA", "GPQA",
         "MMLU", "GSM8K", "ARC-Challenge", "CommonsenseQA"]
        if args.all_datasets else [args.dataset]
    )

    print("=" * 60)
    print("  Adaptive Topology Experiment (Set 2)")
    print("=" * 60)
    print(f"  Datasets: {datasets}")
    print(f"  Samples/dataset: {args.n_samples}")
    print(f"  Agents: {args.n_agents}")
    print(f"  Similarity: {args.sim}")
    if args.csv:
        print(f"  CSV output: {args.csv} (checkpoint/resume enabled)")

    csv_path = args.csv
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or "results", exist_ok=True)

    all_results = {}

    for dataset in datasets:
        print(f"\n\n{'#'*60}")
        print(f"  Dataset: {dataset}")
        print(f"{'#'*60}")

        questions = load_questions(dataset, args.n_samples)
        print(f"  Loaded {len(questions)} questions")

        results = run_adaptive_experiment(
            questions, n_agents=args.n_agents, sim_method=args.sim,
            csv_path=csv_path, dataset_name=dataset)
        print_summary(results, dataset)
        all_results[dataset] = results

    # Save results
    output_path = args.output or f"results/exp_adaptive_{args.n_samples}q.json"
    os.makedirs(os.path.dirname(output_path) or "results", exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=_convert)
    print(f"\n\n  All results saved to: {output_path}")

    # Final cross-dataset summary
    if len(datasets) > 1:
        print(f"\n\n{'='*70}")
        print(f"  CROSS-DATASET SUMMARY")
        print(f"{'='*70}")
        methods = ["Adaptive", "Random-W", "MajVote-Adapt"]
        header = f"  {'Dataset':<16}" + "".join(f"{m:<16}" for m in methods)
        print(f"\n{header}")
        print(f"  {'-'*64}")
        for ds in datasets:
            row = f"  {ds:<16}"
            for method in methods:
                metrics = evaluate_accuracy(all_results[ds][method])
                row += f"{metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['n']})  "
            print(row)

        # Average
        print(f"  {'-'*64}")
        row = f"  {'AVERAGE':<16}"
        for method in methods:
            accs = [evaluate_accuracy(all_results[ds][method])["accuracy"]
                    for ds in datasets]
            row += f"{np.mean(accs):.1%}              "
        print(row)


if __name__ == "__main__":
    main()
