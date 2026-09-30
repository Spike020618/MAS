"""
Multi-Topology Experiment Runner — 统一实验入口

完整流程:
  1. collect: 收集各拓扑signals数据（需API）
  2. optimize: 每种拓扑训练θ*（离线）
  3. evaluate: 运行对比实验（离线）
  4. adaptive: 运行adaptive拓扑实验（需少量planner API调用）

Usage:
  # 完整流程（逐步执行）
  python experiments/run_multi_topology.py --step collect
  python experiments/run_multi_topology.py --step optimize
  python experiments/run_multi_topology.py --step evaluate
  python experiments/run_multi_topology.py --step adaptive

  # 仅验证所有代码路径
  python experiments/run_multi_topology.py --step verify

  # 状态检查
  python experiments/run_multi_topology.py --step status
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOLOGIES = ["parallel", "sequential", "star", "debate", "agentverse"]
FIXED_TOPOLOGIES = ["parallel", "sequential", "star", "debate"]
SIGNALS_DIR = "experiments/results/signals"
CONFIGS_DIR = "experiments/results/configs"


def check_status(args=None):
    """检查各阶段完成状态。"""
    print("=== Experiment Status ===\n")

    # Phase 1: Signals
    print("Phase 1 - Signals Data:")
    signals_ready = {}
    for ds in DATASETS:
        for topo in TOPOLOGIES:
            if topo == "parallel":
                path = os.path.join(SIGNALS_DIR, f"{ds}.json")
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")

            key = f"{ds}_{topo}"
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                n_q = len(data["questions"])
                has_signals = sum(1 for q in data["questions"] if "signals" in q)
                signals_ready[key] = True
                status = f"OK ({n_q}q, {has_signals} signals)"
            else:
                signals_ready[key] = False
                status = "MISSING"
            print(f"  {ds:>12} × {topo:<12}: {status}")
    print()

    # Phase 2: Configs
    print("Phase 2 - Optimized Configs:")
    configs_ready = {}
    for topo in FIXED_TOPOLOGIES:
        path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        configs_ready[topo] = os.path.exists(path)
        print(f"  opt_{topo}.json: {'OK' if configs_ready[topo] else 'MISSING'}")
    print()

    # Summary
    total_signals = sum(signals_ready.values())
    total_configs = sum(configs_ready.values())
    print(f"Summary: {total_signals}/{len(signals_ready)} signals, "
          f"{total_configs}/{len(configs_ready)} configs")

    if total_signals == len(signals_ready) and total_configs == len(configs_ready):
        print("All ready for evaluation!")
    elif total_signals == len(signals_ready):
        print("Signals complete. Run --step optimize next.")
    else:
        missing = [k for k, v in signals_ready.items() if not v]
        print(f"Missing signals: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        print("Run --step collect next.")


def step_collect(args):
    """Phase 1: 收集各拓扑signals。"""
    from experiments.collect_cache_topo import collect_topo_aeic, precompute_signals

    os.makedirs(os.path.join("results", "cache"), exist_ok=True)
    os.makedirs(os.path.join("results", "signals"), exist_ok=True)

    for ds in DATASETS:
        for topo in TOPOLOGIES:
            if topo == "parallel":
                path = os.path.join(SIGNALS_DIR, f"{ds}.json")
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")

            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                n_q = len(data["questions"])
                if n_q >= args.n_samples:
                    print(f"  SKIP {ds} × {topo}: already have {n_q} questions")
                    continue

            if topo == "parallel":
                print(f"  SKIP {ds} × parallel: use collect_cache.py for parallel")
                continue

            print(f"\n{'='*60}")
            print(f"Collecting: {ds} × {topo}")
            print(f"{'='*60}")

            cache_path = collect_topo_aeic(
                ds, topo, args.n_samples, args.n_agents, "results")
            print(f"\nPrecomputing signals...")
            precompute_signals(cache_path)


def step_optimize(args):
    """Phase 2: 每种拓扑独立训练θ*。"""
    from experiments.optimize_per_topology import optimize_topology

    os.makedirs(CONFIGS_DIR, exist_ok=True)

    for topo in FIXED_TOPOLOGIES:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path) and not args.force:
            print(f"  SKIP {topo}: {cfg_path} already exists (use --force to re-run)")
            continue

        print(f"\n{'='*50}")
        print(f"Optimizing θ* for: {topo}")
        print(f"{'='*50}")

        cfg, info = optimize_topology(topo, lambda_ic=0.3)
        if cfg:
            cfg.save(cfg_path)
            print(f"  Result: acc={info['accuracy']:.1%}, IC={info['ic_metric']:.1%}")


def step_evaluate(args):
    """Phase 3: 运行对比实验。"""
    from experiments.baselines_real import run_full_comparison, print_results

    results = run_full_comparison()
    print_results(results)

    out_path = "experiments/results/comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def step_adaptive(args):
    """Phase 4: Adaptive拓扑实验。"""
    from experiments.core.llm import MultiLLMPool
    from experiments.core.planner import TopologyPlanner
    from experiments.core.similarity import SimilarityCalculator
    from experiments.core.adaptive_executor import classify_topology
    from experiments.core.consensus_config import ConsensusConfig
    from experiments.exp_orchestrator import load_questions
    from experiments.optimize_consensus import evaluate_config

    pool = MultiLLMPool()
    planner = TopologyPlanner(pool.get_client(0))

    # 加载所有拓扑的θ*
    configs = {}
    for topo in FIXED_TOPOLOGIES:
        cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        if os.path.exists(cfg_path):
            configs[topo] = ConsensusConfig.load(cfg_path)
        else:
            configs[topo] = ConsensusConfig()

    # 加载所有已有的signals数据
    all_signals = {}
    for ds in DATASETS:
        all_signals[ds] = {}
        for topo in FIXED_TOPOLOGIES:
            if topo == "parallel":
                path = os.path.join(SIGNALS_DIR, f"{ds}.json")
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_signals[ds][topo] = json.load(f)

    # Adaptive: planner为每题选拓扑，查找对应signals+θ*
    cache_path = "experiments/results/cache/adaptive_classification.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            classification_cache = json.load(f)
    else:
        classification_cache = {}

    results = {}
    for ds in DATASETS:
        print(f"\n--- {ds} ---")
        questions = load_questions(ds, args.n_samples)
        correct = 0
        total = 0
        topo_counts = {}

        for q_idx, q_item in enumerate(questions):
            cache_key = f"{ds}_{q_idx}"

            if cache_key in classification_cache:
                chosen_topo = classification_cache[cache_key]
            else:
                plan = planner.plan(q_item["question"], n_agents=5)
                chosen_topo = classify_topology(plan)
                classification_cache[cache_key] = chosen_topo
                # 增量保存
                if (q_idx + 1) % 10 == 0:
                    with open(cache_path, "w") as f:
                        json.dump(classification_cache, f)

            topo_counts[chosen_topo] = topo_counts.get(chosen_topo, 0) + 1

            # 从该拓扑的signals中查找这题的结果
            if chosen_topo in all_signals[ds]:
                sig_data = all_signals[ds][chosen_topo]
                matching = [q for q in sig_data["questions"]
                            if q["question_id"] == q_idx and "signals" in q]
                if matching:
                    cfg = configs[chosen_topo]
                    n_agents = sig_data["meta"]["n_agents"]
                    eval_result = evaluate_config(matching, cfg, n_agents)
                    if eval_result["correct"] > 0:
                        correct += 1
                    total += 1
                    continue

            # fallback: 用parallel
            if "parallel" in all_signals[ds]:
                sig_data = all_signals[ds]["parallel"]
                matching = [q for q in sig_data["questions"]
                            if q["question_id"] == q_idx and "signals" in q]
                if matching:
                    cfg = configs["parallel"]
                    n_agents = sig_data["meta"]["n_agents"]
                    eval_result = evaluate_config(matching, cfg, n_agents)
                    if eval_result["correct"] > 0:
                        correct += 1
                    total += 1

        acc = correct / total if total > 0 else 0
        results[ds] = {"accuracy": acc, "correct": correct, "total": total,
                       "topology_distribution": topo_counts}
        print(f"  Accuracy: {acc:.1%} ({correct}/{total})")
        print(f"  Topology distribution: {topo_counts}")

    # 保存分类缓存
    with open(cache_path, "w") as f:
        json.dump(classification_cache, f, indent=2)

    pool.close()

    out_path = "experiments/results/adaptive_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAdaptive results saved to {out_path}")


def step_verify(args=None):
    """验证所有代码路径。"""
    print("=== Full Verification ===\n")

    print("[1] collect_cache_topo.py...")
    from experiments.collect_cache_topo import verify_code
    ok1 = verify_code("StrategyQA", "debate", 5)

    print(f"\n[2] optimize_per_topology.py...")
    from experiments.optimize_per_topology import verify
    ok2 = verify()

    print(f"\n[3] baselines_real.py...")
    from experiments.baselines_real import verify as verify_bl
    ok3 = verify_bl()

    print(f"\n[4] adaptive_executor.py...")
    from experiments.core.adaptive_executor import AdaptiveExecutor, classify_topology
    from experiments.core.planner import PlanResult
    W_test = np.zeros((5, 5))
    plan_test = PlanResult(
        n_agents=5, W=W_test, roles=["A"]*5, rounds=1,
        round_mode="fixed", execution_order=[[0,1,2,3,4]])
    ct = classify_topology(plan_test)
    print(f"  classify_topology(W=zeros) -> '{ct}'")
    assert ct == "parallel"
    from experiments.core.topology import make_topology
    W_test2 = make_topology("star", 5)
    plan_test2 = PlanResult(
        n_agents=5, W=W_test2, roles=["A"]*5, rounds=1,
        round_mode="fixed", execution_order=[[1,2,3,4],[0]])
    ct2 = classify_topology(plan_test2)
    print(f"  classify_topology(W=star-like) -> '{ct2}'")
    assert ct2 == "star"
    plan_test3 = PlanResult(
        n_agents=5, W=np.ones((5,5))-np.eye(5), roles=["A"]*5, rounds=3,
        round_mode="fixed", execution_order=[[0,1,2,3,4]])
    ct3 = classify_topology(plan_test3)
    print(f"  classify_topology(rounds=3) -> '{ct3}'")
    assert ct3 == "debate"
    print("  All classify_topology tests passed")

    print(f"\n=== Verification {'PASSED' if (ok1 and ok2) else 'PARTIAL'} ===")


def main():
    parser = argparse.ArgumentParser(description="Multi-Topology Experiment")
    parser.add_argument("--step", required=True,
                        choices=["collect", "optimize", "evaluate",
                                 "adaptive", "verify", "status"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if output exists")
    args = parser.parse_args()

    steps = {
        "status": check_status,
        "verify": step_verify,
        "collect": step_collect,
        "optimize": step_optimize,
        "evaluate": step_evaluate,
        "adaptive": step_adaptive,
    }
    steps[args.step](args)


if __name__ == "__main__":
    main()
