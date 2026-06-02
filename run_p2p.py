"""
P2P Multi-Agent Consensus System — 启动器

启动注册中心 + N个独立Agent节点，执行共识任务。

Usage:
  # 单个问题测试
  python run_p2p.py --n_agents 5 --question "Is the sky blue?" --mode parallel

  # 全量实验
  python run_p2p.py --n_agents 5 --experiment --dataset StrategyQA --n_samples 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.p2p.registry import Registry
from core.p2p.agent_node import AgentNode, make_agent_nodes


def start_system(n_agents: int = 5, base_port: int = 8001,
                 registry_port: int = 8000, sim_method: str = "embedding"):
    """启动完整P2P系统: Registry + N个Agent节点。"""
    # 1. Start Registry
    registry = Registry(port=registry_port, sim_method=sim_method)
    registry.start()
    print(f"[Registry] Running on port {registry_port}")

    # 2. Start Agent Nodes
    nodes = make_agent_nodes(n_agents, base_port=base_port)
    for node in nodes:
        node.start()
        # Register with registry
        registry.agents.append({
            "agent_id": node.agent_id,
            "url": node.url,
            "backend": node.backend,
            "persona": node.persona_name,
        })
    time.sleep(0.5)

    print(f"[Agents] {n_agents} agents started:")
    for node in nodes:
        print(f"  Agent-{node.agent_id} @ port {node.port} | {node.backend} | {node.persona_name}")
    print()

    return registry, nodes


def stop_system(registry, nodes):
    """停止所有服务。"""
    for node in nodes:
        node.stop()
    registry.stop()


def run_single_question(registry, question: str, mode: str = "parallel",
                        use_stackelberg: bool = True) -> dict:
    """通过P2P系统处理单个问题。"""
    result = registry.submit_task(question, mode=mode, use_stackelberg=use_stackelberg)
    return result


def run_experiment(registry, dataset_name: str, n_samples: int,
                   modes: list, output_path: str = None):
    """在P2P系统上跑完整实验。"""
    from experiments.exp_orchestrator import load_questions, evaluate_accuracy
    from core.trust import evaluate_and_decide, majority_vote, quality_weighted_vote, extract_stance
    import numpy as np
    from core.topology import make_topology

    questions = load_questions(dataset_name, n_samples)
    print(f"Loaded {len(questions)} questions from {dataset_name}")

    n_agents = len(registry.agents)
    all_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()} ({n_agents} agents, P2P)")
        print(f"{'='*60}")

        mode_results = {"AEIC+SG": [], "AEIC": [], "QualWt": [], "MajVote": []}

        for q_idx, q_item in enumerate(questions):
            question = q_item["question"]
            gt = q_item["ground_truth"]

            print(f"  [{q_idx+1}/{len(questions)}] {question[:55]}...", end="", flush=True)
            t0 = time.time()

            # Run through P2P system
            result_sg = registry.submit_task(question, mode=mode, use_stackelberg=True)
            elapsed = time.time() - t0
            nodes = result_sg["nodes"]

            # AEIC+SG
            mode_results["AEIC+SG"].append({
                "question": question, "ground_truth": gt,
                "decision": result_sg["decision"],
                "mechanism": result_sg["mechanism"],
            })

            # AEIC (reuse nodes, no Stackelberg)
            topo_map = {"parallel": "complete", "sequential": "path",
                        "star": "star", "debate": "complete"}
            W = make_topology(topo_map.get(mode, "complete"), n_agents)
            result_no_sg = evaluate_and_decide(nodes, W, registry.sim_calc,
                                               mode=mode, use_stackelberg=False)
            mode_results["AEIC"].append({
                "question": question, "ground_truth": gt,
                "decision": result_no_sg["decision"],
                "mechanism": result_no_sg["mechanism"],
            })

            # QualWt
            qw_decision, _ = quality_weighted_vote(nodes, registry.sim_calc)
            mode_results["QualWt"].append({
                "question": question, "ground_truth": gt,
                "decision": qw_decision,
            })

            # MajVote
            mv_decision = majority_vote(nodes)
            mode_results["MajVote"].append({
                "question": question, "ground_truth": gt,
                "decision": mv_decision,
            })

            print(f" GT={gt} | SG={result_sg['decision']} | {elapsed:.1f}s")

        all_results[mode] = mode_results

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"  RESULTS SUMMARY (P2P, {n_agents} agents)")
    print(f"{'='*70}")
    methods = ["MajVote", "QualWt", "AEIC", "AEIC+SG"]
    header = f"{'Mode':<12}" + "".join(f"{m:<14}" for m in methods)
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")
    for mode in modes:
        row = f"  {mode:<12}"
        for m in methods:
            results = all_results[mode][m]
            metrics = evaluate_accuracy(results)
            row += f"{metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['n']})  "
        print(row)

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or "results", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="P2P Multi-Agent Consensus System")
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--base_port", type=int, default=8001)
    parser.add_argument("--registry_port", type=int, default=8000)
    parser.add_argument("--sim", default="embedding")

    # Single question mode
    parser.add_argument("--question", default=None, help="Single question to process")
    parser.add_argument("--mode", default="parallel")

    # Experiment mode
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--dataset", default="StrategyQA")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--modes", default="parallel,sequential,star,debate")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    # Start P2P system
    print("=" * 50)
    print(" P2P Multi-Agent Consensus System")
    print("=" * 50)
    registry, nodes = start_system(
        n_agents=args.n_agents,
        base_port=args.base_port,
        registry_port=args.registry_port,
        sim_method=args.sim,
    )

    try:
        if args.question:
            # Single question
            print(f"Question: {args.question}")
            print(f"Mode: {args.mode}")
            print()
            result = run_single_question(registry, args.question, mode=args.mode)
            print(f"Decision: {result['decision']} ({result['mechanism']})")
            print(f"Stances: {result.get('stances', [])}")
            print(f"Trust: {[f'{t:.3f}' for t in result.get('trust', [])]}")

        elif args.experiment:
            # Full experiment
            modes = [m.strip() for m in args.modes.split(",")]
            output = args.output or f"results/exp_p2p_{args.dataset}_{args.n_samples}q.json"
            run_experiment(registry, args.dataset, args.n_samples, modes, output)

        else:
            # Interactive: keep running, accept curl requests
            print("System ready. Submit tasks via:")
            print(f'  curl -X POST http://localhost:{args.registry_port}/task \\')
            print(f'    -H "Content-Type: application/json" \\')
            print(f'    -d \'{{"question": "Is the sky blue?", "mode": "parallel", "sync": true}}\'')
            print("\nPress Ctrl+C to stop.")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_system(registry, nodes)
        print("System stopped.")


if __name__ == "__main__":
    main()
