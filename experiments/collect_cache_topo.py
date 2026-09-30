"""
多拓扑 AEIC Signal 收集 — 断点续跑

为每种拓扑模式生成真实的 AEIC 数据，保存为标准 signals 格式。

Phase 1 (collect): 使用 Orchestrator 在指定拓扑下跑 LLM
Phase 2 (precompute): 从 AEIC 缓存计算 IC/EC/EQ/EV/stance/agreement

Usage:
  # 收集 sequential 拓扑数据
  python experiments/collect_cache_topo.py --dataset MMLU-Hard --topology sequential

  # 仅预计算信号（已有缓存时）
  python experiments/collect_cache_topo.py --precompute experiments/results/cache/MMLU-Hard_sequential.json

  # Smoke test: 5题验证
  python experiments/collect_cache_topo.py --dataset StrategyQA --topology star --n_samples 5 --n_agents 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.llm import MultiLLMPool, AGENT_PERSONAS
from experiments.core.orchestrator import Orchestrator
from experiments.core.similarity import SimilarityCalculator
from experiments.core.topology import make_topology

TOPOLOGIES = ["sequential", "star", "debate", "agentverse", "hierarchical"]
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]


def get_topology_W(topology: str, n_agents: int) -> np.ndarray:
    """为指定拓扑模式生成 W 矩阵。"""
    if topology == "parallel":
        return np.zeros((n_agents, n_agents))
    elif topology == "sequential":
        return make_topology("path", n_agents)
    elif topology == "star":
        return make_topology("star", n_agents)
    elif topology in ("debate", "agentverse", "hierarchical"):
        return make_topology("complete", n_agents)
    else:
        raise ValueError(f"Unknown topology: {topology}")


def collect_topo_aeic(dataset: str, topology: str, n_samples: int,
                      n_agents: int, out_dir: str):
    """Phase 1: 使用 Orchestrator 在指定拓扑下生成 AEIC，增量缓存。"""
    from experiments.exp_orchestrator import load_questions

    cache_path = os.path.join(out_dir, "cache", f"{dataset}_{topology}.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Resuming from checkpoint: {len(cache['questions'])} questions done")
    else:
        cache = {
            "meta": {
                "dataset": dataset,
                "topology": topology,
                "n_agents": n_agents,
                "n_samples": n_samples,
            },
            "questions": [],
        }

    existing_ids = {q["question_id"] for q in cache["questions"]}

    questions = load_questions(dataset, n_samples)
    pool = MultiLLMPool()
    sim_calc = SimilarityCalculator(method="embedding")
    orch = Orchestrator(pool, sim_calc, n_agents=n_agents)
    W = get_topology_W(topology, n_agents)

    new_calls = 0
    skipped = 0
    errors = 0

    for q_idx, q_item in enumerate(questions):
        if q_idx in existing_ids:
            skipped += 1
            continue

        question = q_item["question"]
        gt = q_item["ground_truth"]

        try:
            t0 = time.time()
            result = orch.run(question, mode=topology, topology_W=W,
                              use_stackelberg=False)
            elapsed = time.time() - t0

            nodes = result["nodes"]
            agents_data = []
            for a_id, node in enumerate(nodes):
                backend = pool.get_backend_name(a_id)
                agents_data.append({
                    "agent_id": a_id,
                    "backend": backend,
                    "aeic": node if node else {"A": "", "E": [], "I": "", "C": ""},
                })

            cache["questions"].append({
                "question_id": q_idx,
                "question": question,
                "ground_truth": gt,
                "agents": agents_data,
                "topology": topology,
                "num_rounds": result.get("num_rounds", 1),
            })

            n_done = len(cache["questions"])
            n_total = len(questions)
            calls_per_q = _estimate_calls(topology, n_agents, result)
            new_calls += calls_per_q

            print(f"  [{n_done}/{n_total}] {elapsed:.1f}s "
                  f"q{q_idx} {question[:40]}...", flush=True)

        except Exception as e:
            errors += 1
            print(f"  ERROR q{q_idx}: {e}", flush=True)
            cache["questions"].append({
                "question_id": q_idx,
                "question": question,
                "ground_truth": gt,
                "agents": [{"agent_id": i, "backend": "error",
                            "aeic": {"A": "", "E": [], "I": "", "C": f"Error: {e}"}}
                           for i in range(n_agents)],
                "topology": topology,
                "error": str(e),
            })

        # 每5题保存一次 checkpoint
        if len(cache["questions"]) % 5 == 0:
            _save_cache(cache, cache_path)

    cache["meta"]["n_samples"] = len(cache["questions"])
    cache["meta"]["new_calls"] = new_calls
    cache["meta"]["errors"] = errors
    _save_cache(cache, cache_path)

    pool.close()
    print(f"\n  Done: {len(cache['questions'])} questions, "
          f"{new_calls} API calls, {errors} errors, {skipped} skipped")
    print(f"  Saved to {cache_path}")
    return cache_path


def _estimate_calls(topology: str, n_agents: int, result: dict) -> int:
    """估算本题的 API 调用数。"""
    if topology == "parallel":
        return n_agents
    elif topology == "sequential":
        return n_agents
    elif topology == "star":
        return n_agents + 1  # leaves + hub(with context)
    elif topology == "debate":
        rounds = result.get("num_rounds", 3)
        return n_agents * rounds
    elif topology == "agentverse":
        return n_agents + 1  # solver + reviewers + solver_revise
    return n_agents


def _save_cache(cache: dict, path: str):
    """原子写入（写临时文件再重命名，防断电丢数据）。"""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def precompute_signals(cache_path: str):
    """Phase 2: 从 AEIC 缓存计算 IC/EC/EQ/EV/stance/agreement（使用实际拓扑W）。"""
    from experiments.core.trust import (
        compute_ic_consistency, compute_ec_relevance,
        compute_evidence_quality, compute_evidence_consensus,
        compute_neighbor_agreement, extract_stance,
    )

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    sim = SimilarityCalculator(method="embedding")
    n_agents = cache["meta"]["n_agents"]
    topology = cache["meta"].get("topology", "parallel")
    W = get_topology_W(topology, n_agents)

    for q_idx, q_data in enumerate(cache["questions"]):
        if "signals" in q_data:
            continue

        nodes = [a["aeic"] for a in q_data["agents"]]

        ic = [compute_ic_consistency(nd.get("I", ""), nd.get("C", ""), sim)
              for nd in nodes]
        ec = [compute_ec_relevance(nd.get("E", []), nd.get("C", ""), sim)
              for nd in nodes]
        eq = compute_evidence_quality(nodes)
        ev = compute_evidence_consensus(nodes, W, sim)
        stances = [extract_stance(nd.get("C", "")) for nd in nodes]
        agreement = compute_neighbor_agreement(nodes, W)

        q_data["signals"] = {
            "ic": [round(x, 4) for x in ic],
            "ec": [round(x, 4) for x in ec],
            "eq": [round(x, 4) for x in eq],
            "ev": [round(x, 4) for x in ev],
            "stances": stances,
            "agreement": [round(x, 4) for x in agreement],
        }

        if (q_idx + 1) % 10 == 0:
            print(f"  Precomputed {q_idx+1}/{len(cache['questions'])}")

    # 保存到 signals 目录
    out_path = cache_path.replace("/cache/", "/signals/")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"  Signals saved to {out_path}")
    return out_path


def verify_code(dataset: str, topology: str, n_agents: int):
    """验证模式：不调 API，只检查代码路径是否正确。"""
    from experiments.exp_orchestrator import load_questions

    print(f"=== Verification Mode ===")
    print(f"  Dataset: {dataset}")
    print(f"  Topology: {topology}")
    print(f"  N_agents: {n_agents}")

    # 1. 验证数据集加载
    print(f"\n[1] Loading dataset...")
    questions = load_questions(dataset, 5)
    print(f"  OK: loaded {len(questions)} questions")
    print(f"  Sample: {questions[0]['question'][:60]}...")

    # 2. 验证拓扑 W 矩阵
    print(f"\n[2] Generating topology W matrix...")
    W = get_topology_W(topology, n_agents)
    print(f"  Shape: {W.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(W)}")
    print(f"  W:\n{W}")

    # 3. 验证 Orchestrator 模式存在
    print(f"\n[3] Checking Orchestrator mode '{topology}'...")
    valid_modes = ["parallel", "sequential", "star", "debate",
                   "selfcorrect", "agentverse", "hierarchical"]
    if topology in valid_modes:
        print(f"  OK: mode '{topology}' supported")
    else:
        print(f"  ERROR: mode '{topology}' not in {valid_modes}")
        return False

    # 4. 验证输出路径
    print(f"\n[4] Output paths:")
    cache_path = f"experiments/results/cache/{dataset}_{topology}.json"
    signal_path = f"experiments/results/signals/{dataset}_{topology}.json"
    print(f"  Cache: {cache_path}")
    print(f"  Signals: {signal_path}")

    # 5. 验证 API 开销估算
    print(f"\n[5] Estimated API calls for 200 questions:")
    if topology == "parallel":
        est = 200 * n_agents
    elif topology == "sequential":
        est = 200 * n_agents
    elif topology == "star":
        est = 200 * (n_agents + 1)
    elif topology == "debate":
        est = 200 * n_agents * 3  # 3 rounds
    elif topology == "agentverse":
        est = 200 * (n_agents + 1)
    else:
        est = 0
    print(f"  ~{est} calls")

    # 6. 验证 SimilarityCalculator
    print(f"\n[6] Testing SimilarityCalculator...")
    sim = SimilarityCalculator(method="embedding")
    score = sim.compute("This is a test", "This is another test")
    print(f"  OK: similarity score = {score:.4f}")

    print(f"\n=== All checks passed ===")
    return True


def main():
    parser = argparse.ArgumentParser(description="Multi-Topology AEIC Collection")
    parser.add_argument("--dataset", default="StrategyQA",
                        choices=DATASETS + ["all"])
    parser.add_argument("--topology", default="sequential",
                        choices=TOPOLOGIES + ["all"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--precompute", default=None,
                        help="Path to cache JSON for Phase 2 only")
    parser.add_argument("--verify", action="store_true",
                        help="Verify code without API calls")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "signals"), exist_ok=True)

    if args.verify:
        datasets = DATASETS if args.dataset == "all" else [args.dataset]
        topos = TOPOLOGIES if args.topology == "all" else [args.topology]
        for ds in datasets:
            for topo in topos:
                ok = verify_code(ds, topo, args.n_agents)
                if not ok:
                    sys.exit(1)
        return

    if args.precompute:
        print("Phase 2: Precomputing signals...")
        precompute_signals(args.precompute)
        return

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    topos = TOPOLOGIES if args.topology == "all" else [args.topology]

    for ds in datasets:
        for topo in topos:
            print(f"\n{'='*60}")
            print(f"Collecting: {ds} × {topo}")
            print(f"{'='*60}")

            cache_path = collect_topo_aeic(
                ds, topo, args.n_samples, args.n_agents, args.out_dir)

            print(f"\nPhase 2: Precomputing signals for {ds}_{topo}...")
            precompute_signals(cache_path)


if __name__ == "__main__":
    main()
