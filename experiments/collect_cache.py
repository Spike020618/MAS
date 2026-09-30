"""
AEIC缓存收集 + 信号预计算

Phase 1 (collect): 跑parallel mode，缓存每agent每题的原始AEIC输出
Phase 2 (precompute): 从AEIC缓存计算IC/EC/EQ/EV/stance/agreement

Usage:
  # Phase 1: 收集AEIC（调API）
  python experiments/collect_cache.py --dataset StrategyQA --n_samples 50 --n_agents 3

  # Phase 2: 预计算信号（本地embedding）
  python experiments/collect_cache.py --precompute experiments/results/cache/StrategyQA.json
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


def _atomic_save(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def collect_aeic(dataset: str, n_samples: int, n_agents: int,
                 out_dir: str, agent_map: dict = None):
    """Phase 1: 独立生成每agent每题的AEIC，增量缓存。"""
    from experiments.exp_orchestrator import load_questions

    cache_path = os.path.join(out_dir, "cache", f"{dataset}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {"meta": {"dataset": dataset, "n_agents": n_agents,
                          "n_samples": n_samples},
                 "questions": []}

    existing_ids = {q["question_id"] for q in cache["questions"]}

    questions = load_questions(dataset, n_samples)
    pool = MultiLLMPool(agent_map=agent_map) if agent_map else MultiLLMPool()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _call_one(a_id, question, pool):
        persona_id = a_id % len(AGENT_PERSONAS)
        backend = pool.get_backend_name(a_id)
        try:
            aeic = pool.generate_aeic(question, persona_id=persona_id,
                                      agent_id=a_id)
        except Exception as e:
            print(f"  ERROR a{a_id}: {e}", flush=True)
            aeic = {"A": "", "E": [], "I": "", "C": "No: error"}
        return {"agent_id": a_id, "backend": backend, "aeic": aeic}

    new_calls = 0
    for q_idx, q_item in enumerate(questions):
        if q_idx in existing_ids:
            continue

        question = q_item["question"]
        gt = q_item["ground_truth"]

        with ThreadPoolExecutor(max_workers=n_agents) as executor:
            futures = {executor.submit(_call_one, a_id, question, pool): a_id
                       for a_id in range(n_agents)}
            agents_data = [None] * n_agents
            for future in as_completed(futures):
                a_id = futures[future]
                agents_data[a_id] = future.result()
        new_calls += n_agents

        cache["questions"].append({
            "question_id": q_idx,
            "question": question,
            "ground_truth": gt,
            "agents": agents_data,
        })

        if new_calls % (n_agents * 5) == 0:
            _atomic_save(cache, cache_path)

        print(f"  [{q_idx+1}/{len(questions)}] {question[:50]}... "
              f"({new_calls} calls)", flush=True)

    cache["meta"]["n_samples"] = len(cache["questions"])
    _atomic_save(cache, cache_path)

    pool.close()
    print(f"\n  Saved {len(cache['questions'])} questions to {cache_path}")
    print(f"  New API calls: {new_calls}")
    return cache_path


def precompute_signals(cache_path: str):
    """Phase 2: 从AEIC缓存计算IC/EC/EQ/EV/stance/agreement。"""
    from experiments.core.similarity import SimilarityCalculator
    from experiments.core.trust import (
        compute_ic_consistency, compute_ec_relevance,
        compute_evidence_quality, compute_evidence_consensus,
        compute_neighbor_agreement, extract_stance,
    )

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    sim = SimilarityCalculator(method="embedding")
    n_agents = cache["meta"]["n_agents"]
    W = np.zeros((n_agents, n_agents))

    for q_idx, q_data in enumerate(cache["questions"]):
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

    out_path = cache_path.replace("/cache/", "/signals/")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\n  Signals saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="AEIC Cache Collection")
    parser.add_argument("--dataset", default="StrategyQA")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--precompute", default=None,
                        help="Path to aeic_cache JSON for Phase 2")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "signals"), exist_ok=True)

    if args.precompute:
        print("Phase 2: Precomputing signals...")
        precompute_signals(args.precompute)
    else:
        print(f"Phase 1: Collecting AEIC cache")
        print(f"  Dataset: {args.dataset}, Samples: {args.n_samples}, "
              f"Agents: {args.n_agents}")
        cache_path = collect_aeic(args.dataset, args.n_samples,
                                   args.n_agents, args.out_dir)
        print(f"\nPhase 2: Precomputing signals...")
        precompute_signals(cache_path)


if __name__ == "__main__":
    main()
