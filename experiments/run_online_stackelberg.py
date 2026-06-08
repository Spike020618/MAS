"""
Online Stackelberg Experiment — Compare debate with and without online quality feedback.

For EFFICIENCY, we only make new API calls for online conditions. Offline
conditions are computed from existing cached signals (no new API calls needed).

The comparison is:
1. debate_offline:   Standard 3-round debate from cached signals → MajVote + SG accuracy
2. debate_online:    New API calls with quality feedback injected between rounds
3. hier_offline:     Standard hierarchical from cached signals
4. hier_online:      New API calls with quality feedback between phases

Usage:
  python experiments/run_online_stackelberg.py --n_samples 30 --dataset StrategyQA
  python experiments/run_online_stackelberg.py --n_samples 30 --dataset all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from core.llm import MultiLLMPool
from core.orchestrator import Orchestrator
from core.similarity import SimilarityCalculator
from core.trust import extract_stance, majority_vote
from experiments.exp_orchestrator import load_questions
from experiments.collect_cache_topo import get_topology_W


DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]


def compute_offline_accuracy(dataset, topo, n_samples, seed=42):
    """Compute offline accuracy from cached signals (no API calls)."""
    signals_path = f"results/signals/{dataset}_{topo}.json"
    if not os.path.exists(signals_path):
        print(f"  WARNING: {signals_path} not found, skipping offline")
        return None

    with open(signals_path) as f:
        data = json.load(f)

    questions = load_questions(dataset, n_samples, seed=seed)
    q_texts = {q["question"][:200] for q in questions}

    correct_maj = 0
    total = 0
    for q in data["questions"][:n_samples]:
        if "signals" not in q:
            continue
        gt = q["ground_truth"].lower()
        stances = q["signals"]["stances"]
        counts = Counter(s.lower() for s in stances)
        maj = counts.most_common(1)[0][0]
        if maj == gt:
            correct_maj += 1
        total += 1

    return {
        "maj_acc": correct_maj / max(total, 1),
        "total": total,
        "correct": correct_maj,
    }


def run_online_experiment(datasets, n_samples, n_agents, out_dir, seed=42):
    """Run online-only experiment with real API calls."""
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "online_stackelberg_results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "meta": {"n_agents": n_agents, "seed": seed, "n_samples": n_samples},
            "offline": {},
            "online_experiments": [],
        }

    done_keys = set()
    for exp in all_results.get("online_experiments", []):
        done_keys.add((exp["dataset"], exp["condition"], exp["question_id"]))

    # Phase 1: Compute offline baselines from cached signals (instant)
    print("="*60)
    print("Phase 1: Computing offline baselines from cached signals")
    print("="*60)
    for dataset in datasets:
        for topo in ["debate", "hierarchical"]:
            key = f"{dataset}_{topo}"
            result = compute_offline_accuracy(dataset, topo, n_samples, seed)
            if result:
                all_results["offline"][key] = result
                print(f"  {key}: MajVote={result['maj_acc']:.1%} ({result['correct']}/{result['total']})")

    _save_results(all_results, results_path)

    # Phase 2: Run online conditions with real API calls
    print(f"\n{'='*60}")
    print("Phase 2: Running online experiments with API calls")
    print(f"{'='*60}")

    pool = MultiLLMPool()
    sim_calc = SimilarityCalculator(method="embedding")
    orch = Orchestrator(pool, sim_calc, n_agents=n_agents)

    online_conditions = [
        ("debate", True, "debate_online"),
        ("hierarchical", True, "hierarchical_online"),
    ]

    total_new = 0
    for dataset in datasets:
        questions = load_questions(dataset, n_samples, seed=seed)

        for topo, online, condition in online_conditions:
            n_done = sum(1 for k in done_keys
                         if k[0] == dataset and k[1] == condition)
            n_todo = len(questions) - n_done
            if n_todo <= 0:
                print(f"\n  {dataset}/{condition}: all done, skipping")
                continue

            W = get_topology_W(topo, n_agents)
            print(f"\n  --- {dataset}/{condition} ({n_done} done, {n_todo} todo) ---")

            for q_idx, q_item in enumerate(questions):
                if (dataset, condition, q_idx) in done_keys:
                    continue

                question = q_item["question"]
                gt = q_item["ground_truth"]

                try:
                    t0 = time.time()
                    result = orch.run(
                        question, mode=topo, topology_W=W,
                        use_stackelberg=True, online=True,
                        max_rounds=3,
                    )
                    elapsed = time.time() - t0

                    nodes = result["nodes"]
                    stances = [extract_stance(nd.get("C", "")) for nd in nodes]
                    decision = result["decision"]
                    maj_decision = majority_vote(nodes)

                    exp_entry = {
                        "dataset": dataset,
                        "condition": condition,
                        "question_id": q_idx,
                        "question": question[:200],
                        "ground_truth": gt,
                        "sg_decision": decision,
                        "maj_decision": maj_decision,
                        "sg_correct": decision == gt,
                        "maj_correct": maj_decision == gt,
                        "num_rounds": result.get("num_rounds", 1),
                        "elapsed_s": round(elapsed, 1),
                        "n_feedback_rounds": len(result.get("online_feedback_history", [])),
                    }

                    all_results["online_experiments"].append(exp_entry)
                    done_keys.add((dataset, condition, q_idx))
                    total_new += 1

                    ok = "OK" if decision == gt else "WRONG"
                    print(f"    [{q_idx+1}/{len(questions)}] {ok} "
                          f"({elapsed:.1f}s) SG={decision} Maj={maj_decision} GT={gt}")

                except Exception as e:
                    print(f"    [{q_idx+1}/{len(questions)}] ERROR: {e}")
                    all_results["online_experiments"].append({
                        "dataset": dataset,
                        "condition": condition,
                        "question_id": q_idx,
                        "question": question[:200],
                        "ground_truth": gt,
                        "error": str(e),
                        "sg_correct": False,
                        "maj_correct": False,
                    })
                    done_keys.add((dataset, condition, q_idx))
                    total_new += 1

                if total_new % 3 == 0:
                    _save_results(all_results, results_path)

    pool.close()
    _save_results(all_results, results_path)
    print(f"\nTotal new API experiments: {total_new}")

    print_summary(all_results)
    return all_results


def _save_results(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def print_summary(results):
    """Print comparison summary."""
    print(f"\n{'='*60}")
    print("SUMMARY: Online vs Offline Stackelberg")
    print(f"{'='*60}")

    from collections import defaultdict
    online_stats = defaultdict(lambda: {"sg_correct": 0, "maj_correct": 0, "total": 0})

    for exp in results.get("online_experiments", []):
        if "error" in exp:
            continue
        key = (exp["dataset"], exp["condition"])
        online_stats[key]["total"] += 1
        if exp.get("sg_correct"):
            online_stats[key]["sg_correct"] += 1
        if exp.get("maj_correct"):
            online_stats[key]["maj_correct"] += 1

    datasets = sorted(set(k[0] for k in online_stats.keys()) |
                       set(k.split("_")[0] for k in results.get("offline", {}).keys()))
    if not datasets:
        datasets = DATASETS

    for topo in ["debate", "hierarchical"]:
        print(f"\n--- {topo.upper()} ---")
        header = f"{'Metric':<20}"
        for ds in datasets:
            header += f" {ds:>12}"
        header += f" {'Average':>10}"
        print(header)
        print("-" * len(header))

        # Offline MajVote
        line = f"{'Offline MajVote':<20}"
        off_accs = []
        for ds in datasets:
            key = f"{ds}_{topo}"
            off = results.get("offline", {}).get(key)
            if off:
                acc = off["maj_acc"] * 100
                line += f" {acc:>11.1f}%"
                off_accs.append(acc)
            else:
                line += f" {'---':>12}"
        if off_accs:
            line += f" {sum(off_accs)/len(off_accs):>9.1f}%"
        print(line)

        # Online MajVote
        line = f"{'Online MajVote':<20}"
        on_maj_accs = []
        for ds in datasets:
            s = online_stats.get((ds, f"{topo}_online"))
            if s and s["total"] > 0:
                acc = s["maj_correct"] / s["total"] * 100
                line += f" {acc:>11.1f}%"
                on_maj_accs.append(acc)
            else:
                line += f" {'---':>12}"
        if on_maj_accs:
            line += f" {sum(on_maj_accs)/len(on_maj_accs):>9.1f}%"
        print(line)

        # Online SG
        line = f"{'Online SG-weighted':<20}"
        on_sg_accs = []
        for ds in datasets:
            s = online_stats.get((ds, f"{topo}_online"))
            if s and s["total"] > 0:
                acc = s["sg_correct"] / s["total"] * 100
                line += f" {acc:>11.1f}%"
                on_sg_accs.append(acc)
            else:
                line += f" {'---':>12}"
        if on_sg_accs:
            line += f" {sum(on_sg_accs)/len(on_sg_accs):>9.1f}%"
        print(line)

        # Delta (online MajVote - offline MajVote)
        line = f"{'Δ MajVote (on-off)':<20}"
        deltas = []
        for i, ds in enumerate(datasets):
            if i < len(off_accs) and i < len(on_maj_accs):
                d = on_maj_accs[i] - off_accs[i]
                line += f" {d:>+10.1f}pp"
                deltas.append(d)
            else:
                line += f" {'---':>12}"
        if deltas:
            line += f" {sum(deltas)/len(deltas):>+8.1f}pp"
        print(line)

    print()


def main():
    parser = argparse.ArgumentParser(description="Online Stackelberg Experiment")
    parser.add_argument("--dataset", default="StrategyQA",
                        choices=DATASETS + ["all"])
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/online_stackelberg")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    run_online_experiment(datasets, args.n_samples, args.n_agents, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
