"""
Online Stackelberg A/B Experiment — Fair comparison with live API calls.

Both online and offline conditions use fresh API calls to eliminate
cached-vs-live confounding. The only difference is whether quality
feedback is injected between rounds.

Conditions:
  debate_offline:  3-round debate, no feedback injection (live API)
  debate_online:   3-round debate, quality feedback between rounds (live API)
  hier_offline:    3-phase hierarchical, no feedback (live API)
  hier_online:     3-phase hierarchical, quality feedback between phases (live API)

Resume support: completed entries are tracked by (dataset, condition, question_id).
Errors are saved but NOT counted as done — they will be retried on re-run.

Usage:
  python experiments/run_online_stackelberg.py --dataset all --n_samples 50
  python experiments/run_online_stackelberg.py --dataset MMLU-Hard --conditions debate_online debate_offline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

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
CONDITIONS = [
    # (topology, online, dynamic, condition_name)
    ("debate",       False, False, "debate_offline"),
    ("debate",       True,  False, "debate_online"),
    ("debate",       False, True,  "debate_dynamic"),
    ("hierarchical", False, False, "hier_offline"),
    ("hierarchical", True,  False, "hier_online"),
]


def _save_results(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"meta": {}, "experiments": []}


def _get_done_keys(results):
    """Return set of (dataset, condition, question_id) for successfully completed entries."""
    done = set()
    for exp in results.get("experiments", []):
        if "error" not in exp:
            done.add((exp["dataset"], exp["condition"], exp["question_id"]))
    return done


def run_experiment(datasets, conditions, n_samples, n_agents, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "online_ab_results.json")

    results = _load_results(results_path)
    results["meta"] = {
        "n_agents": n_agents, "seed": seed, "n_samples": n_samples,
        "design": "A/B with live API calls for all conditions",
    }

    done_keys = _get_done_keys(results)
    print(f"Loaded {len(done_keys)} completed entries from {results_path}")

    pool = MultiLLMPool()
    sim_calc = SimilarityCalculator(method="embedding")
    orch = Orchestrator(pool, sim_calc, n_agents=n_agents)

    total_new = 0
    for dataset in datasets:
        questions = load_questions(dataset, n_samples, seed=seed)

        for topo, online, dynamic, condition in conditions:
            n_done = sum(1 for k in done_keys
                         if k[0] == dataset and k[1] == condition)
            n_todo = len(questions) - n_done
            if n_todo <= 0:
                print(f"\n  {dataset}/{condition}: all {len(questions)} done, skipping")
                continue

            W = get_topology_W(topo, n_agents)
            print(f"\n  --- {dataset}/{condition} ({n_done} done, {n_todo} remaining) ---")

            for q_idx, q_item in enumerate(questions):
                if (dataset, condition, q_idx) in done_keys:
                    continue

                question = q_item["question"]
                gt = q_item["ground_truth"]

                try:
                    t0 = time.time()
                    result = orch.run(
                        question, mode=topo, topology_W=W,
                        use_stackelberg=True, online=online,
                        dynamic=dynamic, max_rounds=3,
                    )
                    elapsed = time.time() - t0

                    nodes = result["nodes"]
                    stances = [extract_stance(nd.get("C", "")) for nd in nodes]
                    decision = result["decision"]
                    maj_decision = majority_vote(nodes)

                    exp_entry = {
                        "dataset": dataset,
                        "condition": condition,
                        "topology": topo,
                        "question_id": q_idx,
                        "question": question[:200],
                        "ground_truth": gt,
                        "sg_decision": decision,
                        "maj_decision": maj_decision,
                        "sg_correct": decision == gt,
                        "maj_correct": maj_decision == gt,
                        "stances": stances,
                        "num_rounds": result.get("num_rounds", 1),
                        "elapsed_s": round(elapsed, 1),
                        "n_feedback_rounds": len(result.get("online_feedback_history", [])),
                    }
                    if dynamic:
                        dyn_hist = result.get("dynamics_history", [])
                        exp_entry["dynamics_history"] = dyn_hist
                        exp_entry["dynamic_states"] = [
                            d["state"] for d in dyn_hist]

                    results["experiments"].append(exp_entry)
                    done_keys.add((dataset, condition, q_idx))
                    total_new += 1

                    ok = "OK" if maj_decision == gt else "WRONG"
                    print(f"    [{q_idx+1}/{len(questions)}] {ok} "
                          f"({elapsed:.1f}s) Maj={maj_decision} SG={decision} GT={gt}")

                except Exception as e:
                    print(f"    [{q_idx+1}/{len(questions)}] ERROR: {e}")
                    results["experiments"].append({
                        "dataset": dataset,
                        "condition": condition,
                        "topology": topo,
                        "question_id": q_idx,
                        "question": question[:200],
                        "ground_truth": gt,
                        "error": str(e),
                    })
                    total_new += 1

                if total_new % 3 == 0:
                    _save_results(results, results_path)

    pool.close()
    _save_results(results, results_path)
    print(f"\nTotal new entries: {total_new}")

    print_summary(results)
    return results


def print_summary(results):
    print(f"\n{'='*70}")
    print("SUMMARY: Online vs Offline Stackelberg (A/B, live API)")
    print(f"{'='*70}")

    stats = defaultdict(lambda: {"maj_correct": 0, "sg_correct": 0, "total": 0})
    for exp in results.get("experiments", []):
        if "error" in exp:
            continue
        key = (exp["dataset"], exp["condition"])
        stats[key]["total"] += 1
        if exp.get("maj_correct"):
            stats[key]["maj_correct"] += 1
        if exp.get("sg_correct"):
            stats[key]["sg_correct"] += 1

    datasets = sorted(set(k[0] for k in stats.keys()))
    if not datasets:
        print("  No valid results yet.")
        return

    for topo in ["debate", "hierarchical"]:
        off_cond = f"{topo}_offline" if topo == "debate" else "hier_offline"
        on_cond = f"{topo}_online" if topo == "debate" else "hier_online"
        dyn_cond = f"{topo}_dynamic" if topo == "debate" else None

        print(f"\n--- {topo.upper()} ---")
        header = f"{'Condition':<20}"
        for ds in datasets:
            header += f" {ds:>12}"
        header += f" {'Average':>10}"
        print(header)
        print("-" * len(header))

        cond_labels = [(off_cond, "Offline (no fb)"),
                       (on_cond, "Online (static)")]
        if dyn_cond:
            cond_labels.append((dyn_cond, "Dynamic (adaptive)"))

        for cond, label in cond_labels:
            line = f"{label:<20}"
            accs = []
            for ds in datasets:
                s = stats.get((ds, cond))
                if s and s["total"] > 0:
                    acc = s["maj_correct"] / s["total"] * 100
                    line += f" {acc:>10.1f}% "
                    accs.append(acc)
                else:
                    line += f" {'---':>12}"
            if accs:
                line += f" {sum(accs)/len(accs):>9.1f}%"
            print(line)

        # Deltas
        for ref_cond, ref_label, cmp_cond, cmp_label in [
            (off_cond, "off", on_cond, "on"),
            (off_cond, "off", dyn_cond, "dyn") if dyn_cond else (None,)*4,
        ]:
            if ref_cond is None:
                continue
            line = f"{'Δ (' + cmp_label + ' - ' + ref_label + ')':<20}"
            for ds in datasets:
                s_ref = stats.get((ds, ref_cond))
                s_cmp = stats.get((ds, cmp_cond))
                if s_ref and s_cmp and s_ref["total"] > 0 and s_cmp["total"] > 0:
                    d = (s_cmp["maj_correct"]/s_cmp["total"] - s_ref["maj_correct"]/s_ref["total"]) * 100
                    line += f" {d:>+10.1f}pp"
                else:
                    line += f" {'---':>12}"
            print(line)

    # Error count
    n_errors = sum(1 for e in results.get("experiments", []) if "error" in e)
    if n_errors > 0:
        print(f"\n  ({n_errors} entries had errors and will be retried on next run)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Online Stackelberg A/B Experiment")
    parser.add_argument("--dataset", default="all",
                        choices=DATASETS + ["all"])
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/online_stackelberg")
    parser.add_argument("--conditions", nargs="+",
                        choices=[c[3] for c in CONDITIONS],
                        default=None,
                        help="Subset of conditions to run (default: all)")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    if args.conditions:
        conditions = [c for c in CONDITIONS if c[3] in args.conditions]
    else:
        conditions = CONDITIONS

    run_experiment(datasets, conditions, args.n_samples, args.n_agents,
                   args.out_dir, args.seed)


if __name__ == "__main__":
    main()
