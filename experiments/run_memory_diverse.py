"""
Memory-Diverse Consensus Experiment.

Tests whether AEIC+Stackelberg correctly weights agents with better
knowledge when agents have heterogeneous memory quality.

Design:
- 5 agents, 3 with correct background knowledge, 2 with misleading
- 3 conditions: baseline (majority vote), sg_online, sg_dynamic
- Topology: debate (3 rounds)
- Dataset: MMLU-Hard (30 questions pilot)

Hypothesis:
- SG online will NOT significantly outperform baseline (quality
  signals measure reasoning coherence, not correctness — consistent
  with paper's main finding)
- Dynamic regulation may help by detecting collective behavior patterns
  (sycophancy to misleading agents, deadlock)

Usage:
  python experiments/run_memory_diverse.py [--n_samples 30] [--dataset MMLU-Hard]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from core.llm import LLMClient
from core.orchestrator import Orchestrator
from core.similarity import SimilarityCalculator
from core.trust import majority_vote, extract_stance
from experiments.collect_cache_topo import get_topology_W
from experiments.exp_orchestrator import load_questions


def generate_evidence_hints(question_item: dict, rng: random.Random) -> dict:
    """Generate per-agent evidence hints based on question metadata.

    Returns dict mapping agent_id -> list of hint strings.
    For MMLU-formatted questions, uses correct/incorrect choices.
    """
    gt = question_item["ground_truth"]
    question = question_item["question"]

    # Extract the proposed answer from the question text
    # Format: "Question: ...\nProposed answer: ...\nIs this proposed answer correct?"
    proposed = ""
    if "Proposed answer:" in question:
        proposed = question.split("Proposed answer:")[1].split("\n")[0].strip()
        # Remove trailing question
        proposed = proposed.rstrip("?.")

    # For correct-hint agents: confirm the proposed answer
    # For misleading agents: contradict it
    if gt == "yes":
        correct_hint = (
            f"Background research indicates that the proposed answer "
            f"'{proposed}' is well-supported by evidence."
        )
        misleading_hint = (
            f"Background research indicates that the proposed answer "
            f"'{proposed}' is controversial and likely incorrect."
        )
    else:
        # GT=no means proposed answer is wrong
        correct_hint = (
            f"Background research indicates that the proposed answer "
            f"'{proposed}' is incorrect based on reliable sources."
        )
        misleading_hint = (
            f"Background research indicates that the proposed answer "
            f"'{proposed}' is actually correct according to new evidence."
        )

    return {
        0: [correct_hint],    # Analyst — correct knowledge
        1: [correct_hint],    # Skeptic — correct knowledge
        2: [correct_hint],    # Creative — correct knowledge
        3: [misleading_hint], # Expert — misleading knowledge
        4: [misleading_hint], # Pragmatist — misleading knowledge
    }


def evaluate_accuracy(results: list) -> dict:
    """Compute accuracy with per-agent weight analysis."""
    from collections import defaultdict
    stats = {
        "baseline": {"correct": 0, "total": 0, "maj_correct": 0},
        "sg_online": {"correct": 0, "total": 0, "maj_correct": 0},
        "sg_dynamic": {"correct": 0, "total": 0, "maj_correct": 0},
    }

    for r in results:
        cond = r["condition"]
        stats[cond]["total"] += 1
        if r["sg_correct"]:
            stats[cond]["correct"] += 1
        if r["maj_correct"]:
            stats[cond]["maj_correct"] += 1

    summary = {}
    for cond, s in stats.items():
        n = s["total"]
        summary[cond] = {
            "sg_accuracy": round(s["correct"] / n * 100, 1) if n else 0,
            "maj_accuracy": round(s["maj_correct"] / n * 100, 1) if n else 0,
            "n": n,
        }
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--dataset", default="MMLU-Hard")
    parser.add_argument("--out_dir", default="results/memory_diverse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "memory_diverse_results.json")

    # Load existing results for resume
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Resumed: {len(all_results['experiments'])} existing entries")
    else:
        all_results = {"meta": {}, "experiments": []}

    done_keys = set()
    for exp in all_results.get("experiments", []):
        if "error" not in exp:
            done_keys.add((exp["dataset"], exp["condition"], exp["question_id"]))

    all_results["meta"] = {
        "n_agents": 5,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "dataset": args.dataset,
        "design": "Memory-diverse: 3 correct memory, 2 misleading memory",
    }

    # Initialize
    llm = LLMClient(backend="deepseek")
    sim_calc = SimilarityCalculator(method="embedding")
    orch = Orchestrator(llm, sim_calc, n_agents=5)
    W = get_topology_W("debate", 5)
    rng = random.Random(args.seed)

    questions = load_questions(args.dataset, args.n_samples, seed=args.seed)
    print(f"Loaded {len(questions)} questions from {args.dataset}")

    conditions = [
        # (condition_name, use_stackelberg, online, dynamic)
        ("baseline",   False, False, False),
        ("sg_online",  True,  True,  False),
        ("sg_dynamic", True,  True,  True),
    ]

    total_new = 0
    for q_idx, q_item in enumerate(questions):
        question = q_item["question"]
        gt = q_item["ground_truth"]
        evidence_map = generate_evidence_hints(q_item, rng)
        evidence_hints_list = [evidence_map[i] for i in range(5)]

        for cond_name, use_sg, online, dynamic in conditions:
            key = (args.dataset, cond_name, q_idx)
            if key in done_keys:
                continue

            print(f"\n[{q_idx+1}/{len(questions)}] {cond_name}: {question[:80]}...")
            t0 = time.time()

            try:
                result = orch.run(
                    question, mode="debate", topology_W=W,
                    use_stackelberg=use_sg, online=online,
                    dynamic=dynamic, max_rounds=3,
                    evidence_hints_list=evidence_hints_list,
                )
                elapsed = time.time() - t0

                nodes = result["nodes"]
                decision = result["decision"]
                maj_decision = majority_vote(nodes)
                stances = [extract_stance(nd.get("C", "")) for nd in nodes]

                exp_entry = {
                    "dataset": args.dataset,
                    "condition": cond_name,
                    "topology": "debate",
                    "use_stackelberg": use_sg,
                    "online": online,
                    "dynamic": dynamic,
                    "question_id": q_idx,
                    "question": question[:200],
                    "ground_truth": gt,
                    "sg_decision": decision,
                    "maj_decision": maj_decision,
                    "sg_correct": decision == gt,
                    "maj_correct": maj_decision == gt,
                    "stances": stances,
                    "evidence_map": {str(k): v for k, v in evidence_map.items()},
                    "num_rounds": result.get("num_rounds", 1),
                    "elapsed_s": round(elapsed, 1),
                }

                # Per-agent weights if available
                theta = result.get("theta", None)
                trust = result.get("trust", None)
                if theta is not None:
                    exp_entry["theta"] = [round(float(t), 4) for t in theta]
                if trust is not None:
                    exp_entry["trust"] = [round(float(t), 4) for t in trust]

                if dynamic:
                    dyn_hist = result.get("dynamics_history", [])
                    exp_entry["dynamics_history"] = dyn_hist
                    exp_entry["dynamic_states"] = [
                        d["state"] for d in dyn_hist]

                all_results["experiments"].append(exp_entry)
                done_keys.add(key)
                total_new += 1

                correct_str = "✓" if decision == gt else "✗"
                print(f"  → {correct_str} Maj={maj_decision} SG={decision} GT={gt} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  → ERROR: {e}")
                all_results["experiments"].append({
                    "dataset": args.dataset,
                    "condition": cond_name,
                    "topology": "debate",
                    "question_id": q_idx,
                    "question": question[:200],
                    "ground_truth": gt,
                    "error": str(e),
                })
                total_new += 1

        # Save periodically
        if (q_idx + 1) % 10 == 0:
            tmp = results_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            os.replace(tmp, results_path)

    # Final save
    tmp = results_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, results_path)

    llm.close()
    print(f"\n=== New entries: {total_new} ===")

    summary = evaluate_accuracy(all_results["experiments"])
    print("\n=== Accuracy Summary ===")
    for cond, s in sorted(summary.items()):
        print(f"  {cond:12s}: SG={s['sg_accuracy']:5.1f}% Maj={s['maj_accuracy']:5.1f}% (n={s['n']})")

    # Per-agent weight analysis (SG conditions)
    print("\n=== Per-Agent Weight Analysis (correct-informed vs misled) ===")
    for cond_name, _, _, _ in conditions:
        if cond_name == "baseline":
            continue
        weights_correct = []
        weights_misled = []
        for exp in all_results["experiments"]:
            if exp["condition"] != cond_name or "error" in exp:
                continue
            if "theta" not in exp:
                continue
            t = exp["theta"]
            # Agents 0-2: correct memory, 3-4: misleading
            w_c = sum(t[0:3]) / 3
            w_m = sum(t[3:5]) / 2
            weights_correct.append(w_c)
            weights_misled.append(w_m)
        if weights_correct:
            print(f"  {cond_name:12s}: correct-informed mean θ={np.mean(weights_correct):.3f} "
                  f"vs misled mean θ={np.mean(weights_misled):.3f} "
                  f"(diff={np.mean(weights_correct)-np.mean(weights_misled):+.3f})")


if __name__ == "__main__":
    main()
