"""
Fixed Topology Experiment — AEIC-Stackelberg 四种编排模式对比

对比方法: MajVote, QualWt, AEIC (no SG), AEIC+SG
拓扑模式: parallel, sequential, star, debate
n_agents: 1, 3, 5

每方法输出独立CSV，统一格式，每题一行。

Usage:
  python experiments/exp_orchestrator.py --n_samples 20 --n_agents 3
  python experiments/exp_orchestrator.py --dataset StrategyQA --modes parallel,sequential
  python experiments/exp_orchestrator.py --dataset StrategyQA --n_samples 20 --csv_dir results
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

from experiments.core.llm import LLMClient, MultiLLMPool, AGENT_PERSONAS
from experiments.core.orchestrator import Orchestrator
from experiments.core.similarity import SimilarityCalculator
from experiments.core.topology import make_topology
from experiments.core.trust import (
    evaluate_and_decide,
    majority_vote,
    quality_weighted_vote,
    extract_stance,
    compute_ic_consistency,
    compute_ec_relevance,
    compute_evidence_quality,
    compute_evidence_consensus,
)
from experiments.core.trust_tracker import TrustTracker


# ─── 统一CSV格式 ───────────────────────────────────────────────────

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


# ─── 数据加载 ───────────────────────────────────────────────────────

def load_questions(dataset_name: str, n_samples: int, seed: int = 42,
                   offset: int = 0) -> List[Dict]:
    """Load questions + ground truth from dataset.

    All datasets are converted to yes/no questions:
    - StrategyQA/TruthfulQA: native boolean
    - MCQ datasets: converted to "Is [correct_answer] the answer to: [question]?"
      with 50% positive (GT=yes) and 50% negative (GT=no, using wrong answer)
    """
    import random as _rng
    from experiments.dataset_loader import DATASET_REGISTRY

    reg = DATASET_REGISTRY[dataset_name]
    raw = reg["loader"](n_samples * 2 + offset)  # load extra for MCQ splitting
    rng = _rng.Random(seed)

    questions = []
    if dataset_name == "StrategyQA":
        for item in raw[offset:]:
            q = item["question"]
            gt = item["answer"]  # boolean
            gt_stance = "yes" if gt else "no"
            questions.append({"question": q, "ground_truth": gt_stance,
                              "source": dataset_name})
    elif dataset_name == "TruthfulQA":
        for item in raw[offset:]:
            q = item.get("question", "")
            correct_exp = item.get("correct_explanation", "")
            wrong_exp = item.get("incorrect_explanation", "")
            gt_bool = item.get("ground_truth", True)
            if correct_exp:
                yes_q = (f"Question: {q}\nProposed answer: {correct_exp}\n"
                         f"Is this proposed answer factually correct?")
                questions.append({"question": yes_q, "ground_truth": "yes",
                                  "source": dataset_name})
            if wrong_exp:
                no_q = (f"Question: {q}\nProposed answer: {wrong_exp}\n"
                        f"Is this proposed answer factually correct?")
                questions.append({"question": no_q, "ground_truth": "no",
                                  "source": dataset_name})
        rng.shuffle(questions)
    elif dataset_name == "HotpotQA":
        for item in raw[offset:]:
            q = item.get("question", "")
            ans = str(item.get("answer", "")).strip().lower()
            if ans in ("yes", "no"):
                questions.append({"question": q, "ground_truth": ans,
                                  "source": dataset_name})
            else:
                yes_q = (f"Question: {q}\nProposed answer: {item['answer']}\n"
                         f"Research the question independently, then judge: "
                         f"is the proposed answer correct?")
                questions.append({"question": yes_q, "ground_truth": "yes",
                                  "source": dataset_name})
    elif dataset_name in ("GSM8K", "GSM-Hard"):
        for item in raw[offset:]:
            q = item.get("question", "")
            ans = str(item.get("answer", ""))
            yes_q = (f"Solve this math problem step by step: {q}\n"
                     f"Someone claims the answer is {ans}. "
                     f"Is this answer correct? Verify by solving independently.")
            questions.append({"question": yes_q, "ground_truth": "yes",
                              "source": dataset_name})
            wrong_ans = int(ans) + rng.choice([-5, -3, -2, 2, 3, 5, 10]) if ans.lstrip('-').isdigit() else "0"
            no_q = (f"Solve this math problem step by step: {q}\n"
                    f"Someone claims the answer is {wrong_ans}. "
                    f"Is this answer correct? Verify by solving independently.")
            questions.append({"question": no_q, "ground_truth": "no",
                              "source": dataset_name})
        rng.shuffle(questions)
    else:
        # MCQ datasets (GPQA, MMLU, ARC-Challenge, CommonsenseQA)
        for item in raw[offset:]:
            q = item.get("question", "")
            correct = item.get("correct_answer", "")
            choices = item.get("choices", [])
            if not q or not correct:
                continue

            yes_q = (f"Question: {q}\nProposed answer: {correct}\n"
                     f"First solve the question independently, then judge: "
                     f"is the proposed answer correct?")
            questions.append({"question": yes_q, "ground_truth": "yes",
                              "source": dataset_name})

            wrong_choices = [c for c in choices if c != correct]
            if wrong_choices:
                wrong = rng.choice(wrong_choices)
                no_q = (f"Question: {q}\nProposed answer: {wrong}\n"
                        f"First solve the question independently, then judge: "
                        f"is the proposed answer correct?")
                questions.append({"question": no_q, "ground_truth": "no",
                                  "source": dataset_name})

        rng.shuffle(questions)

    return questions[:n_samples]


def evaluate_accuracy(results: List[Dict]) -> Dict[str, float]:
    """Compute accuracy metrics."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "n": 0}

    correct = sum(1 for r in results if r["decision"] == r["ground_truth"])
    return {
        "accuracy": correct / total,
        "correct": correct,
        "n": total,
    }


# ─── 实验主逻辑 ─────────────────────────────────────────────────────

def run_experiment(
    questions: List[Dict],
    modes: List[str],
    n_agents: int = 3,
    sim_method: str = "embedding",
    csv_dir: str = None,
    dataset_name: str = "",
) -> Dict[str, Dict[str, List[Dict]]]:
    """Run all modes × all methods on the questions."""

    llm = MultiLLMPool()
    sim = SimilarityCalculator(method=sim_method)
    orch = Orchestrator(llm, sim, n_agents=n_agents)

    topology_map = {
        "parallel": "complete",
        "sequential": "path",
        "star": "star",
        "debate": "complete",
        "selfcorrect": "complete",
        "agentverse": "complete",
    }

    methods = ["AEIC+SG", "AEIC", "QualWt", "MajVote"]

    # Load completed rows per method file
    completed_per_method = {}
    method_paths = {}
    if csv_dir:
        for method in methods:
            fname = f"fixed_{method}_{dataset_name}.csv".replace("+", "_")
            fpath = os.path.join(csv_dir, fname)
            method_paths[method] = fpath
            completed_per_method[method] = _load_completed(fpath)
        total_completed = sum(len(v) for v in completed_per_method.values())
        if total_completed:
            print(f"  CSV checkpoint: {total_completed} rows already completed")

    # TrustTracker across all questions (all modes share same tracker)
    tracker = TrustTracker(n_agents=n_agents, warmup=3)

    all_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*60}")

        topo_name = topology_map[mode]
        n = n_agents if mode != "star" else n_agents + 1
        W = make_topology(topo_name, n)
        condition = f"{mode}_{n_agents}a"

        mode_results = {m: [] for m in methods}

        for q_idx, q_item in enumerate(questions):
            question = q_item["question"]
            gt = q_item["ground_truth"]

            # 断点续跑
            if csv_dir:
                all_done = all(
                    (dataset_name, condition, str(q_idx))
                    in completed_per_method.get(m, set())
                    for m in methods
                )
                if all_done:
                    continue

            print(f"\n  [{q_idx+1}/{len(questions)}] {question[:60]}...")
            t0 = time.time()

            # n=1: single agent, no consensus
            if n_agents == 1:
                persona_id = 0
                try:
                    aeic = llm.generate_aeic(question, persona_id=0, agent_id=0)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
                stance = extract_stance(aeic.get("C", ""))
                elapsed = time.time() - t0

                backends = llm.backends if hasattr(llm, 'backends') else ["deepseek"]
                base = {
                    "dataset": dataset_name, "condition": condition,
                    "question_id": str(q_idx),
                    "question": question, "ground_truth": gt,
                    "n_agents_used": "1",
                    "selected_agents": json.dumps([0]),
                    "per_agent_stance": json.dumps([stance]),
                    "per_agent_ic": json.dumps([]),
                    "per_agent_ec": json.dumps([]),
                    "per_agent_eq": json.dumps([]),
                    "per_agent_theta": json.dumps([]),
                    "per_agent_backend": json.dumps([backends[0] if backends else "unknown"]),
                    "trust_scores": json.dumps(tracker.get_all_scores()),
                    "elapsed_s": f"{elapsed:.2f}",
                }
                for method in methods:
                    mode_results[method].append({
                        "question": question, "ground_truth": gt,
                        "decision": stance, "mechanism": "single_agent",
                    })
                    if csv_dir:
                        _append_unified_row(method_paths[method], {
                            **base, "decision": stance, "mechanism": "single_agent",
                        })

                tracker.update(0, 0.5, 0.5, 0.5, 0.5,
                               agreed_with_consensus=True)
                tracker.increment_question()
                print(f"    GT={gt} | Single={stance} | {elapsed:.1f}s")
                continue

            try:
                result_sg = orch.run(question, mode=mode, topology_W=W,
                                     use_stackelberg=True)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            elapsed = time.time() - t0
            nodes = result_sg["nodes"]

            # AEIC without Stackelberg
            result_no_sg = evaluate_and_decide(nodes, W, sim,
                                               mode=mode, use_stackelberg=False)
            # Quality-weighted vote
            qw_decision, qw_weights = quality_weighted_vote(nodes, sim)
            # Majority vote
            mv_decision = majority_vote(nodes)

            # Compute per-agent metrics for CSV
            ic_list = result_sg.get("ic", [])
            ec_list = result_sg.get("ec", [])
            eq_list = result_sg.get("eq", [])
            theta_list = result_sg.get("theta", [])
            stances_list = result_sg.get("stances", [])

            # Update TrustTracker (using AEIC+SG decision as reference)
            sg_decision = result_sg["decision"]
            actual_n = len(nodes)
            for i in range(actual_n):
                theta_i = theta_list[i] if i < len(theta_list) else 0.5
                ic_i = ic_list[i] if i < len(ic_list) else 0.5
                ec_i = ec_list[i] if i < len(ec_list) else 0.5
                eq_i = eq_list[i] if i < len(eq_list) else 0.5
                stance_i = stances_list[i] if i < len(stances_list) else "neutral"
                tracker.update(i, theta_i, ic_i, ec_i, eq_i,
                               agreed_with_consensus=(stance_i == sg_decision))
            tracker.increment_question()

            backends = llm.backends if hasattr(llm, 'backends') else ["deepseek"]
            base = {
                "dataset": dataset_name, "condition": condition,
                "question_id": str(q_idx),
                "question": question, "ground_truth": gt,
                "n_agents_used": str(actual_n),
                "selected_agents": json.dumps(list(range(actual_n))),
                "per_agent_stance": json.dumps(stances_list),
                "per_agent_ic": json.dumps([round(x, 3) for x in ic_list]),
                "per_agent_ec": json.dumps([round(x, 3) for x in ec_list]),
                "per_agent_eq": json.dumps([round(x, 3) for x in eq_list]),
                "per_agent_theta": json.dumps([round(x, 3) for x in theta_list]),
                "per_agent_backend": json.dumps(
                    [backends[i % len(backends)] for i in range(actual_n)]),
                "trust_scores": json.dumps(tracker.get_all_scores()),
                "elapsed_s": f"{elapsed:.2f}",
            }

            methods_data = [
                ("AEIC+SG", result_sg["decision"], result_sg.get("mechanism", "")),
                ("AEIC", result_no_sg["decision"], result_no_sg.get("mechanism", "")),
                ("QualWt", qw_decision, "quality_weighted"),
                ("MajVote", mv_decision, "majority"),
            ]

            for method, dec, mech in methods_data:
                mode_results[method].append({
                    "question": question, "ground_truth": gt,
                    "decision": dec, "mechanism": mech,
                })
                if csv_dir:
                    _append_unified_row(method_paths[method], {
                        **base, "decision": dec, "mechanism": mech,
                    })

            print(f"    GT={gt} | AEIC+SG={sg_decision}({result_sg['mechanism']}) "
                  f"| AEIC={result_no_sg['decision']} | QualWt={qw_decision} "
                  f"| MajVote={mv_decision} | {elapsed:.1f}s")

        all_results[mode] = mode_results
        print(f"\n  API calls so far: {llm.call_count}")

    llm.close()
    return all_results


def print_summary(all_results: Dict[str, Dict[str, List[Dict]]]):
    """Print accuracy summary table."""
    print(f"\n\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    methods = ["MajVote", "QualWt", "AEIC", "AEIC+SG"]
    modes = list(all_results.keys())

    header = f"{'Mode':<12}" + "".join(f"{m:<14}" for m in methods)
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for mode in modes:
        row = f"  {mode:<12}"
        for method in methods:
            results = all_results[mode][method]
            metrics = evaluate_accuracy(results)
            acc = metrics["accuracy"]
            n = metrics["n"]
            row += f"{acc:.1%} ({metrics.get('correct',0)}/{n})  "
        print(row)


def run_adversarial_experiment(
    questions: List[Dict],
    n_agents: int = 5,
    n_corrupted: int = 2,
    sim_method: str = "embedding",
) -> Dict[str, List[Dict]]:
    """
    Adversarial experiment: inject corrupted agents that flip conclusions.

    Corrupted agents: I says "supports yes" but C says "No" (Tier1 attack).
    Tests whether AEIC quality gate can detect and override.
    """
    llm = MultiLLMPool()
    sim = SimilarityCalculator(method=sim_method)
    W = make_topology("complete", n_agents)

    methods_results = {"AEIC+SG": [], "AEIC": [], "QualWt": [], "MajVote": []}

    print(f"\n  Adversarial: {n_agents} agents, {n_corrupted} corrupted (Tier1: C flip)")

    for q_idx, q_item in enumerate(questions):
        question = q_item["question"]
        gt = q_item["ground_truth"]

        print(f"\n  [{q_idx+1}/{len(questions)}] {question[:55]}...")

        nodes = []
        for i in range(n_agents):
            persona_id = i % 7
            aeic = llm.generate_aeic(question, persona_id=persona_id, agent_id=i)
            nodes.append(aeic)

        for i in range(n_corrupted):
            original_c = nodes[i]["C"]
            stance = extract_stance(original_c)
            if stance == "yes":
                nodes[i]["C"] = "No: After analysis, the answer is negative. " + original_c[4:]
            elif stance == "no":
                nodes[i]["C"] = "Yes: After analysis, the answer is affirmative. " + original_c[3:]
            else:
                nodes[i]["C"] = "No: Unable to confirm. " + original_c

        result_sg = evaluate_and_decide(nodes, W, sim, mode="parallel", use_stackelberg=True)
        result_no_sg = evaluate_and_decide(nodes, W, sim, mode="parallel", use_stackelberg=False)
        qw_decision, _ = quality_weighted_vote(nodes, sim)
        mv_decision = majority_vote(nodes)

        for method, decision, mechanism in [
            ("AEIC+SG", result_sg["decision"], result_sg["mechanism"]),
            ("AEIC", result_no_sg["decision"], result_no_sg["mechanism"]),
            ("QualWt", qw_decision, "quality_weighted"),
            ("MajVote", mv_decision, "majority"),
        ]:
            methods_results[method].append({
                "question": question,
                "ground_truth": gt,
                "decision": decision,
                "mechanism": mechanism,
            })

        print(f"    GT={gt} | AEIC+SG={result_sg['decision']}({result_sg['mechanism']}) "
              f"| AEIC={result_no_sg['decision']}({result_no_sg['mechanism']}) "
              f"| QualWt={qw_decision} | MajVote={mv_decision}")
        print(f"    IC={[f'{x:.2f}' for x in result_sg['ic']]} "
              f"Stances={result_sg['stances']}")

    llm.close()
    return methods_results


def main():
    parser = argparse.ArgumentParser(description="Fixed Topology Experiment")
    parser.add_argument("--dataset", default="StrategyQA",
                        help="Dataset name (default: StrategyQA)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of questions to evaluate")
    parser.add_argument("--n_agents", type=int, default=3,
                        help="Number of agents per question")
    parser.add_argument("--modes", default="parallel,sequential",
                        help="Comma-separated modes to test")
    parser.add_argument("--sim", default="embedding",
                        help="Similarity method")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run adversarial experiment")
    parser.add_argument("--n_corrupted", type=int, default=2,
                        help="Number of corrupted agents in adversarial mode")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path")
    parser.add_argument("--csv_dir", default=None,
                        help="CSV output directory (enables per-method files + checkpoint)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    print(f"Fixed Topology Experiment")
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Agents: {args.n_agents}")
    print(f"  Modes: {modes}")
    print(f"  Similarity: {args.sim}")

    print(f"\nLoading {args.dataset} (offset={args.offset})...")
    questions = load_questions(args.dataset, args.n_samples, offset=args.offset)
    print(f"  Loaded {len(questions)} questions")

    calls_per_q = {
        "parallel": args.n_agents,
        "sequential": args.n_agents,
        "star": args.n_agents + 1,
        "debate": args.n_agents * 2,
        "selfcorrect": args.n_agents * 3,
        "agentverse": args.n_agents + 1,
    }
    total_est = sum(calls_per_q.get(m, args.n_agents) for m in modes) * len(questions)
    print(f"  Estimated API calls: ~{total_est}")

    csv_dir = args.csv_dir
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
        print(f"  CSV output: {csv_dir}/fixed_{{method}}_{args.dataset}.csv")

    if args.adversarial:
        print(f"\n  === ADVERSARIAL MODE ===")
        adv_results = run_adversarial_experiment(
            questions, n_agents=args.n_agents,
            n_corrupted=args.n_corrupted, sim_method=args.sim
        )
        print(f"\n\n{'='*70}")
        print(f"  ADVERSARIAL RESULTS ({args.n_corrupted}/{args.n_agents} corrupted)")
        print(f"{'='*70}")
        methods = ["MajVote", "QualWt", "AEIC", "AEIC+SG"]
        header = "  " + "".join(f"{m:<14}" for m in methods)
        print(f"\n{header}")
        print(f"  {'-'*56}")
        row = "  "
        for method in methods:
            metrics = evaluate_accuracy(adv_results[method])
            row += f"{metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['n']})  "
        print(row)
        all_results = {"adversarial": adv_results}
    else:
        all_results = run_experiment(questions, modes, args.n_agents, args.sim,
                                     csv_dir=csv_dir,
                                     dataset_name=args.dataset)
        print_summary(all_results)

    # Save raw results
    if args.output:
        out_path = args.output
    else:
        os.makedirs("results", exist_ok=True)
        out_path = f"experiments/results/exp_fixed_{args.dataset}_{len(questions)}q.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable = {}
    for mode, method_results in all_results.items():
        serializable[mode] = {}
        for method, results in method_results.items():
            serializable[mode][method] = [
                {k: convert(v) if not isinstance(v, list)
                 else [convert(x) for x in v]
                 for k, v in r.items()}
                for r in results
            ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
