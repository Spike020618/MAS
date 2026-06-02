"""
实验脚本 — AEIC-Stackelberg 四种编排模式对比

对比：
  方法: MajVote, QualWt, AEIC (no SG), AEIC+SG
  模式: parallel, sequential, star, debate
  数据集: StrategyQA (boolean GT, yes/no questions)

Usage:
  python experiments/exp_orchestrator.py --n_samples 20 --n_agents 3
  python experiments/exp_orchestrator.py --dataset StrategyQA --modes parallel,sequential
  python experiments/exp_orchestrator.py --dataset StrategyQA --n_samples 20 --csv results/set1.csv
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
from core.orchestrator import Orchestrator
from core.similarity import SimilarityCalculator
from core.topology import make_topology
from core.trust import (
    evaluate_and_decide,
    majority_vote,
    quality_weighted_vote,
    extract_stance,
)


# ─── CSV断点续跑 ────────────────────────────────────────────────────

SET1_FIELDS = [
    "dataset", "mode", "n_agents", "method", "question_id",
    "question", "ground_truth", "decision", "mechanism",
    "stances", "ic", "ec", "trust", "theta", "elapsed_s",
    "trace",
]


def load_completed(csv_path: str) -> Set[Tuple]:
    """加载已完成的 (dataset, mode, n_agents, method, question_id) 组合。"""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["mode"], row["n_agents"],
                   row["method"], row["question_id"])
            completed.add(key)
    return completed


def append_csv_row(csv_path: str, row: Dict, fieldnames: List[str] = None):
    """追加一行到CSV（首次自动写header）。"""
    if fieldnames is None:
        fieldnames = SET1_FIELDS
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _serialize_list(lst) -> str:
    """将list序列化为字符串存入CSV单元格。"""
    if lst is None:
        return ""
    if isinstance(lst, np.ndarray):
        lst = lst.tolist()
    return json.dumps(lst, ensure_ascii=False)


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
            gt_stance = "yes" if item.get("ground_truth", True) else "no"
            questions.append({"question": q, "ground_truth": gt_stance,
                              "source": dataset_name})
    elif dataset_name == "HotpotQA":
        for item in raw[offset:]:
            q = item.get("question", "")
            ans = str(item.get("answer", "")).strip().lower()
            if ans in ("yes", "no"):
                questions.append({"question": q, "ground_truth": ans,
                                  "source": dataset_name})
            else:
                # Convert factoid to yes/no: "Is the answer to '{q}' equal to '{ans}'?"
                yes_q = f"For the question '{q}', is the answer '{item['answer']}'?"
                questions.append({"question": yes_q, "ground_truth": "yes",
                                  "source": dataset_name})
    elif dataset_name == "GSM8K":
        for item in raw[offset:]:
            q = item.get("question", "")
            ans = str(item.get("answer", ""))
            # Convert: "Is the answer to this math problem X?"
            yes_q = f"{q} Is the answer {ans}?"
            questions.append({"question": yes_q, "ground_truth": "yes",
                              "source": dataset_name})
            # Also add a wrong answer version
            wrong_ans = int(ans) + rng.choice([-5, -3, -2, 2, 3, 5, 10]) if ans.lstrip('-').isdigit() else "0"
            no_q = f"{q} Is the answer {wrong_ans}?"
            questions.append({"question": no_q, "ground_truth": "no",
                              "source": dataset_name})
        rng.shuffle(questions)
    else:
        # MCQ datasets (GPQA, MMLU, ARC-Challenge, CommonsenseQA)
        # Half: "Is [correct] the answer?" (GT=yes)
        # Half: "Is [wrong] the answer?" (GT=no)
        for item in raw[offset:]:
            q = item.get("question", "")
            correct = item.get("correct_answer", "")
            choices = item.get("choices", [])
            if not q or not correct:
                continue

            yes_q = f"For the question '{q}', is the correct answer '{correct}'?"
            questions.append({"question": yes_q, "ground_truth": "yes",
                              "source": dataset_name})

            wrong_choices = [c for c in choices if c != correct]
            if wrong_choices:
                wrong = rng.choice(wrong_choices)
                no_q = f"For the question '{q}', is the correct answer '{wrong}'?"
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


def run_experiment(
    questions: List[Dict],
    modes: List[str],
    n_agents: int = 3,
    sim_method: str = "tfidf",
    multi_llm: bool = True,
    csv_path: str = None,
    dataset_name: str = "",
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Run all modes × all methods on the questions.

    Returns: {mode: {method: [results]}}
    """
    completed = load_completed(csv_path) if csv_path else set()
    if completed:
        print(f"  CSV checkpoint: {len(completed)} rows already completed")

    if multi_llm:
        llm = MultiLLMPool()
    else:
        llm = LLMClient(backend="deepseek")
    sim = SimilarityCalculator(method=sim_method)
    orch = Orchestrator(llm, sim, n_agents=n_agents)

    topology_map = {
        "parallel": "complete",
        "sequential": "path",
        "star": "star",
        "debate": "complete",
    }

    all_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*60}")

        topo_name = topology_map[mode]
        n = n_agents if mode != "star" else n_agents + 1
        W = make_topology(topo_name, n)

        mode_results = {"AEIC+SG": [], "AEIC": [], "QualWt": [], "MajVote": []}

        for q_idx, q_item in enumerate(questions):
            question = q_item["question"]
            gt = q_item["ground_truth"]

            # 断点续跑: 检查该题所有method是否都已完成
            if csv_path:
                all_done = all(
                    (dataset_name, mode, str(n_agents), method, str(q_idx))
                    in completed
                    for method in ["AEIC+SG", "AEIC", "QualWt", "MajVote"]
                )
                if all_done:
                    continue

            print(f"\n  [{q_idx+1}/{len(questions)}] {question[:60]}...")
            t0 = time.time()

            try:
                result_sg = orch.run(question, mode=mode, topology_W=W,
                                     use_stackelberg=True)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            elapsed = time.time() - t0
            nodes = result_sg["nodes"]

            # Method 1: AEIC + Stackelberg
            mode_results["AEIC+SG"].append({
                "question": question,
                "ground_truth": gt,
                "decision": result_sg["decision"],
                "mechanism": result_sg["mechanism"],
                "trust": result_sg["trust"],
                "stances": result_sg["stances"],
            })

            # Method 2: AEIC without Stackelberg
            result_no_sg = evaluate_and_decide(nodes, W, sim,
                                               mode=mode, use_stackelberg=False)
            mode_results["AEIC"].append({
                "question": question,
                "ground_truth": gt,
                "decision": result_no_sg["decision"],
                "mechanism": result_no_sg["mechanism"],
                "trust": result_no_sg["trust"],
                "stances": result_no_sg["stances"],
            })

            # Method 3: Quality-weighted vote
            qw_decision, qw_weights = quality_weighted_vote(nodes, sim)
            mode_results["QualWt"].append({
                "question": question,
                "ground_truth": gt,
                "decision": qw_decision,
                "mechanism": "quality_weighted",
                "trust": qw_weights,
                "stances": [extract_stance(nd.get("C", "")) for nd in nodes],
            })

            # Method 4: Simple majority vote
            mv_decision = majority_vote(nodes)
            mode_results["MajVote"].append({
                "question": question,
                "ground_truth": gt,
                "decision": mv_decision,
                "mechanism": "majority",
                "stances": [extract_stance(nd.get("C", "")) for nd in nodes],
            })

            # CSV即时写入
            if csv_path:
                # 构建trace JSON
                agent_times = result_sg.get("agent_times", [0.0] * n_agents)
                timing = result_sg.get("timing", {})
                backends = llm.backends if hasattr(llm, 'backends') else ["deepseek"]
                trace = {
                    "agents": [
                        {
                            "id": i,
                            "role": AGENT_PERSONAS[i % len(AGENT_PERSONAS)]["name"],
                            "model": backends[i % len(backends)] if backends else "unknown",
                            "llm_time_s": round(agent_times[i], 2) if i < len(agent_times) else 0,
                            "stance": result_sg.get("stances", [])[i] if i < len(result_sg.get("stances", [])) else "",
                            "ic": round(result_sg.get("ic", [])[i], 3) if i < len(result_sg.get("ic", [])) else 0,
                            "ec": round(result_sg.get("ec", [])[i], 3) if i < len(result_sg.get("ec", [])) else 0,
                        }
                        for i in range(n_agents)
                    ],
                    "total_llm_time_s": round(sum(agent_times), 2),
                    "embedding_time_s": round(timing.get("ic_ec_time_s", 0) + timing.get("evidence_time_s", 0), 3),
                    "stackelberg_time_s": round(timing.get("stackelberg_time_s", 0), 4),
                    "consensus_time_s": round(sum(timing.values()), 3),
                    "overhead_s": round(max(0, elapsed - sum(agent_times) - sum(timing.values())), 2),
                }
                trace_json = json.dumps(trace, ensure_ascii=False)

                methods_data = [
                    ("AEIC+SG", result_sg["decision"], result_sg["mechanism"],
                     result_sg.get("stances", []), result_sg.get("ic", []),
                     result_sg.get("ec", []), result_sg.get("trust", []),
                     result_sg.get("theta", [])),
                    ("AEIC", result_no_sg["decision"], result_no_sg["mechanism"],
                     result_no_sg.get("stances", []), result_no_sg.get("ic", []),
                     result_no_sg.get("ec", []), result_no_sg.get("trust", []),
                     result_no_sg.get("theta", [])),
                    ("QualWt", qw_decision, "quality_weighted",
                     [extract_stance(nd.get("C", "")) for nd in nodes],
                     [], [], qw_weights, []),
                    ("MajVote", mv_decision, "majority",
                     [extract_stance(nd.get("C", "")) for nd in nodes],
                     [], [], [], []),
                ]
                for (method, dec, mech, stances, ic, ec, trust, theta) in methods_data:
                    row = {
                        "dataset": dataset_name,
                        "mode": mode,
                        "n_agents": str(n_agents),
                        "method": method,
                        "question_id": str(q_idx),
                        "question": question,
                        "ground_truth": gt,
                        "decision": dec,
                        "mechanism": mech,
                        "stances": _serialize_list(stances),
                        "ic": _serialize_list(ic),
                        "ec": _serialize_list(ec),
                        "trust": _serialize_list(trust),
                        "theta": _serialize_list(theta),
                        "elapsed_s": f"{elapsed:.2f}",
                        "trace": trace_json,
                    }
                    append_csv_row(csv_path, row)

            print(f"    GT={gt} | AEIC+SG={result_sg['decision']}({result_sg['mechanism']}) "
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

    # Header
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

    # Override statistics
    print(f"\n  Quality Override Analysis:")
    print(f"  {'Mode':<12}{'AEIC+SG overrides':<22}{'AEIC overrides'}")
    print(f"  {'-'*50}")
    for mode in modes:
        sg_overrides = sum(1 for r in all_results[mode]["AEIC+SG"]
                          if r["mechanism"] == "quality_override")
        no_sg_overrides = sum(1 for r in all_results[mode]["AEIC"]
                             if r["mechanism"] == "quality_override")
        n = len(all_results[mode]["AEIC+SG"])
        print(f"  {mode:<12}{sg_overrides}/{n:<20}{no_sg_overrides}/{n}")


def run_adversarial_experiment(
    questions: List[Dict],
    n_agents: int = 5,
    n_corrupted: int = 2,
    sim_method: str = "tfidf",
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

        # Generate all agents' AEIC honestly
        nodes = []
        for i in range(n_agents):
            persona_id = i % 7
            aeic = llm.generate_aeic(question, persona_id=persona_id, agent_id=i)
            nodes.append(aeic)

        # Corrupt first n_corrupted agents: flip their C (Tier1 attack)
        for i in range(n_corrupted):
            original_c = nodes[i]["C"]
            stance = extract_stance(original_c)
            if stance == "yes":
                nodes[i]["C"] = "No: After analysis, the answer is negative. " + original_c[4:]
            elif stance == "no":
                nodes[i]["C"] = "Yes: After analysis, the answer is affirmative. " + original_c[3:]
            else:
                nodes[i]["C"] = "No: Unable to confirm. " + original_c

        # Evaluate with all methods
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
    parser = argparse.ArgumentParser(description="AEIC-Stackelberg Orchestration Experiment")
    parser.add_argument("--dataset", default="StrategyQA",
                        help="Dataset name (default: StrategyQA)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of questions to evaluate")
    parser.add_argument("--n_agents", type=int, default=3,
                        help="Number of agents per question")
    parser.add_argument("--modes", default="parallel,sequential",
                        help="Comma-separated modes to test")
    parser.add_argument("--sim", default="embedding",
                        help="Similarity method (tfidf/bm25/embedding/api_embedding)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run adversarial experiment (inject corrupted agents)")
    parser.add_argument("--n_corrupted", type=int, default=2,
                        help="Number of corrupted agents in adversarial mode")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path")
    parser.add_argument("--csv", default=None,
                        help="CSV output path (enables checkpoint/resume)")
    parser.add_argument("--single_llm", action="store_true",
                        help="Use single LLM backend (DeepSeek only)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    print(f"AEIC-Stackelberg Experiment")
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Agents: {args.n_agents}")
    print(f"  Modes: {modes}")
    print(f"  Similarity: {args.sim}")

    # Load questions
    print(f"\nLoading {args.dataset} (offset={args.offset})...")
    questions = load_questions(args.dataset, args.n_samples, offset=args.offset)
    print(f"  Loaded {len(questions)} questions")

    # Estimate API calls
    calls_per_q = {
        "parallel": args.n_agents,
        "sequential": args.n_agents,
        "star": args.n_agents + 1,
        "debate": args.n_agents * 2,  # assume 2 rounds avg
    }
    total_est = sum(calls_per_q.get(m, args.n_agents) for m in modes) * len(questions)
    print(f"  Estimated API calls: ~{total_est}")

    # CSV路径处理
    csv_path = args.csv
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or "results", exist_ok=True)

    # Run
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

        print(f"\n  Override stats:")
        for method in ["AEIC", "AEIC+SG"]:
            overrides = sum(1 for r in adv_results[method] if r["mechanism"] == "quality_override")
            print(f"    {method}: {overrides} quality overrides")

        all_results = {"adversarial": adv_results}
    else:
        all_results = run_experiment(questions, modes, args.n_agents, args.sim,
                                     multi_llm=not args.single_llm,
                                     csv_path=csv_path,
                                     dataset_name=args.dataset)
        print_summary(all_results)

    # Save raw results
    if args.output:
        out_path = args.output
    else:
        os.makedirs("results", exist_ok=True)
        out_path = f"results/exp_orch_{args.dataset}_{len(questions)}q.json"

    # Convert numpy arrays for JSON serialization
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
