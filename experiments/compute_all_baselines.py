"""
Compute ALL baselines from existing cached data (no API calls needed).

Baselines:
  1. Single Agent — agent 0's stance vs ground truth
  2. Majority Vote (no Stackelberg) — per topology, pure majority vote
  3. AEIC + Default θ — per topology, default ConsensusConfig
  4. AEIC + Optimized θ* — per topology, optimized ConsensusConfig
  5. Weighted Vote (confidence-based, no game theory)
  6. Oracle — any topology gets it right
  7. Adaptive Router — from V9 results
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config

SIGNALS_DIR = "results/signals"
CONFIGS_DIR = "results/configs"
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]
TOPOS_BASELINE = ["parallel", "debate"]  # for ChatEval / UnguidedDebate


def load_signals(dataset, topology):
    if topology == "parallel":
        path = f"{SIGNALS_DIR}/{dataset}.json"
    else:
        path = f"{SIGNALS_DIR}/{dataset}_{topology}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def single_agent_baseline(questions):
    """Agent 0's stance vs ground truth."""
    correct = total = 0
    for q in questions:
        if "signals" not in q:
            continue
        stances = q["signals"]["stances"]
        gt = q["ground_truth"]
        if len(stances) > 0:
            agent0_stance = stances[0].lower()
            if agent0_stance == gt.lower():
                correct += 1
        total += 1
    return correct, total


def majority_vote(questions):
    """Simple majority vote on stances."""
    correct = total = 0
    for q in questions:
        if "signals" not in q:
            continue
        stances = [s.lower() for s in q["signals"]["stances"]]
        gt = q["ground_truth"].lower()

        counts = Counter(stances)
        if not counts:
            total += 1
            continue

        majority = counts.most_common(1)[0][0]
        if majority == gt:
            correct += 1
        total += 1
    return correct, total


def weighted_vote(questions):
    """Confidence-weighted vote using IC scores (no Stackelberg)."""
    correct = total = 0
    for q in questions:
        if "signals" not in q:
            continue
        s = q["signals"]
        stances = [st.lower() for st in s["stances"]]
        ic = np.array(s["ic"])
        gt = q["ground_truth"].lower()

        # Weight each agent's vote by their IC score
        vote_scores = {}
        for i, stance in enumerate(stances):
            w = ic[i] if i < len(ic) else 0.5
            vote_scores[stance] = vote_scores.get(stance, 0) + w

        if vote_scores:
            decision = max(vote_scores, key=vote_scores.get)
            if decision == gt:
                correct += 1
        total += 1
    return correct, total


def agreement_weighted_vote(questions):
    """Vote weighted by agreement scores (no Stackelberg)."""
    correct = total = 0
    for q in questions:
        if "signals" not in q:
            continue
        s = q["signals"]
        stances = [st.lower() for st in s["stances"]]
        agr = np.array(s["agreement"])
        gt = q["ground_truth"].lower()

        vote_scores = {}
        for i, stance in enumerate(stances):
            w = agr[i] if i < len(agr) else 0.5
            vote_scores[stance] = vote_scores.get(stance, 0) + w

        if vote_scores:
            decision = max(vote_scores, key=vote_scores.get)
            if decision == gt:
                correct += 1
        total += 1
    return correct, total


def aeic_default(questions, n_agents):
    """AEIC + default ConsensusConfig (no optimization)."""
    cfg = ConsensusConfig()  # all defaults
    valid_qs = [q for q in questions if "signals" in q]
    result = evaluate_config(valid_qs, cfg, n_agents)
    return result["correct"], result["total"]


def aeic_optimized(questions, n_agents, topo):
    """AEIC + optimized θ* for this topology."""
    cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
    if not os.path.exists(cfg_path):
        return 0, 0
    cfg = ConsensusConfig.load(cfg_path)
    valid_qs = [q for q in questions if "signals" in q]
    result = evaluate_config(valid_qs, cfg, n_agents)
    return result["correct"], result["total"]


def oracle_any_topo(all_data, n_q=300):
    """Oracle: correct if ANY topology gets it right (with optimized θ*)."""
    correct = total = 0
    for q_idx in range(n_q):
        any_correct = False
        valid = True
        for topo in TOPOS:
            if topo not in all_data or all_data[topo] is None:
                valid = False
                break
            questions = all_data[topo]["questions"]
            if q_idx >= len(questions):
                valid = False
                break
            q = questions[q_idx]
            if "signals" not in q:
                valid = False
                break

            n_agents = all_data[topo]["meta"]["n_agents"]
            cfg_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
            if os.path.exists(cfg_path):
                cfg = ConsensusConfig.load(cfg_path)
            else:
                cfg = ConsensusConfig()
            res = evaluate_config([q], cfg, n_agents)
            if res["correct"] > 0:
                any_correct = True

        if not valid:
            continue
        if any_correct:
            correct += 1
        total += 1
    return correct, total


def main():
    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 80)
    print("COMPLETE BASELINE COMPARISON (from cached data, no API calls)")
    print("=" * 80)

    # ================================================================
    # Per-dataset, per-method results
    # ================================================================
    all_results = {}

    for ds in DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {ds}")
        print(f"{'='*80}")

        results = {}
        all_data = {}

        for topo in TOPOS:
            data = load_signals(ds, topo)
            all_data[topo] = data

        # Also load agentverse
        agentverse_data = load_signals(ds, "agentverse")

        # --- 1. Single Agent (agent 0, parallel topology) ---
        par_data = all_data.get("parallel")
        if par_data:
            questions = par_data["questions"][:300]
            c, t = single_agent_baseline(questions)
            results["Single Agent"] = (c, t)
            print(f"  Single Agent (agent 0):    {c}/{t} = {c/t*100:.1f}%")

        # --- 2. Majority Vote per topology ---
        for topo in TOPOS:
            data = all_data.get(topo)
            if data:
                questions = data["questions"][:300]
                c, t = majority_vote(questions)
                results[f"MajVote_{topo}"] = (c, t)
                print(f"  MajVote ({topo:12s}):  {c}/{t} = {c/t*100:.1f}%")

        # --- 3. Weighted Vote (IC-weighted, parallel) ---
        if par_data:
            questions = par_data["questions"][:300]
            c, t = weighted_vote(questions)
            results["WeightedVote_IC"] = (c, t)
            print(f"  Weighted Vote (IC):        {c}/{t} = {c/t*100:.1f}%")

            c, t = agreement_weighted_vote(questions)
            results["WeightedVote_Agr"] = (c, t)
            print(f"  Weighted Vote (Agreement): {c}/{t} = {c/t*100:.1f}%")

        # --- 4. AEIC + Default θ per topology ---
        for topo in TOPOS:
            data = all_data.get(topo)
            if data:
                questions = data["questions"][:300]
                n_agents = data["meta"]["n_agents"]
                c, t = aeic_default(questions, n_agents)
                results[f"AEIC_default_{topo}"] = (c, t)
                print(f"  AEIC+Default ({topo:12s}): {c}/{t} = {c/t*100:.1f}%")

        # --- 5. AEIC + Optimized θ* per topology ---
        for topo in TOPOS:
            data = all_data.get(topo)
            if data:
                questions = data["questions"][:300]
                n_agents = data["meta"]["n_agents"]
                c, t = aeic_optimized(questions, n_agents, topo)
                results[f"AEIC_opt_{topo}"] = (c, t)
                print(f"  AEIC+θ* ({topo:12s}):     {c}/{t} = {c/t*100:.1f}%")

        # --- 6. Baseline reimplementations ---
        # ChatEval = parallel + majority vote (already computed above)
        # UnguidedDebate = debate + majority vote (already computed above)

        # AgentVerse = agentverse topology + solver decision
        if agentverse_data:
            from core.trust import extract_stance
            c_av = t_av = 0
            for q in agentverse_data["questions"][:300]:
                agents = q.get("agents", [])
                if not agents:
                    continue
                gt = q["ground_truth"]
                solver_aeic = agents[0].get("aeic", {})
                solver_conclusion = solver_aeic.get("C", "")
                decision = extract_stance(solver_conclusion)
                if decision == gt:
                    c_av += 1
                t_av += 1
            results["AgentVerse_solver"] = (c_av, t_av)
            print(f"  AgentVerse (solver):       {c_av}/{t_av} = {c_av/t_av*100:.1f}%")

        # --- 7. Oracle (any topo) ---
        c, t = oracle_any_topo(all_data)
        results["Oracle_5topo"] = (c, t)
        print(f"  Oracle (5-topo):           {c}/{t} = {c/t*100:.1f}%")

        all_results[ds] = results

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE (for paper)")
    print(f"{'='*80}")

    # Methods to show in final table
    methods = [
        ("Single Agent", "Single Agent"),
        ("MajVote (parallel)", "MajVote_parallel"),
        ("MajVote (debate)", "MajVote_debate"),
        ("MajVote (star)", "MajVote_star"),
        ("AgentVerse (solver)", "AgentVerse_solver"),
        ("WeightedVote (IC)", "WeightedVote_IC"),
        ("WeightedVote (Agr)", "WeightedVote_Agr"),
        ("AEIC+Default (star)", "AEIC_default_star"),
        ("AEIC+θ* (parallel)", "AEIC_opt_parallel"),
        ("AEIC+θ* (sequential)", "AEIC_opt_sequential"),
        ("AEIC+θ* (star)", "AEIC_opt_star"),
        ("AEIC+θ* (debate)", "AEIC_opt_debate"),
        ("AEIC+θ* (hierarchical)", "AEIC_opt_hierarchical"),
        ("Oracle (5-topo)", "Oracle_5topo"),
    ]

    header = f"{'Method':<28}"
    for ds in DATASETS:
        header += f" {ds:>12}"
    header += f" {'Avg':>8}"
    print(header)
    print("-" * len(header))

    for display_name, key in methods:
        line = f"{display_name:<28}"
        accs = []
        for ds in DATASETS:
            if key in all_results[ds]:
                c, t = all_results[ds][key]
                acc = c / t if t > 0 else 0
                accs.append(acc)
                line += f" {acc*100:>11.1f}%"
            else:
                line += f" {'N/A':>12}"
        if accs:
            avg = sum(accs) / len(accs)
            line += f" {avg*100:>7.1f}%"
        print(line)

    # ================================================================
    # Paper-ready comparison table
    # ================================================================
    print(f"\n\n{'='*80}")
    print("PAPER TABLE 1 — Method Comparison (same LLM, same data)")
    print(f"{'='*80}")

    paper_methods = [
        ("Single Agent (N=1)", "Single Agent"),
        ("Majority Vote†", "MajVote_parallel"),
        ("Multi-round Debate†", "MajVote_debate"),
        ("AgentVerse†", "AgentVerse_solver"),
        ("Weighted Vote (IC)", "WeightedVote_IC"),
        ("AEIC + Default θ", "AEIC_default_star"),
        ("AEIC + SG (star)", "AEIC_opt_star"),
        ("AEIC + SG (best topo)", None),  # special: best per dataset
        ("Oracle (5-topo)", "Oracle_5topo"),
    ]

    header = f"{'Method':<28}"
    for ds in DATASETS:
        header += f" {ds:>12}"
    header += f" {'Avg':>8}"
    print(header)
    print("-" * len(header))

    for display_name, key in paper_methods:
        line = f"{display_name:<28}"
        accs = []

        if key is None:
            # Best AEIC+SG across all topologies
            for ds in DATASETS:
                best_acc = 0
                for topo in TOPOS:
                    k = f"AEIC_opt_{topo}"
                    if k in all_results[ds]:
                        c, t = all_results[ds][k]
                        acc = c / t if t > 0 else 0
                        if acc > best_acc:
                            best_acc = acc
                accs.append(best_acc)
                line += f" {best_acc*100:>11.1f}%"
        else:
            for ds in DATASETS:
                if key in all_results[ds]:
                    c, t = all_results[ds][key]
                    acc = c / t if t > 0 else 0
                    accs.append(acc)
                    line += f" {acc*100:>11.1f}%"
                else:
                    line += f" {'N/A':>12}"

        if accs:
            avg = sum(accs) / len(accs)
            line += f" {avg*100:>7.1f}%"
        print(line)

    print("\n† = Our reimplementation using DeepSeek-V3")
    print("SG = Stackelberg Game optimized θ*")

    # ================================================================
    # Save to JSON
    # ================================================================
    output = {}
    for ds in DATASETS:
        output[ds] = {}
        for key, (c, t) in all_results[ds].items():
            output[ds][key] = {"correct": c, "total": t, "accuracy": c/t if t > 0 else 0}

    out_path = "results/all_baselines.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
