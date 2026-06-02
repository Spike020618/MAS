"""
Real Multi-Agent Consensus Experiment

使用 DeepSeek API 进行真实 LLM 推理，测试不同 agent 数量和网络拓扑
对共识质量的影响。

实验变量:
  - Agent 数量: 3, 5, 7
  - 网络拓扑: complete, ring, star
  - 共识方法: Majority Vote, Quality-Weighted, AEIC Trust, AEIC+Stackelberg
  - 对抗场景: 正常 vs 腐蚀agent

Usage:
  python3 experiments/exp_real.py [--sim tfidf|embedding]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import time
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from core.llm import LLMClient, AGENT_PERSONAS
from core.topology import make_topology, laplacian_lambda2
from core.stackelberg import StackelbergGame, AgentBid
from core.similarity import SimilarityCalculator
from experiments.plot_config import ieee_style, DOUBLE_COL, PALETTE, add_subfig_label

ieee_style()

# ═══════════════════════════════════════════════════════════════
# 数据集: 30道精选布尔判断题
# ═══════════════════════════════════════════════════════════════
QUESTIONS = [
    # === Easy (10): 明确事实 ===
    {"q": "Is water composed of hydrogen and oxygen?", "a": True, "diff": "easy"},
    {"q": "Is the sun a planet?", "a": False, "diff": "easy"},
    {"q": "Does the Earth revolve around the Sun?", "a": True, "diff": "easy"},
    {"q": "Is Python a compiled programming language?", "a": False, "diff": "easy"},
    {"q": "Did World War II end before 1940?", "a": False, "diff": "easy"},
    {"q": "Is Mount Everest the tallest mountain on Earth?", "a": True, "diff": "easy"},
    {"q": "Are dolphins mammals?", "a": True, "diff": "easy"},
    {"q": "Is mercury a gas at room temperature?", "a": False, "diff": "easy"},
    {"q": "Does light travel faster than sound?", "a": True, "diff": "easy"},
    {"q": "Is the chemical symbol for gold 'Au'?", "a": True, "diff": "easy"},
    # === Medium (10): 需要推理 ===
    {"q": "Is the boiling point of water always 100 degrees Celsius?", "a": False, "diff": "medium"},
    {"q": "Do all mammals give live birth?", "a": False, "diff": "medium"},
    {"q": "Is the Dead Sea actually a sea?", "a": False, "diff": "medium"},
    {"q": "Can some metals be liquid at room temperature?", "a": True, "diff": "medium"},
    {"q": "Are all prime numbers odd?", "a": False, "diff": "medium"},
    {"q": "Is Antarctica a continent?", "a": True, "diff": "medium"},
    {"q": "Do penguins live in the Arctic?", "a": False, "diff": "medium"},
    {"q": "Can sound travel through a vacuum?", "a": False, "diff": "medium"},
    {"q": "Is a tomato a fruit?", "a": True, "diff": "medium"},
    {"q": "Does the human body contain more bacteria than human cells?", "a": False, "diff": "medium"},
    # === Hard (10): 常见误区 ===
    {"q": "Is the Great Wall of China visible from space with the naked eye?", "a": False, "diff": "hard"},
    {"q": "Did Albert Einstein fail mathematics in school?", "a": False, "diff": "hard"},
    {"q": "Do humans only use 10 percent of their brain?", "a": False, "diff": "hard"},
    {"q": "Does lightning never strike the same place twice?", "a": False, "diff": "hard"},
    {"q": "Do goldfish have a three-second memory?", "a": False, "diff": "hard"},
    {"q": "Is a banana a berry in botanical terms?", "a": True, "diff": "hard"},
    {"q": "Does sugar cause hyperactivity in children?", "a": False, "diff": "hard"},
    {"q": "Did Vikings wear horned helmets?", "a": False, "diff": "hard"},
    {"q": "Is glass a slowly flowing liquid?", "a": False, "diff": "hard"},
    {"q": "Do we lose most body heat through our head?", "a": False, "diff": "hard"},
]

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache_llm_responses.json")

AGENT_COUNTS = [3, 5, 7]
TOPOLOGIES = ["complete", "ring", "star"]
METHOD_NAMES = ["MajVote", "QualWt", "AEIC", "AEIC+SG"]


# ═══════════════════════════════════════════════════════════════
# Phase 1: LLM Response Generation
# ═══════════════════════════════════════════════════════════════

def load_cache() -> Dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def generate_all_responses(questions: List[Dict], n_agents: int = 7) -> Dict:
    """Call DeepSeek for all (question, persona) pairs, with caching."""
    cache = load_cache()
    client = LLMClient(mode="deepseek")

    new_calls = 0
    cached_hits = 0
    total = len(questions) * n_agents

    print(f"\n  Generating {total} LLM responses ({n_agents} agents × {len(questions)} questions)...")

    for q_idx, q in enumerate(questions):
        for agent_id in range(n_agents):
            key = f"q{q_idx}_a{agent_id}"
            if key in cache:
                cached_hits += 1
                continue

            result = client.generate_aeic(q["q"], persona_id=agent_id)
            cache[key] = result
            new_calls += 1

            if new_calls % 10 == 0:
                save_cache(cache)
                print(f"    Progress: {cached_hits + new_calls}/{total} "
                      f"({new_calls} new API calls)")
            time.sleep(0.3)

    save_cache(cache)
    print(f"  Done: {cached_hits} cached, {new_calls} new API calls")
    return cache


# ═══════════════════════════════════════════════════════════════
# Stance Extraction & Evaluation
# ═══════════════════════════════════════════════════════════════

def extract_stance(conclusion: str) -> str:
    """Extract Yes/No stance from conclusion text."""
    c = conclusion.strip().lower()
    if c.startswith("yes"):
        return "yes"
    if c.startswith("no"):
        return "no"
    # Chinese signals
    yes_signals = ["是的", "正确", "对", "肯定", "yes", "true", "确实"]
    no_signals = ["不是", "错误", "否", "不对", "no", "false", "不正确", "并非"]
    yes_score = sum(1 for s in yes_signals if s in c[:50])
    no_score = sum(1 for s in no_signals if s in c[:50])
    if yes_score > no_score:
        return "yes"
    if no_score > yes_score:
        return "no"
    return "yes"


def check_accuracy(stance: str, ground_truth: bool) -> bool:
    return (stance == "yes") == ground_truth


# ═══════════════════════════════════════════════════════════════
# Phase 2: Consensus Methods
# ═══════════════════════════════════════════════════════════════

def majority_vote(nodes: List[Dict]) -> Tuple[str, Dict]:
    """Simple majority vote on conclusions."""
    stances = [extract_stance(n["C"]) for n in nodes]
    counter = Counter(stances)
    decision = counter.most_common(1)[0][0]
    return decision, {"method": "MajVote", "agreement": counter[decision] / len(nodes)}


def quality_weighted(nodes: List[Dict], sim_calc: SimilarityCalculator) -> Tuple[str, Dict]:
    """Quality-weighted vote (like AgentVerse evaluator)."""
    scores = []
    for node in nodes:
        e_text = " ".join(node["E"]) if isinstance(node["E"], list) else str(node["E"])
        # Evidence coverage
        e_count = len(node["E"]) if isinstance(node["E"], list) else 1
        coverage = min(e_count / 4.0, 1.0)
        # Reasoning coherence (A→I similarity)
        coherence = sim_calc.compute(str(node["A"]), str(node["I"]))
        # E→C support
        ec_support = sim_calc.compute(e_text, str(node["C"]))
        quality = 0.3 * coverage + 0.35 * coherence + 0.35 * ec_support
        scores.append(quality)

    best_idx = int(np.argmax(scores))
    decision = extract_stance(nodes[best_idx]["C"])
    return decision, {"method": "QualWt", "scores": scores, "selected": best_idx}


def aeic_trust(nodes: List[Dict], sim_calc: SimilarityCalculator,
               W: np.ndarray = None) -> Tuple[str, Dict]:
    """AEIC trust evaluation without Stackelberg (ablation)."""
    n = len(nodes)
    if W is None:
        W = np.full((n, n), 1.0 / max(n - 1, 1))
        np.fill_diagonal(W, 0.0)

    trust_scores = _compute_trust(nodes, W, sim_calc)

    # Trust-weighted vote
    stances = [extract_stance(node["C"]) for node in nodes]
    vote_scores = {}
    for i, s in enumerate(stances):
        vote_scores[s] = vote_scores.get(s, 0.0) + trust_scores[i]
    decision = max(vote_scores, key=vote_scores.get)

    return decision, {"method": "AEIC", "trust": trust_scores}


def aeic_stackelberg(nodes: List[Dict], sim_calc: SimilarityCalculator,
                     W: np.ndarray = None) -> Tuple[str, Dict]:
    """Full AEIC + Stackelberg with topology-aware consensus."""
    n = len(nodes)
    if W is None:
        W = np.full((n, n), 1.0 / max(n - 1, 1))
        np.fill_diagonal(W, 0.0)

    trust_scores = _compute_trust(nodes, W, sim_calc)

    # Stackelberg game
    game = StackelbergGame(
        num_agents=n, lambda_c=1.0, mu=0.1,
        eta=0.05, gamma=0.1, R=100.0, W=W,
    )
    bids = [
        AgentBid(
            agent_id=f"agent-{i}",
            model="deepseek-chat",
            alignment_willingness=trust_scores[i],
            revision_cost=15.0 * (1.1 - trust_scores[i]),
            max_revision_depth=trust_scores[i],
            historical_alignment=trust_scores[i],
        )
        for i in range(n)
    ]
    history = game.run(bids, num_rounds=15)
    theta = history[-1]["theta"] if history else np.ones(n) / n

    # Combined weight: θ × trust
    weights = theta * np.array(trust_scores)
    stances = [extract_stance(node["C"]) for node in nodes]
    vote_scores = {}
    for i, s in enumerate(stances):
        vote_scores[s] = vote_scores.get(s, 0.0) + weights[i]
    decision = max(vote_scores, key=vote_scores.get)

    return decision, {
        "method": "AEIC+SG",
        "trust": trust_scores,
        "theta": theta.tolist(),
        "weights": weights.tolist(),
        "convergence_rounds": len(history),
        "final_energy": history[-1]["energy"] if history else 0,
    }


def _compute_trust(nodes: List[Dict], W: np.ndarray,
                   sim_calc: SimilarityCalculator) -> List[float]:
    """Compute AEIC trust scores with topology awareness."""
    n = len(nodes)

    # IC consistency: does inference support conclusion?
    ic_scores = []
    for node in nodes:
        i_text = str(node.get("I", ""))
        c_text = str(node.get("C", ""))
        if i_text and c_text:
            ic_scores.append(sim_calc.compute(i_text, c_text))
        else:
            ic_scores.append(0.5)

    # EC relevance: does evidence support conclusion?
    ec_scores = []
    for node in nodes:
        e_text = " ".join(node["E"]) if isinstance(node.get("E"), list) else str(node.get("E", ""))
        c_text = str(node.get("C", ""))
        if e_text and c_text:
            ec_scores.append(sim_calc.compute(e_text, c_text))
        else:
            ec_scores.append(0.5)

    # Evidence consensus: topology-aware (only connected neighbors)
    ev_consensus = []
    for i in range(n):
        e_i = " ".join(nodes[i]["E"]) if isinstance(nodes[i].get("E"), list) else str(nodes[i].get("E", ""))
        neighbors = [j for j in range(n) if j != i and W[i, j] > 0]
        if neighbors and e_i:
            sims = []
            for j in neighbors:
                e_j = " ".join(nodes[j]["E"]) if isinstance(nodes[j].get("E"), list) else str(nodes[j].get("E", ""))
                if e_j:
                    sims.append(sim_calc.compute(e_i, e_j))
            ev_consensus.append(np.mean(sims) if sims else 0.5)
        else:
            ev_consensus.append(0.5)

    # Conclusion agreement with neighbors
    stances = [extract_stance(node["C"]) for node in nodes]
    agreement = []
    for i in range(n):
        neighbors = [j for j in range(n) if j != i and W[i, j] > 0]
        if neighbors:
            agree_count = sum(1 for j in neighbors if stances[j] == stances[i])
            agreement.append(agree_count / len(neighbors))
        else:
            agreement.append(0.5)

    # Composite trust: IC(0.30) + EC(0.25) + EvidCons(0.25) + Agreement(0.20)
    trust = []
    for i in range(n):
        t = (0.30 * ic_scores[i] + 0.25 * ec_scores[i] +
             0.25 * ev_consensus[i] + 0.20 * agreement[i])
        trust.append(float(np.clip(t, 0.1, 1.0)))

    return trust


# ═══════════════════════════════════════════════════════════════
# Phase 3: Adversarial Testing
# ═══════════════════════════════════════════════════════════════

def corrupt_agent(nodes: List[Dict], agent_idx: int, tier: int = 1) -> List[Dict]:
    """Corrupt one agent's output to test robustness.

    Tier 1: Only flip C (detectable via I→C mismatch)
    Tier 2: Flip both I and C (harder to detect)
    """
    corrupted = [dict(n) for n in nodes]
    original_c = corrupted[agent_idx]["C"]
    stance = extract_stance(original_c)
    flipped = "no" if stance == "yes" else "yes"

    if tier >= 1:
        corrupted[agent_idx]["C"] = f"{flipped.capitalize()}: 经过分析，答案是{flipped}。（高置信度）"

    if tier >= 2:
        corrupted[agent_idx]["I"] = f"综合证据分析，逻辑上推导出答案应该是{flipped}。证据明确支持这一结论。"

    return corrupted


# ═══════════════════════════════════════════════════════════════
# Phase 4: Main Experiment
# ═══════════════════════════════════════════════════════════════

def run(sim_method: str = "tfidf"):
    print("\n" + "=" * 70)
    print("  REAL Multi-Agent Consensus Experiment")
    print("  DeepSeek API • 30 Questions • 7 Agent Personas")
    print("  Agent Counts: [3, 5, 7] × Topologies: [complete, ring, star]")
    print("=" * 70)

    # Phase 1: Generate responses
    cache = generate_all_responses(QUESTIONS, n_agents=7)

    # Initialize similarity calculator
    sim_calc = SimilarityCalculator(method=sim_method)
    print(f"\n  Similarity method: {sim_method}")

    # Phase 2: Evaluate all configurations
    print("\n  Phase 2: Evaluating configurations...")
    results = {}  # {(n_agents, topo, method): [correct/incorrect per question]}
    trust_data = {}  # for analysis

    for n_agents in AGENT_COUNTS:
        for topo in TOPOLOGIES:
            W = make_topology(topo, n_agents)
            config_key = f"{n_agents}-{topo}"

            for method_name in METHOD_NAMES:
                results[(n_agents, topo, method_name)] = []

            for q_idx, q in enumerate(QUESTIONS):
                # Get agent responses for this question
                nodes = [cache[f"q{q_idx}_a{i}"] for i in range(n_agents)]

                # Method 1: Majority Vote
                decision, _ = majority_vote(nodes)
                results[(n_agents, topo, "MajVote")].append(
                    check_accuracy(decision, q["a"]))

                # Method 2: Quality-Weighted
                decision, _ = quality_weighted(nodes, sim_calc)
                results[(n_agents, topo, "QualWt")].append(
                    check_accuracy(decision, q["a"]))

                # Method 3: AEIC Trust (no Stackelberg)
                decision, _ = aeic_trust(nodes, sim_calc, W)
                results[(n_agents, topo, "AEIC")].append(
                    check_accuracy(decision, q["a"]))

                # Method 4: AEIC + Stackelberg (full method)
                decision, info = aeic_stackelberg(nodes, sim_calc, W)
                results[(n_agents, topo, "AEIC+SG")].append(
                    check_accuracy(decision, q["a"]))
                trust_data[(config_key, q_idx)] = info

    # Print results table
    print(f"\n  {'Config':<14s}", end="")
    for m in METHOD_NAMES:
        print(f" {m:>8s}", end="")
    print()
    print("  " + "-" * 52)

    for n_agents in AGENT_COUNTS:
        for topo in TOPOLOGIES:
            config = f"{n_agents}-{topo}"
            print(f"  {config:<14s}", end="")
            for method in METHOD_NAMES:
                acc = np.mean(results[(n_agents, topo, method)])
                print(f" {acc:>8.3f}", end="")
            print()

    # Phase 3: Adversarial testing
    print("\n  Phase 3: Adversarial robustness...")
    adv_results = {}  # {(n_agents, topo, method, tier): accuracy}

    for n_agents in AGENT_COUNTS:
        for topo in TOPOLOGIES:
            W = make_topology(topo, n_agents)

            for tier in [1, 2]:
                for method_name in METHOD_NAMES:
                    key = (n_agents, topo, method_name, tier)
                    adv_results[key] = []

                for q_idx, q in enumerate(QUESTIONS):
                    nodes = [cache[f"q{q_idx}_a{i}"] for i in range(n_agents)]
                    # Corrupt last agent
                    nodes_corrupt = corrupt_agent(nodes, n_agents - 1, tier=tier)

                    d1, _ = majority_vote(nodes_corrupt)
                    adv_results[(n_agents, topo, "MajVote", tier)].append(
                        check_accuracy(d1, q["a"]))

                    d2, _ = quality_weighted(nodes_corrupt, sim_calc)
                    adv_results[(n_agents, topo, "QualWt", tier)].append(
                        check_accuracy(d2, q["a"]))

                    d3, _ = aeic_trust(nodes_corrupt, sim_calc, W)
                    adv_results[(n_agents, topo, "AEIC", tier)].append(
                        check_accuracy(d3, q["a"]))

                    d4, info = aeic_stackelberg(nodes_corrupt, sim_calc, W)
                    adv_results[(n_agents, topo, "AEIC+SG", tier)].append(
                        check_accuracy(d4, q["a"]))

    # Print adversarial results
    print(f"\n  Adversarial (Tier 1: C flipped):")
    print(f"  {'Config':<14s}", end="")
    for m in METHOD_NAMES:
        print(f" {m:>8s}", end="")
    print()
    print("  " + "-" * 52)
    for n_agents in AGENT_COUNTS:
        for topo in TOPOLOGIES:
            config = f"{n_agents}-{topo}"
            print(f"  {config:<14s}", end="")
            for method in METHOD_NAMES:
                acc = np.mean(adv_results[(n_agents, topo, method, 1)])
                print(f" {acc:>8.3f}", end="")
            print()

    # Star topology: center vs leaf corruption
    print("\n  Star topology: center (agent 0) vs leaf corruption:")
    star_center_results = {}
    for n_agents in [5, 7]:
        W = make_topology("star", n_agents)
        for method_name in METHOD_NAMES:
            star_center_results[(n_agents, method_name)] = []

        for q_idx, q in enumerate(QUESTIONS):
            nodes = [cache[f"q{q_idx}_a{i}"] for i in range(n_agents)]
            nodes_corrupt = corrupt_agent(nodes, 0, tier=1)  # center corrupted

            d1, _ = majority_vote(nodes_corrupt)
            star_center_results[(n_agents, "MajVote")].append(check_accuracy(d1, q["a"]))
            d2, _ = quality_weighted(nodes_corrupt, sim_calc)
            star_center_results[(n_agents, "QualWt")].append(check_accuracy(d2, q["a"]))
            d3, _ = aeic_trust(nodes_corrupt, sim_calc, W)
            star_center_results[(n_agents, "AEIC")].append(check_accuracy(d3, q["a"]))
            d4, _ = aeic_stackelberg(nodes_corrupt, sim_calc, W)
            star_center_results[(n_agents, "AEIC+SG")].append(check_accuracy(d4, q["a"]))

    for n_agents in [5, 7]:
        print(f"    {n_agents}-star (center attack): ", end="")
        for method in METHOD_NAMES:
            acc = np.mean(star_center_results[(n_agents, method)])
            print(f"{method}={acc:.3f}  ", end="")
        print()

    # Phase 4: Generate figures
    print("\n  Phase 4: Generating figures...")
    _plot_results(results, adv_results, trust_data, star_center_results)

    # Statistical tests
    print("\n  Statistical tests (AEIC+SG vs baselines, paired t-test):")
    for baseline in ["MajVote", "QualWt", "AEIC"]:
        our_accs = []
        bl_accs = []
        for n_agents in AGENT_COUNTS:
            for topo in TOPOLOGIES:
                our_accs.extend([1.0 if x else 0.0 for x in results[(n_agents, topo, "AEIC+SG")]])
                bl_accs.extend([1.0 if x else 0.0 for x in results[(n_agents, topo, baseline)]])
        t, p = stats.ttest_rel(our_accs, bl_accs)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    AEIC+SG vs {baseline:<8s}: t={t:+.3f}, p={p:.4f} {sig}")

    print("\n  → figures/exp_real.{pdf,png}")
    return results, adv_results


def _plot_results(results, adv_results, trust_data, star_center_results):
    """Generate IEEE 2×2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    method_colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]

    # (a) Accuracy by agent count × topology (our method)
    ax = axes[0, 0]
    x = np.arange(len(TOPOLOGIES))
    bar_width = 0.22
    for i, n_agents in enumerate(AGENT_COUNTS):
        accs = [np.mean(results[(n_agents, topo, "AEIC+SG")]) for topo in TOPOLOGIES]
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, accs, bar_width,
                      label=f"N={n_agents}", color=PALETTE[i],
                      edgecolor="black", linewidth=0.4)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{acc:.2f}", ha="center", fontsize=6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Complete", "Ring", "Star"], fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("AEIC+Stackelberg: Topology × Agent Count")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "a")

    # (b) Method ablation (best config = 7-complete)
    ax = axes[0, 1]
    x = np.arange(len(METHOD_NAMES))
    # Show all topologies for N=7
    bar_width = 0.25
    for t_idx, topo in enumerate(TOPOLOGIES):
        accs = [np.mean(results[(7, topo, m)]) for m in METHOD_NAMES]
        offset = (t_idx - 1) * bar_width
        ax.bar(x + offset, accs, bar_width,
               label=topo.capitalize(), color=PALETTE[t_idx],
               edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Method Comparison (N=7)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "b")

    # (c) Adversarial robustness: normal vs Tier1 vs Tier2
    ax = axes[1, 0]
    # Compare for 7-complete across methods
    x = np.arange(len(METHOD_NAMES))
    normal_accs = [np.mean(results[(7, "complete", m)]) for m in METHOD_NAMES]
    tier1_accs = [np.mean(adv_results[(7, "complete", m, 1)]) for m in METHOD_NAMES]
    tier2_accs = [np.mean(adv_results[(7, "complete", m, 2)]) for m in METHOD_NAMES]

    bar_width = 0.25
    ax.bar(x - bar_width, normal_accs, bar_width, label="Normal",
           color=PALETTE[0], edgecolor="black", linewidth=0.4)
    ax.bar(x, tier1_accs, bar_width, label="Tier1 (C flipped)",
           color=PALETTE[1], edgecolor="black", linewidth=0.4)
    ax.bar(x + bar_width, tier2_accs, bar_width, label="Tier2 (I+C flipped)",
           color=PALETTE[2], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Adversarial Robustness (7-Complete)")
    ax.legend(fontsize=6, loc="lower left")
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "c")

    # (d) Star topology vulnerability: center vs leaf attack
    ax = axes[1, 1]
    x = np.arange(len(METHOD_NAMES))
    # Normal 7-star
    normal = [np.mean(results[(7, "star", m)]) for m in METHOD_NAMES]
    # Leaf attack (last agent)
    leaf_attack = [np.mean(adv_results[(7, "star", m, 1)]) for m in METHOD_NAMES]
    # Center attack
    center_attack = [np.mean(star_center_results[(7, m)]) for m in METHOD_NAMES]

    bar_width = 0.25
    ax.bar(x - bar_width, normal, bar_width, label="Normal",
           color=PALETTE[0], edgecolor="black", linewidth=0.4)
    ax.bar(x, leaf_attack, bar_width, label="Leaf attacked",
           color=PALETTE[1], edgecolor="black", linewidth=0.4)
    ax.bar(x + bar_width, center_attack, bar_width, label="Center attacked",
           color=PALETTE[3], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Star Topology Vulnerability (N=7)")
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "d")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/exp_real.pdf")
    plt.savefig("figures/exp_real.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="tfidf",
                        choices=["embedding", "bm25", "tfidf"])
    args = parser.parse_args()
    run(sim_method=args.sim)
