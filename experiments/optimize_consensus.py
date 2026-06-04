"""
共识参数优化器 — 在缓存信号上学习最优bid构建参数

读取precompute后的signals缓存，用scipy全局优化搜索最优ConsensusConfig。
零API开销，纯数学运算。

Usage:
  python experiments/optimize_consensus.py \
      --signals results/signals_StrategyQA_3a.json

  # 在adaptive缓存上微调
  python experiments/optimize_consensus.py \
      --signals results/signals_adaptive_7a.json \
      --init results/optimized_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus_config import ConsensusConfig
from core.stackelberg import AgentBid, StackelbergGame
from core.trust import extract_stance


def consensus_from_signals(signals: Dict, config: ConsensusConfig,
                           n_agents: int) -> str:
    """从预计算信号 + config参数 → Stackelberg → 决策。"""
    ic = signals["ic"]
    ec = signals["ec"]
    eq = signals["eq"]
    ev = signals["ev"]
    stances = signals["stances"]
    agreement = signals["agreement"]
    n = len(ic)

    if n < 2:
        return stances[0] if stances else "neutral"

    W_game = np.ones((n, n)) - np.eye(n)
    W_game /= max(n - 1, 1)

    bids = []
    for i in range(n):
        ic_i = float(np.clip(ic[i], 0, 1))
        ec_i = float(np.clip(ec[i], 0, 1))
        eq_i = float(np.clip(eq[i], 0, 1))
        ev_i = float(np.clip(ev[i], 0, 1))
        ag_i = float(np.clip(agreement[i], 0, 1))
        aw = ic_i**config.aw_ic_exp * eq_i**config.aw_eq_exp * \
             (config.aw_ev_base + config.aw_ev_scale * ev_i)
        ha = config.ha_ic*ic_i + config.ha_ec*ec_i + \
             config.ha_eq*eq_i + config.ha_ev*ev_i
        bids.append(AgentBid(
            agent_id=f"agent-{i}",
            model="opt",
            alignment_willingness=max(config.aw_floor, aw),
            revision_cost=max(config.rc_floor,
                              config.rc_scale * (1.0 - ag_i)),
            max_revision_depth=float(np.clip(ec_i, config.mrd_lo,
                                             config.mrd_hi)),
            historical_alignment=ha,
        ))

    game = StackelbergGame(num_agents=n, W=W_game,
                           lambda_c=config.sg_lambda, mu=config.sg_mu,
                           eta=config.sg_eta, gamma=config.sg_gamma)
    history = game.run(bids)
    theta = np.array(history[-1]["theta"])

    if theta.max() > theta.min():
        theta_norm = (theta - theta.min()) / (theta.max() - theta.min())
    else:
        theta_norm = np.ones(n) * 0.5

    vote_scores = {}
    for i in range(n):
        if stances[i] != "neutral":
            vote_scores[stances[i]] = vote_scores.get(stances[i], 0.0) + \
                                      theta_norm[i]

    if not vote_scores:
        return "neutral"
    return max(vote_scores, key=vote_scores.get)


def evaluate_config(questions: List[Dict], config: ConsensusConfig,
                    n_agents: int) -> Dict:
    """用给定config在缓存数据上计算准确率。"""
    correct = 0
    total = 0
    for q in questions:
        if "signals" not in q:
            continue
        decision = consensus_from_signals(q["signals"], config, n_agents)
        if decision == q["ground_truth"]:
            correct += 1
        total += 1
    acc = correct / max(total, 1)
    return {"accuracy": acc, "correct": correct, "total": total}


def objective(params_vec, questions, n_agents):
    """scipy优化目标: 负准确率。"""
    config = ConsensusConfig.from_vector(params_vec)
    result = evaluate_config(questions, config, n_agents)
    return -result["accuracy"]


def optimize(signals_paths, init_config_path: str = None,
             out_path: str = None,
             method: str = "differential_evolution"):
    """全局优化共识参数。支持多signals文件合并训练。"""
    from scipy.optimize import differential_evolution

    if isinstance(signals_paths, str):
        signals_paths = [signals_paths]

    all_questions = []
    n_agents = None
    train_datasets = []
    for sp in signals_paths:
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_questions.extend(data["questions"])
        n_agents = data["meta"]["n_agents"]
        train_datasets.append(data["meta"]["dataset"])

    questions = all_questions

    # Baseline: default config
    default_cfg = ConsensusConfig()
    default_result = evaluate_config(questions, default_cfg, n_agents)
    print(f"Default config:  {default_result['accuracy']:.1%} "
          f"({default_result['correct']}/{default_result['total']})")

    # Init from file if provided (for fine-tuning)
    if init_config_path:
        init_cfg = ConsensusConfig.load(init_config_path)
        init_result = evaluate_config(questions, init_cfg, n_agents)
        print(f"Init config:     {init_result['accuracy']:.1%} "
              f"({init_result['correct']}/{init_result['total']})")
        x0 = init_cfg.to_vector()
    else:
        x0 = default_cfg.to_vector()

    bounds = ConsensusConfig.bounds()

    print(f"\nOptimizing {len(ConsensusConfig.param_names())} parameters "
          f"on {len(questions)} questions...")

    result = differential_evolution(
        objective, bounds,
        args=(questions, n_agents),
        seed=42,
        maxiter=100,
        popsize=10,
        tol=1e-4,
        polish=True,
        init="latinhypercube",
        workers=1,
        disp=True,
    )

    best_cfg = ConsensusConfig.from_vector(result.x)
    best_result = evaluate_config(questions, best_cfg, n_agents)

    print(f"\nOptimized config: {best_result['accuracy']:.1%} "
          f"({best_result['correct']}/{best_result['total']})")
    delta = best_result["accuracy"] - default_result["accuracy"]
    print(f"Improvement:      {delta:+.1%}")

    # Print parameter comparison
    print(f"\n{'Parameter':<16} {'Default':>10} {'Optimized':>10} {'Change':>10}")
    print(f"{'-'*46}")
    for name, d_val, o_val in zip(
        ConsensusConfig.param_names(),
        default_cfg.to_vector(),
        best_cfg.to_vector()
    ):
        ch = o_val - d_val
        print(f"{name:<16} {d_val:>10.4f} {o_val:>10.4f} {ch:>+10.4f}")

    # Save
    if out_path is None:
        out_dir = os.path.dirname(signals_paths[0]) or "results"
        out_path = os.path.join(out_dir, "optimized_config.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    best_cfg.save(out_path)
    print(f"\nSaved to {out_path}")

    return best_cfg, {
        "train_datasets": train_datasets,
        "train_accuracy": best_result["accuracy"],
        "default_accuracy": default_result["accuracy"],
        "improvement": best_result["accuracy"] - default_result["accuracy"],
        "config_path": out_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize consensus parameters on cached signals")
    parser.add_argument("--signals", required=True, nargs="+",
                        help="Path(s) to signals JSON from collect_cache.py")
    parser.add_argument("--init", default=None,
                        help="Init config JSON for fine-tuning")
    parser.add_argument("--out", default=None,
                        help="Output config path (default: results/optimized_config.json)")
    args = parser.parse_args()

    optimize(args.signals, args.init, args.out)


if __name__ == "__main__":
    main()
