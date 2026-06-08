"""
共识参数优化器 — 在缓存信号上学习最优bid构建参数

读取precompute后的signals缓存，用scipy全局优化搜索最优ConsensusConfig。
零API开销，纯数学运算。

Usage:
  python experiments/optimize_consensus.py \
      --signals results/signals/StrategyQA.json

  # 多数据集联合训练
  python experiments/optimize_consensus.py \
      --signals results/signals/MMLU-Hard.json results/signals/StrategyQA.json
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


def compute_weights(signals: Dict, config: ConsensusConfig,
                    n_agents: int) -> np.ndarray:
    """从信号 + config → Stackelberg → 归一化权重向量。"""
    ic = signals["ic"]
    ec = signals["ec"]
    eq = signals["eq"]
    ev = signals["ev"]
    agreement = signals["agreement"]
    n = len(ic)

    if n < 2:
        return np.ones(1) * 0.5

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
        return (theta - theta.min()) / (theta.max() - theta.min())
    return np.ones(n) * 0.5


def consensus_from_signals(signals: Dict, config: ConsensusConfig,
                           n_agents: int) -> str:
    """从预计算信号 + config参数 → Stackelberg → 决策。"""
    stances = signals["stances"]
    if len(signals["ic"]) < 2:
        return stances[0] if stances else "neutral"

    weights = compute_weights(signals, config, n_agents)

    vote_scores = {}
    for i in range(len(stances)):
        if stances[i] != "neutral":
            vote_scores[stances[i]] = vote_scores.get(stances[i], 0.0) + \
                                      weights[i]

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


def compute_ic_metric(questions: List[Dict], config: ConsensusConfig,
                      n_agents: int) -> float:
    """IC(θ): 高质量agent组权重 > 低质量agent组的比例。

    对每道题，按推理质量(IC·EC·EQ均值，不含EV)将agent分为
    top-2组(H)和bottom-2组(L)，计算H平均权重>L平均权重的比例。
    """
    ic_count = 0
    total = 0
    for q in questions:
        if "signals" not in q:
            continue
        s = q["signals"]
        n = len(s["ic"])
        if n < 4:
            continue

        quality = [(s["ic"][i] + s["ec"][i] + s["eq"][i]) / 3.0
                   for i in range(n)]
        weights = compute_weights(s, config, n_agents)

        ranked = sorted(range(n), key=lambda i: quality[i], reverse=True)
        h = ranked[:2]
        l = ranked[-2:]
        avg_h = sum(weights[i] for i in h) / 2.0
        avg_l = sum(weights[j] for j in l) / 2.0
        if avg_h > avg_l:
            ic_count += 1
        total += 1
    return ic_count / max(total, 1)


def objective(params_vec, questions, n_agents, lambda_ic=0.0):
    """scipy优化目标: -(accuracy + λ·IC)。"""
    config = ConsensusConfig.from_vector(params_vec)
    result = evaluate_config(questions, config, n_agents)
    if lambda_ic > 0:
        ic = compute_ic_metric(questions, config, n_agents)
        return -(result["accuracy"] + lambda_ic * ic)
    return -result["accuracy"]


def optimize(signals_paths, init_config_path: str = None,
             out_path: str = None, lambda_ic: float = 0.0):
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

    default_cfg = ConsensusConfig()
    default_result = evaluate_config(questions, default_cfg, n_agents)
    default_ic = compute_ic_metric(questions, default_cfg, n_agents)
    print(f"Default config:  acc={default_result['accuracy']:.1%}  "
          f"IC={default_ic:.1%}  ({default_result['correct']}/{default_result['total']})")

    if init_config_path:
        init_cfg = ConsensusConfig.load(init_config_path)
        init_result = evaluate_config(questions, init_cfg, n_agents)
        init_ic = compute_ic_metric(questions, init_cfg, n_agents)
        print(f"Init config:     acc={init_result['accuracy']:.1%}  IC={init_ic:.1%}")

    bounds = ConsensusConfig.bounds()

    obj_desc = f"accuracy + {lambda_ic}·IC" if lambda_ic > 0 else "accuracy"
    print(f"\nOptimizing {len(ConsensusConfig.param_names())} parameters "
          f"on {len(questions)} questions (objective: {obj_desc})...")

    result = differential_evolution(
        objective, bounds,
        args=(questions, n_agents, lambda_ic),
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
    best_ic = compute_ic_metric(questions, best_cfg, n_agents)

    print(f"\nOptimized config: acc={best_result['accuracy']:.1%}  IC={best_ic:.1%}  "
          f"({best_result['correct']}/{best_result['total']})")
    delta = best_result["accuracy"] - default_result["accuracy"]
    print(f"Acc improvement:  {delta:+.1%}")
    print(f"IC  improvement:  {best_ic - default_ic:+.1%}")

    print(f"\n{'Parameter':<16} {'Default':>10} {'Optimized':>10} {'Change':>10}")
    print(f"{'-'*46}")
    for name, d_val, o_val in zip(
        ConsensusConfig.param_names(),
        default_cfg.to_vector(),
        best_cfg.to_vector()
    ):
        ch = o_val - d_val
        print(f"{name:<16} {d_val:>10.4f} {o_val:>10.4f} {ch:>+10.4f}")

    if out_path is None:
        out_path = os.path.join("results", "configs", "optimized_config.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    best_cfg.save(out_path)
    print(f"\nSaved to {out_path}")

    return best_cfg, {
        "train_datasets": train_datasets,
        "train_accuracy": best_result["accuracy"],
        "train_ic": best_ic,
        "default_accuracy": default_result["accuracy"],
        "default_ic": default_ic,
        "lambda_ic": lambda_ic,
        "config_path": out_path,
    }


def optimize_cv(signals_paths, out_path: str = None,
                lambda_ic: float = 0.0, n_folds: int = 5, seed: int = 42):
    """Cross-validated θ* optimization. Reports held-out test accuracy."""
    from sklearn.model_selection import StratifiedKFold
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
    labels = [q["ground_truth"] for q in questions]

    default_cfg = ConsensusConfig()
    default_result = evaluate_config(questions, default_cfg, n_agents)
    print(f"Default config (all data): acc={default_result['accuracy']:.1%}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    bounds = ConsensusConfig.bounds()

    fold_results = []
    fold_configs = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(questions, labels)):
        train_q = [questions[i] for i in train_idx]
        test_q = [questions[i] for i in test_idx]

        print(f"\n{'='*50}")
        print(f"Fold {fold_i+1}/{n_folds}: train={len(train_q)}, test={len(test_q)}")

        result = differential_evolution(
            objective, bounds,
            args=(train_q, n_agents, lambda_ic),
            seed=seed + fold_i,
            maxiter=50,
            popsize=8,
            tol=1e-3,
            polish=False,
            init="latinhypercube",
            workers=1,
            disp=False,
        )

        fold_cfg = ConsensusConfig.from_vector(result.x)
        fold_configs.append(fold_cfg)

        train_res = evaluate_config(train_q, fold_cfg, n_agents)
        test_res = evaluate_config(test_q, fold_cfg, n_agents)
        train_ic = compute_ic_metric(train_q, fold_cfg, n_agents)
        test_ic = compute_ic_metric(test_q, fold_cfg, n_agents)

        print(f"  Train: acc={train_res['accuracy']:.1%}  IC={train_ic:.1%}")
        print(f"  Test:  acc={test_res['accuracy']:.1%}  IC={test_ic:.1%}")

        fold_results.append({
            "fold": fold_i + 1,
            "train_acc": train_res["accuracy"],
            "test_acc": test_res["accuracy"],
            "train_ic": train_ic,
            "test_ic": test_ic,
            "train_size": len(train_q),
            "test_size": len(test_q),
        })

    avg_train = np.mean([r["train_acc"] for r in fold_results])
    avg_test = np.mean([r["test_acc"] for r in fold_results])
    std_test = np.std([r["test_acc"] for r in fold_results])
    avg_train_ic = np.mean([r["train_ic"] for r in fold_results])
    avg_test_ic = np.mean([r["test_ic"] for r in fold_results])

    print(f"\n{'='*50}")
    print(f"CV Summary ({n_folds}-fold):")
    print(f"  Train acc: {avg_train:.1%}")
    print(f"  Test  acc: {avg_test:.1%} ± {std_test:.1%}")
    print(f"  Train IC:  {avg_train_ic:.1%}")
    print(f"  Test  IC:  {avg_test_ic:.1%}")
    print(f"  Overfit gap: {avg_train - avg_test:+.1%}")

    print(f"\nTraining final model on all {len(questions)} questions...")
    result_full = differential_evolution(
        objective, bounds,
        args=(questions, n_agents, lambda_ic),
        seed=seed,
        maxiter=60,
        popsize=8,
        tol=1e-3,
        polish=False,
        init="latinhypercube",
        workers=1,
        disp=False,
    )
    final_cfg = ConsensusConfig.from_vector(result_full.x)
    final_result = evaluate_config(questions, final_cfg, n_agents)

    if out_path is None:
        out_path = os.path.join("results", "configs", "optimized_config.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    final_cfg.save(out_path)

    summary = {
        "train_datasets": train_datasets,
        "n_folds": n_folds,
        "cv_train_acc": avg_train,
        "cv_test_acc": avg_test,
        "cv_test_std": std_test,
        "cv_train_ic": avg_train_ic,
        "cv_test_ic": avg_test_ic,
        "overfit_gap": avg_train - avg_test,
        "full_train_acc": final_result["accuracy"],
        "default_acc": default_result["accuracy"],
        "lambda_ic": lambda_ic,
        "fold_results": fold_results,
        "config_path": out_path,
    }

    summary_path = out_path.replace(".json", "_cv_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved config to {out_path}")
    print(f"Saved CV summary to {summary_path}")

    return final_cfg, summary


def main():
    parser = argparse.ArgumentParser(
        description="Optimize consensus parameters on cached signals")
    parser.add_argument("--signals", required=True, nargs="+",
                        help="Path(s) to signals JSON from collect_cache.py")
    parser.add_argument("--init", default=None,
                        help="Init config JSON for fine-tuning")
    parser.add_argument("--out", default=None,
                        help="Output config path (default: results/configs/optimized_config.json)")
    parser.add_argument("--lambda_ic", type=float, default=0.0,
                        help="IC incentive weight (0=pure accuracy, >0=accuracy+λ·IC)")
    parser.add_argument("--cv", action="store_true",
                        help="Use cross-validated optimization (reports held-out accuracy)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    if args.cv:
        optimize_cv(args.signals, args.out, args.lambda_ic, args.n_folds)
    else:
        optimize(args.signals, args.init, args.out, args.lambda_ic)


if __name__ == "__main__":
    main()
