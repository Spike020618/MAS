"""
start.py - 博弈驱动的去中心化语义机制 · 实验入口
================================================================

论文实验框架（三大实验）：
  Exp-1  收敛性验证    验证共识能量 E 随迭代单调递减
  Exp-2  语义方法对比  在多节点数据集上对比各相似度方法
  Exp-3  基线对比      SCG vs Random / Pure-Stackelberg / Pure-Consensus

数据来源：
  DeepSeek-V3 动态生成多节点 AEIC 共识数据（首次运行调 API，后续读缓存）
  缓存位置：results/generated_dataset.json

用法:
  python start.py                  # 运行全部实验（有缓存直接用）
  python start.py --exp 1          # 仅收敛性实验
  python start.py --exp 2          # 仅语义方法对比
  python start.py --exp 3          # 仅基线对比
  python start.py --regen          # 强制重新生成数据后运行
  python start.py --exp all --plot # 全部 + 生成图表
  python start.py --nodes 4        # 每轮使用 4 个节点（默认3）

输出:
  results/generated_dataset.json
  results/exp1_convergence.csv
  results/exp2_semantic_comparison.csv
  results/exp3_baseline_comparison.csv
  results/figures/  (若 --plot)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── 项目路径 ──────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
warnings.filterwarnings("ignore")

# ── 输出目录 ──────────────────────────────────────────
RESULTS_DIR = os.path.join(project_root, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CACHE_PATH  = os.path.join(RESULTS_DIR, "generated_dataset.json")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# 懒加载核心模块
# ─────────────────────────────────────────────────────

def _import_core():
    from mas.consensus.stackelberg import StackelbergConsensusGame, AgentBid
    from mas.consensus.consensus   import ConsensusEngine, run_consensus_simulation
    from mas.data                  import load_or_generate, GeneratedDataset
    try:
        import mas.config as _cfg
    except ImportError:
        _cfg = None
    return StackelbergConsensusGame, AgentBid, ConsensusEngine, run_consensus_simulation, load_or_generate, _cfg


# ─────────────────────────────────────────────────────
# 数据集加载（全局单例）
# ─────────────────────────────────────────────────────

_DATASET_CACHE = None

def _load_dataset(force_regenerate: bool = False, n_nodes: int = 3):
    global _DATASET_CACHE
    if _DATASET_CACHE is not None and not force_regenerate:
        return _DATASET_CACHE
    _, _, _, _, load_or_generate, _ = _import_core()
    _DATASET_CACHE = load_or_generate(
        cache_path=CACHE_PATH,
        force_regenerate=force_regenerate,
        n_nodes=n_nodes,
    )
    return _DATASET_CACHE


# ═════════════════════════════════════════════════════
# Experiment 1：收敛性验证
# ═════════════════════════════════════════════════════

def exp1_convergence(
    num_rounds: int = 15,
    num_agents: int = 5,
    lambda_values: Optional[List[float]] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    验证公式3的收敛性：
        θ_i^{t+1} = θ_i^t + η∇U_i - γΣ_j w_ij(θ_i - θ_j)

    对多个 λ 运行 Stackelberg 博弈，记录
    consensus_energy(E)、param_change、leader_utility。
    """
    print("\n" + "=" * 70)
    print("🔬 Experiment 1: 收敛性验证 (Convergence Analysis)")
    print("=" * 70)

    StackelbergConsensusGame, AgentBid, *_ = _import_core()

    if lambda_values is None:
        lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    bids = [
        AgentBid(f"node_{i}", 8002 + i, "deepseek", "solver",
                 quality_promise=0.7 + 0.05 * i,
                 cost_per_task=15 + 3 * i,
                 capacity=0.5 + 0.1 * i,
                 past_performance=0.8)
        for i in range(num_agents)
    ]

    all_rows: List[Dict] = []

    for lam in lambda_values:
        print(f"\n  ▶ λ = {lam}")
        game   = StackelbergConsensusGame(leader_port=8001, num_agents=num_agents, lambda_c=lam)
        result = game.run_game_rounds(bids, total_workload=1.0, num_rounds=num_rounds)

        for r in result["results"]:
            all_rows.append({
                "lambda":           lam,
                "round":            r["round"],
                "consensus_energy": r["consensus_energy"],
                "param_change":     r["param_change"],
                "leader_utility":   r["leader_utility"],
            })

        energies = [r["consensus_energy"] for r in result["results"]]
        is_mono  = all(energies[i] >= energies[i+1] for i in range(len(energies)-1))
        conv_r   = next((r["round"] for r in result["results"] if r["param_change"] < 1e-4), num_rounds)
        print(f"     E_final={energies[-1]:.6f}  "
              f"{'✅ 单调递减' if is_mono else '⚠️  非严格单调'}  "
              f"收敛轮次≈{conv_r}")

    df = pd.DataFrame(all_rows)
    if save:
        path = os.path.join(RESULTS_DIR, "exp1_convergence.csv")
        df.to_csv(path, index=False)
        print(f"\n  💾 {path}")

    summary = (
        df.groupby("lambda")
        .agg(final_energy=("consensus_energy", "last"),
             final_change=("param_change", "last"),
             avg_utility=("leader_utility", "mean"))
        .reset_index()
    )
    print("\n  📊 λ 参数影响汇总:")
    print(summary.to_string(index=False))
    return df


# ═════════════════════════════════════════════════════
# Experiment 2：语义相似度方法对比
# ═════════════════════════════════════════════════════

def exp2_semantic_comparison(
    methods: Optional[List[str]] = None,
    force_regenerate: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    在 DeepSeek 生成的多节点 AEIC 数据集上对比各相似度方法。

    每轮：N 个节点各自提交 AEIC 记录，引擎计算全量两两相似度，
    取平均值与 GT 对比。

    指标：
      MAE   = |pred_avg_sim - gt_sim|
      Acc   = 三分类（high/medium/low）准确率
      ESS率 = ESS_Consensus 决策比例
      AvgU  = 平均博弈效用
    """
    print("\n" + "=" * 70)
    print("🔬 Experiment 2: 语义相似度方法对比（多节点共识）")
    print("=" * 70)

    _, _, ConsensusEngine, _, _, _cfg = _import_core()

    # ── 加载数据集 ──────────────────────────────────
    ds             = _load_dataset(force_regenerate=force_regenerate)
    node_recs_list = ds.get_node_records()   # List[List[Dict]]
    gts            = ds.ground_truths()
    gt_labs        = ds.labels()

    print(f"\n  数据集: {ds.summary()}")

    # ── 确定对比方法列表 ────────────────────────────
    if methods is None:
        methods = ["char_jaccard", "word_tfidf", "bm25"]
        if _cfg and getattr(_cfg, "API_KEY", None):
            methods.append("sentence_bert")

    all_rows: List[Dict] = []

    for method in methods:
        t0 = time.time()
        print(f"\n  ▶ 方法: {method}")

        engine_kwargs = dict(similarity_method=method)
        if method == "sentence_bert" and _cfg:
            engine_kwargs.update(
                use_sentence_bert=True,
                api_key=getattr(_cfg, "API_KEY", None),
                api_base=getattr(_cfg, "API_BASE", None),
                api_model=getattr(_cfg, "API_MODEL", "text-embedding-v4"),
            )
        engine = ConsensusEngine(**engine_kwargs)

        for i, node_recs in enumerate(node_recs_list):
            try:
                res        = engine.evaluate_consensus(node_recs)
                pred_sim   = res["avg_similarity"]
                pred_label = ("high"   if pred_sim >= 0.75 else
                              "medium" if pred_sim >= 0.40 else "low")
                all_rows.append({
                    "method":          method,
                    "round_id":        ds.all_rounds[i].id,
                    "domain":          ds.all_rounds[i].domain,
                    "scenario":        ds.all_rounds[i].scenario,
                    "n_nodes":         res["n_nodes"],
                    "gt_label":        gt_labs[i],
                    "gt_similarity":   gts[i],
                    "pred_similarity": pred_sim,
                    "pred_label":      pred_label,
                    "abs_error":       abs(pred_sim - gts[i]),
                    "label_correct":   int(pred_label == gt_labs[i]),
                    "utility":         res["utility"],
                    "decision":        res["decision"],
                })
            except Exception as e:
                print(f"     Round {i} 失败: {e}")

        elapsed = time.time() - t0
        print(f"     完成 ({elapsed:.1f}s)")

    df = pd.DataFrame(all_rows)
    if save:
        path = os.path.join(RESULTS_DIR, "exp2_semantic_comparison.csv")
        df.to_csv(path, index=False)
        print(f"\n  💾 {path}")

    print("\n  📊 方法对比汇总:")
    print(f"  {'方法':<18} {'MAE':>8} {'Acc':>8} {'ESS率':>8} {'AvgU':>8}")
    print("  " + "-" * 56)
    for method, grp in df.groupby("method"):
        mae   = grp["abs_error"].mean()
        acc   = grp["label_correct"].mean()
        ess_r = (grp["decision"] == "ESS_Consensus").mean()
        avg_u = grp["utility"].mean()
        print(f"  {method:<18} {mae:>8.4f} {acc:>8.2%} {ess_r:>8.2%} {avg_u:>8.2f}")

    if df["domain"].nunique() > 1:
        print("\n  📊 领域分解（MAE）:")
        pivot = df.pivot_table(values="abs_error", index="domain",
                               columns="method", aggfunc="mean")
        print(pivot.round(4).to_string())

    return df


# ═════════════════════════════════════════════════════
# Experiment 3：与基线对比
# ═════════════════════════════════════════════════════

def exp3_baseline_comparison(
    num_trials: int = 20,
    num_agents: int = 5,
    num_rounds: int = 10,
    save: bool = True,
) -> pd.DataFrame:
    """
    对比四种任务分配方案的社会福利（Social Welfare）：
      1. Random             随机分配
      2. Pure-Stackelberg   无共识约束（λ=0）
      3. Pure-Consensus     无博弈激励（容量比例均分）
      4. SCG (Ours)         Stackelberg Consensus Game（λ=1）

    每次试验随机生成 num_agents 个节点（DistributedAgent），
    各节点有独立的 quality_promise / cost / capacity。
    """
    print("\n" + "=" * 70)
    print("🔬 Experiment 3: 基线对比 (Baseline Comparison)")
    print("=" * 70)

    StackelbergConsensusGame, AgentBid, *_ = _import_core()
    np.random.seed(42)
    all_rows: List[Dict] = []

    for trial in range(num_trials):
        # 随机生成 N 个节点的投标信息
        bids = [
            AgentBid(
                agent_id=f"node_{i}", port=8002 + i,
                model=np.random.choice(["deepseek", "qwen", "gpt4"]),
                role=np.random.choice(["solver", "reviewer"]),
                quality_promise=np.random.uniform(0.6, 0.95),
                cost_per_task=np.random.uniform(10, 35),
                capacity=np.random.uniform(0.3, 0.8),
                past_performance=np.random.uniform(0.7, 1.0),
            )
            for i in range(num_agents)
        ]

        # (1) Random
        rw = _random_allocation(bids)

        # (2) Pure Stackelberg（λ=0，无共识约束）
        g_pure = StackelbergConsensusGame(leader_port=8001, num_agents=num_agents, lambda_c=0.0)
        pr     = g_pure.execute_stackelberg_game(bids, total_workload=1.0)
        pw     = _social_welfare(bids, pr["allocation"])

        # (3) Pure Consensus（容量比例均分，无博弈激励）
        cw = _pure_consensus_allocation(bids)

        # (4) SCG - Ours（λ=1，共识+博弈）
        g_scg  = StackelbergConsensusGame(leader_port=8001, num_agents=num_agents, lambda_c=1.0)
        scg_r  = g_scg.run_game_rounds(bids, total_workload=1.0, num_rounds=num_rounds)
        scg_final = scg_r["results"][-1]
        scg_w  = _social_welfare(bids, g_scg.optimize_allocation(bids, 1.0))
        conv_r = next(
            (r["round"] for r in scg_r["results"] if r["param_change"] < 1e-4),
            num_rounds
        )

        for method, sw, lu, cr in [
            ("Random",           rw,    rw * 0.8,                   float("nan")),
            ("Pure-Stackelberg", pw,    pr["leader_utility"],        float("nan")),
            ("Pure-Consensus",   cw,    cw * 0.9,                   float("nan")),
            ("SCG (Ours)",       scg_w, scg_final["leader_utility"], conv_r),
        ]:
            all_rows.append({
                "trial": trial + 1, "method": method,
                "social_welfare": sw, "leader_utility": lu,
                "convergence_rounds": cr,
            })

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{num_trials} 完成")

    df = pd.DataFrame(all_rows)
    if save:
        path = os.path.join(RESULTS_DIR, "exp3_baseline_comparison.csv")
        df.to_csv(path, index=False)
        print(f"\n  💾 {path}")

    print("\n  📊 基线对比汇总（均值 ± 标准差）:")
    print(f"  {'方法':<20} {'社会福利':>16} {'Leader效用':>16} {'收敛轮次':>10}")
    print("  " + "-" * 66)
    for method, grp in df.groupby("method", sort=False):
        sw   = f"{grp['social_welfare'].mean():+.2f}±{grp['social_welfare'].std():.2f}"
        lu   = f"{grp['leader_utility'].mean():+.2f}±{grp['leader_utility'].std():.2f}"
        cr_v = grp["convergence_rounds"].dropna()
        cr   = f"{cr_v.mean():.1f}" if len(cr_v) > 0 else "—"
        print(f"  {method:<20} {sw:>16} {lu:>16} {cr:>10}")

    try:
        from scipy import stats
        s1 = df[df["method"] == "SCG (Ours)"]["social_welfare"]
        s2 = df[df["method"] == "Pure-Stackelberg"]["social_welfare"]
        if len(s1) > 1:
            t, p = stats.ttest_ind(s1, s2)
            sig  = "✅ 显著" if p < 0.05 else "❌ 不显著"
            print(f"\n  SCG vs Pure-Stackelberg  t={t:.3f}  p={p:.4f}  {sig}")
    except ImportError:
        pass

    return df


# ─────────────────────────────────────────────────────
# 基线辅助函数
# ─────────────────────────────────────────────────────

def _random_allocation(bids) -> float:
    if not bids: return 0.0
    share = 1.0 / len(bids)
    return sum(b.quality_promise * b.past_performance * share * 100
               - b.cost_per_task * share for b in bids)


def _pure_consensus_allocation(bids) -> float:
    if not bids: return 0.0
    total_cap = sum(b.capacity for b in bids) or 1.0
    return sum(
        b.quality_promise * b.past_performance * (b.capacity / total_cap) * 100
        - b.cost_per_task * (b.capacity / total_cap)
        for b in bids
    )


def _social_welfare(bids, allocation: dict) -> float:
    bid_map = {b.agent_id: b for b in bids}
    total = 0.0
    for nid, wl in allocation.items():
        if nid not in bid_map: continue
        b = bid_map[nid]
        q = b.quality_promise * b.past_performance
        if wl > b.capacity > 0:
            q *= b.capacity / wl
        total += q * wl * 100 - b.cost_per_task * wl
    return total


# ═════════════════════════════════════════════════════
# 可视化
# ═════════════════════════════════════════════════════

def plot_results(df1=None, df2=None, df3=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过绘图")
        return

    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    if df1 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for lam, grp in df1.groupby("lambda"):
            axes[0].plot(grp["round"], grp["consensus_energy"], marker="o", ms=4, label=f"λ={lam}")
            axes[1].plot(grp["round"], grp["param_change"],     marker="s", ms=4, label=f"λ={lam}")
        for ax, title, ylabel in zip(
            axes,
            ["Fig 1a: Consensus Energy Convergence", "Fig 1b: Parameter Change"],
            ["Consensus Energy E", "‖Δθ‖"]
        ):
            ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
            ax.set_title(title);    ax.legend(); ax.grid(alpha=0.3)
        axes[1].set_yscale("log")
        fig.tight_layout()
        p = os.path.join(FIGURES_DIR, "fig1_convergence.png")
        fig.savefig(p); plt.close(fig); print(f"  🖼  {p}")

    if df2 is not None and "abs_error" in df2.columns:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        summary = df2.groupby("method").agg(
            MAE=("abs_error",      "mean"),
            Acc=("label_correct",  "mean"),
            AvgU=("utility",       "mean"),
        ).reset_index()
        colors = plt.cm.Set2.colors
        for ax, col, title, ylabel in zip(
            axes,
            ["MAE", "Acc", "AvgU"],
            ["Fig 2a: MAE (↓)", "Fig 2b: Accuracy (↑)", "Fig 2c: Avg Utility"],
            ["MAE", "Accuracy", "Avg Utility"]
        ):
            ax.bar(summary["method"], summary[col], color=colors[:len(summary)])
            ax.set_title(title); ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(FIGURES_DIR, "fig2_semantic_comparison.png")
        fig.savefig(p); plt.close(fig); print(f"  🖼  {p}")

    if df3 is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        order = ["Random", "Pure-Consensus", "Pure-Stackelberg", "SCG (Ours)"]
        data  = [df3[df3["method"] == m]["social_welfare"].values
                 for m in order if m in df3["method"].values]
        labs  = [m for m in order if m in df3["method"].values]
        bp    = ax.boxplot(data, labels=labs, patch_artist=True)
        for patch, c in zip(bp["boxes"], plt.cm.Set2.colors):
            patch.set_facecolor(c)
        ax.set_title("Fig 3: Social Welfare Comparison")
        ax.set_ylabel("Social Welfare"); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        p = os.path.join(FIGURES_DIR, "fig3_baseline_comparison.png")
        fig.savefig(p); plt.close(fig); print(f"  🖼  {p}")


# ═════════════════════════════════════════════════════
# 主函数
# ═════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="博弈驱动的去中心化语义机制 - 实验框架")
    parser.add_argument("--exp",    type=str, default="all",
                        choices=["1", "2", "3", "all"], help="运行指定实验")
    parser.add_argument("--plot",   action="store_true",    help="生成论文图表")
    parser.add_argument("--regen",  action="store_true",    help="强制重新生成实验数据")
    parser.add_argument("--nodes",  type=int, default=3,    help="每轮节点数（默认3）")
    parser.add_argument("--rounds", type=int, default=15,   help="收敛实验迭代轮数")
    parser.add_argument("--trials", type=int, default=20,   help="基线对比随机试验次数")
    parser.add_argument("--agents", type=int, default=5,    help="Stackelberg 博弈节点数")
    args = parser.parse_args()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   博弈驱动的去中心化语义机制 · 实验框架                        ║")
    print("║   Game-Driven Decentralized Semantic Mechanism               ║")
    print("╚" + "═" * 68 + "╝")

    run_all = args.exp == "all"

    if run_all or args.exp == "2":
        print("\n📡 准备多节点共识数据集...")
        _load_dataset(force_regenerate=args.regen, n_nodes=args.nodes)

    df1 = df2 = df3 = None

    if run_all or args.exp == "1":
        df1 = exp1_convergence(num_rounds=args.rounds, num_agents=args.agents)

    if run_all or args.exp == "2":
        df2 = exp2_semantic_comparison(force_regenerate=False)

    if run_all or args.exp == "3":
        df3 = exp3_baseline_comparison(
            num_trials=args.trials,
            num_agents=args.agents,
            num_rounds=args.rounds,
        )

    if args.plot:
        print("\n" + "=" * 70)
        print("🎨 生成论文图表...")
        plot_results(df1, df2, df3)

    print("\n" + "=" * 70)
    print("✅ 实验完成！结果保存在 results/ 目录")
    if df1 is not None:
        best_lam = df1.groupby("lambda")["consensus_energy"].last().idxmin()
        best_e   = df1.groupby("lambda")["consensus_energy"].last().min()
        print(f"  Exp1: λ={best_lam} 收敛最快，终态能量={best_e:.6f}")
    if df2 is not None and "abs_error" in df2.columns:
        best_m   = df2.groupby("method")["abs_error"].mean().idxmin()
        best_mae = df2.groupby("method")["abs_error"].mean().min()
        print(f"  Exp2: {best_m} 方法 MAE 最低 ({best_mae:.4f})")
    if df3 is not None:
        best_sw  = df3.groupby("method")["social_welfare"].mean().idxmax()
        best_val = df3.groupby("method")["social_welfare"].mean().max()
        print(f"  Exp3: {best_sw} 社会福利最高 ({best_val:.2f})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
