"""
Study 2a —— 静态激励层：校准聚合 θ*（治"拒稿偏置"）

从 Study 1 parallel（独立评审，300 篇 × 5 评审员）提取信号向量，
train(150) 上用差分进化优化聚合参数 θ*（评审员权重 w_i、信号权重 γ、
校准偏置 τ、决策阈值 t），test(150) 上与基线比较：
  多数投票 / 均分阈值 / meta-review vs θ* 聚合。

目标函数 = 与官方决定的一致率 − λ·评分校准 MAE（vs 人类均分）。

Usage:
  python experiments/study2a_calibrated_aggregation.py --lam 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARALLEL_PATH = "experiments/results/review/panel/parallel_2024.json"
DEBATE_R1_PATH = "experiments/results/review/panel/debate_2024.json"  # r1 轮 = 独立评审
STAR_PATH = "experiments/results/review/panel/star_2024.json"
OUT_PATH = "experiments/results/review/study2a_theta_star.json"
STRATA = ["clear_accept", "borderline", "clear_reject"]

# 论文锚点：证据具体性启发式
ANCHOR_RE = re.compile(
    r"(?i)(section\s*\d|table\s*\d|figure\s*\d|fig\.\s*\d|equation|eq\.\s*\d|"
    r"theorem|lemma|appendix|algorithm\s*\d|line\s*\d|\[\d+\])")


def extract_signals(review: dict) -> dict:
    """单份评审的信号向量。"""
    e_list = review["aeic"].get("E") or []
    n_pos = sum(1 for e in e_list if e.strip().startswith("(+)"))
    n_neg = sum(1 for e in e_list if e.strip().startswith("(-)"))
    n_pol = n_pos + n_neg
    spec = (sum(1 for e in e_list if ANCHOR_RE.search(e)) / len(e_list)
            if e_list else 0.0)
    return {
        "rating": review["rating"],
        "pos_ratio": n_pos / n_pol if n_pol else 0.5,
        "specificity": spec,
        "i_len": len(review["aeic"].get("I", "")),
    }


def load_panel(path: str = PARALLEL_PATH) -> list:
    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)["papers"]
    rows = []
    for p in papers:
        sigs = [extract_signals(r) for r in p["rounds"][0]["reviews"]]
        rows.append({
            "paper_id": p["paper_id"], "stratum": p["stratum"],
            "decision": int(p["decision_binary"]),
            "human_mean": p["human_score_mean"],
            "sigs": sigs,
            "stances": [r["stance"] for r in p["rounds"][0]["reviews"]],
        })
    return rows


def stratified_split(rows: list, seed: int):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for s in STRATA:
        idx = [i for i, r in enumerate(rows) if r["stratum"] == s]
        rng.shuffle(idx)
        half = len(idx) // 2
        tr.extend(idx[:half])
        te.extend(idx[half:])
    return sorted(tr), sorted(te)


def panel_features(rows: list) -> tuple:
    """(N,5) rating / pos_ratio / specificity 矩阵（缺失为 nan）。"""
    n = len(rows)
    R = np.full((n, 5), np.nan)
    P = np.full((n, 5), 0.5)
    S = np.zeros((n, 5))
    for j, row in enumerate(rows):
        for i, sg in enumerate(row["sigs"][:5]):
            if sg["rating"] is not None:
                R[j, i] = sg["rating"]
            P[j, i] = sg["pos_ratio"]
            S[j, i] = sg["specificity"]
    return R, P, S


def theta_aggregate(theta, R, P, S):
    """θ* 聚合：返回 (score, decision)。
    theta = [w0..w4, g1, g2, tau, threshold]"""
    w = np.clip(theta[:5], 0, None)
    g1, g2, tau, thr = theta[5], theta[6], theta[7], theta[8]
    eff = R + g1 * (S - 0.5) + g2 * (P - 0.5)
    mask = ~np.isnan(eff)
    wmat = np.where(mask, w, 0.0)
    denom = wmat.sum(axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    score = np.where(mask, eff, 0.0).dot(w) / denom + tau
    decision = (score >= thr).astype(int)
    return score, decision


def objective(theta, R, P, S, y, human, lam):
    score, dec = theta_aggregate(theta, R, P, S)
    agree = (dec == y).mean()
    mae = np.abs(score - human).mean()
    return -(agree - lam * mae)  # 最小化


def mean_threshold_baseline(R, thr):
    mean_r = np.nanmean(R, axis=1)
    return mean_r, (mean_r >= thr).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.1, help="MAE 惩罚系数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--panel", default=PARALLEL_PATH,
                    help="面板 json（parallel 或 debate r1 轮）")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    rows = load_panel(args.panel)
    y = np.array([r["decision"] for r in rows])
    human = np.array([r["human_mean"] for r in rows])
    R, P, S = panel_features(rows)
    tr, te = stratified_split(rows, args.seed)
    print(f"train {len(tr)} / test {len(te)}（分层各半，seed={args.seed}）")

    # ── θ* 优化（train）────────────────────────────────────
    bounds = ([(0.05, 2.0)] * 5 + [(-2, 2), (-2, 2), (-3, 3), (4, 8)])
    res = differential_evolution(
        objective, bounds, args=(R[tr], P[tr], S[tr], y[tr], human[tr], args.lam),
        seed=args.seed, maxiter=200, popsize=20, tol=1e-7, polish=True)
    theta_star = res.x
    print(f"θ* = w{np.round(theta_star[:5], 3).tolist()}, "
          f"γ1={theta_star[5]:.3f}, γ2={theta_star[6]:.3f}, "
          f"τ={theta_star[7]:+.3f}, t={theta_star[8]:.3f}")

    # ── 基线：多数投票 / 均分阈值（t 在 train 上调）────────
    stances = [[1 if s == "accept" else 0 if s == "reject" else np.nan
                for s in row["stances"]] for row in rows]
    V = np.array(stances, dtype=float)
    mv_dec = (np.nanmean(V, axis=1) > 0.5).astype(int)

    best_t, best_a = 6.0, -1
    for t in np.arange(4.0, 8.01, 0.1):
        _, d = mean_threshold_baseline(R[tr], t)
        a = (d == y[tr]).mean()
        if a > best_a:
            best_a, best_t = a, t
    print(f"均分阈值基线最优 t*={best_t:.1f}（train 一致率 {best_a:.3f}）")

    # meta-review（star 条件的 hub 决策）
    meta_dec = {}
    if os.path.exists(STAR_PATH):
        with open(STAR_PATH, "r", encoding="utf-8") as f:
            for p in json.load(f)["papers"]:
                md = p.get("meta_decision") or {}
                dec = (md.get("decision") or "").lower()
                if dec in ("accept", "reject"):
                    meta_dec[p["paper_id"]] = int(dec == "accept")

    # ── test 评估 ──────────────────────────────────────────
    def eval_subset(idx, dec_all, score_all=None):
        d, yy = dec_all[idx], y[idx]
        out = {"agreement": float((d == yy).mean()),
               "accept_rate": float(d.mean())}
        for s in STRATA:
            m = np.array([rows[i]["stratum"] == s for i in idx])
            out[f"agree[{s}]"] = float((d[m] == yy[m]).mean())
        if score_all is not None:
            out["calib_mae"] = float(np.abs(score_all[idx] - human[idx]).mean())
        return out

    sc_all, th_dec = theta_aggregate(theta_star, R, P, S)
    _, mt_dec = mean_threshold_baseline(R, best_t)
    te_meta_idx = [i for i in te if rows[i]["paper_id"] in meta_dec]
    meta_arr = np.array([meta_dec[rows[i]["paper_id"]] for i in te_meta_idx])

    results = {
        "config": {"lam": args.lam, "seed": args.seed,
                   "n_train": len(tr), "n_test": len(te)},
        "theta_star": {"w": theta_star[:5].tolist(), "gamma1": float(theta_star[5]),
                       "gamma2": float(theta_star[6]), "tau": float(theta_star[7]),
                       "threshold": float(theta_star[8])},
        "test": {
            "majority_vote": eval_subset(te, mv_dec),
            "mean_threshold": eval_subset(te, mt_dec, np.nanmean(R, axis=1)),
            "theta_star": eval_subset(te, th_dec, sc_all),
            "meta_review": {
                "agreement": float((meta_arr == y[te_meta_idx]).mean()),
                "n": len(te_meta_idx),
            } if te_meta_idx else None,
        },
        "train_agreement_theta": float(-res.fun + args.lam *
                                       np.abs(theta_aggregate(
                                           theta_star, R[tr], P[tr], S[tr])[0]
                                           - human[tr]).mean()),
    }

    # McNemar：θ* vs 多数投票（test），精确二项检验
    from scipy.stats import binomtest
    b = int(((th_dec[te] == y[te]) & (mv_dec[te] != y[te])).sum())
    c = int(((th_dec[te] != y[te]) & (mv_dec[te] == y[te])).sum())
    p_mc = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0
    results["mcnemar_theta_vs_mv"] = {"b": b, "c": c, "pvalue": p_mc}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 报告 ───────────────────────────────────────────────
    t = results["test"]
    print("\n===== test 集决策一致率（n=150）=====")
    print(f"{'方法':<18s}{'总体':>8s}{'clear_acc':>10s}{'borderline':>12s}"
          f"{'clear_rej':>11s}{'accept率':>9s}{'MAE':>7s}")
    for name, key in [("多数投票", "majority_vote"), ("均分阈值", "mean_threshold"),
                      ("θ*聚合", "theta_star")]:
        d = t[key]
        mae = f"{d['calib_mae']:.2f}" if "calib_mae" in d else "—"
        print(f"{name:<18s}{d['agreement']:>8.1%}{d['agree[clear_accept]']:>10.1%}"
              f"{d['agree[borderline]']:>12.1%}{d['agree[clear_reject]']:>11.1%}"
              f"{d['accept_rate']:>9.1%}{mae:>7s}")
    if t["meta_review"]:
        print(f"{'meta-review':<18s}{t['meta_review']['agreement']:>8.1%}"
              f"{'':>10s}{'':>12s}{'':>11s}{'':>9s}")
    print(f"\nMcNemar θ* vs 多数投票: b={b}, c={c}, p={p_mc:.4f}")
    print(f"→ saved to {args.out}")


if __name__ == "__main__":
    main()
