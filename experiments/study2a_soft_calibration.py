"""
Study 2a' —— 软校准（robust calibration）：消除硬阈值的选择噪声

动机：硬阈值 argmax 在 train 平原上的 t* 选择近乎随机，导致 test 一致率
在 61.5%~71.6% 间漂移（20 次独立 tune 实证）。软校准（1 维 logistic：
面板均分 → P(accept)）参数连续、无网格随机性，结果确定性可复现。

双臂：parallel_2024.json（300）+ debate_2024.json r1 轮（300），
train/test 150/150（seed=42 分层，与 study2a 同口径）。

输出：MV / 硬阈值（单次+20次敏感性分布）/ 软校准 的 test 指标、
分层一致率、McNemar（软校准 vs MV）。

Usage:
  python experiments/study2a_soft_calibration.py
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression

PANELS = {
    "parallel": "results/review/panel/parallel_2024.json",
    "debate_r1": "results/review/panel/debate_2024.json",
}
OUT_PATH = "results/review/study2a_soft_calibration.json"
STRATA = ["clear_accept", "borderline", "clear_reject"]
STANCE_MAP = {"accept": 1, "weak_accept": 1, "reject": 0, "weak_reject": 0}
N_TUNE_REP = 20


def load_panel(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        papers = json.load(f)["papers"]
    rows = []
    for p in papers:
        revs = p["rounds"][0]["reviews"]
        ratings = [r["rating"] for r in revs if r["rating"] is not None]
        stances = [STANCE_MAP.get((r["stance"] or "").lower()) for r in revs]
        stances = [s for s in stances if s is not None]
        rows.append({
            "stratum": p["stratum"], "y": int(p["decision_binary"]),
            "mean": float(np.mean(ratings)) if ratings else np.nan,
            "mv": int(np.mean(stances) > 0.5) if stances else 0,
            "human_mean": float(p["human_score_mean"]),
        })
    return rows


def stratified_split(rows, seed=42):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for s in STRATA:
        idx = [i for i, r in enumerate(rows) if r["stratum"] == s]
        rng.shuffle(idx)
        half = len(idx) // 2
        tr += idx[:half]
        te += idx[half:]
    return sorted(tr), sorted(te)


def eval_dec(dec, y, rows, idx):
    out = {"agreement": float((dec[idx] == y[idx]).mean())}
    for s in STRATA:
        m = np.array([rows[i]["stratum"] == s for i in idx])
        out[f"agree[{s}]"] = float((dec[idx][m] == y[idx][m]).mean())
    return out


def run_arm(name: str, path: str) -> dict:
    rows = [r for r in load_panel(path) if not np.isnan(r["mean"])]
    y = np.array([r["y"] for r in rows])
    mv = np.array([r["mv"] for r in rows])
    mean = np.array([r["mean"] for r in rows])
    human = np.array([r["human_mean"] for r in rows])
    tr, te = stratified_split(rows)
    print(f"\n===== {name}（{len(rows)} 篇，train {len(tr)} / test {len(te)}）=====")

    # ── 硬阈值：单次 argmax（原 study2a 口径）────────────────
    best_t, best_a = 6.0, -1
    for t in np.arange(4.0, 8.51, 0.1):
        a = ((mean[tr] >= t).astype(int) == y[tr]).mean()
        if a > best_a:
            best_a, best_t = a, t
    dec_hard = (mean >= best_t).astype(int)

    # ── 硬阈值敏感性：20 次 bootstrap tune ───────────────────
    t_stars, acc_dist = [], []
    for rep in range(N_TUNE_REP):
        rng2 = np.random.RandomState(rep)
        bi = rng2.randint(0, len(tr), size=len(tr))
        bt, ba = 6.0, -1
        for t in np.arange(4.0, 8.51, 0.05):
            a = ((mean[tr][bi] >= t).astype(int) == y[tr][bi]).mean()
            if a > ba:
                ba, bt = a, t
        t_stars.append(float(bt))
        acc_dist.append(float(((mean[te] >= bt).astype(int) == y[te]).mean()))
    acc_dist = np.array(acc_dist)

    # ── 软校准：1 维 logistic ────────────────────────────────
    clf = LogisticRegression(max_iter=2000).fit(
        mean[tr].reshape(-1, 1), y[tr])
    dec_soft = clf.predict(mean.reshape(-1, 1))
    prob_soft = clf.predict_proba(mean.reshape(-1, 1))[:, 1]
    implied_t = float(-clf.intercept_[0] / clf.coef_[0][0])

    # ── McNemar：软校准 vs MV（test）─────────────────────────
    b = int(((dec_soft[te] == y[te]) & (mv[te] != y[te])).sum())
    c = int(((dec_soft[te] != y[te]) & (mv[te] == y[te])).sum())
    p_mc = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0

    res = {
        "n": len(rows), "n_train": len(tr), "n_test": len(te),
        "hard_threshold": {
            "t_star": best_t, "test": eval_dec(dec_hard, y, rows, te),
            "sensitivity": {
                "n_rep": N_TUNE_REP,
                "t_star_min": min(t_stars), "t_star_max": max(t_stars),
                "t_star_median": float(np.median(t_stars)),
                "test_mean": float(acc_dist.mean()),
                "test_std": float(acc_dist.std()),
                "test_min": float(acc_dist.min()),
                "test_max": float(acc_dist.max()),
            },
        },
        "soft_calibration": {
            "a": float(clf.coef_[0][0]), "b": float(clf.intercept_[0]),
            "implied_threshold": implied_t,
            "test": eval_dec(dec_soft, y, rows, te),
            "calib_mae_vs_human": float(
                np.abs(prob_soft[te] * 10 - human[te]).mean()),
        },
        "majority_vote": eval_dec(mv, y, rows, te),
        "mcnemar_soft_vs_mv": {"b": b, "c": c, "pvalue": p_mc},
    }

    t = res["soft_calibration"]["test"]
    h = res["hard_threshold"]["test"]
    s = res["hard_threshold"]["sensitivity"]
    v = res["majority_vote"]
    print(f"{'方法':<22s}{'总体':>8s}{'clear_acc':>10s}{'borderline':>12s}"
          f"{'clear_rej':>11s}")
    print(f"{'多数投票':<22s}{v['agreement']:>8.1%}{v['agree[clear_accept]']:>10.1%}"
          f"{v['agree[borderline]']:>12.1%}{v['agree[clear_reject]']:>11.1%}")
    print(f"{'硬阈值(单次argmax)':<22s}{h['agreement']:>8.1%}"
          f"{h['agree[clear_accept]']:>10.1%}{h['agree[borderline]']:>12.1%}"
          f"{h['agree[clear_reject]']:>11.1%}  (t*={best_t:.2f})")
    print(f"{'软校准(1维LR)':<22s}{t['agreement']:>8.1%}{t['agree[clear_accept]']:>10.1%}"
          f"{t['agree[borderline]']:>12.1%}{t['agree[clear_reject]']:>11.1%}"
          f"  (隐含t={implied_t:.2f})")
    print(f"硬阈值敏感性: t*∈[{s['t_star_min']:.2f},{s['t_star_max']:.2f}], "
          f"test={s['test_mean']:.1%}±{s['test_std']:.1%} "
          f"∈[{s['test_min']:.1%},{s['test_max']:.1%}]")
    print(f"McNemar 软校准 vs MV: b={b}, c={c}, p={p_mc:.4f}")
    return res


def main():
    results = {name: run_arm(name, path) for name, path in PANELS.items()}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n→ saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
