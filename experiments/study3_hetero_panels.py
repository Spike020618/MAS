"""
Study 3 —— 异质面板：偏置对冲 + 分歧利用（离线模拟，零 LLM 成本）

数据：150 篇共同子集，每篇 5×V3 + 5×R1 + 5×GLM = 15 个独立评审（r1 轮）。
诊断已知：V3/R1 是拒稿吸引子（stance ~90% reject，均分 ~4.5），
GLM 是接收吸引子（98.5% accept，均分 7.9）。

假设 A（偏置对冲）：混合组成（V3+R1+GLM）+ 校准 > 最好同质面板 + 校准。
假设 B（分歧重审）：高分歧面板追加独立评审 > 固定面板。

评估：train/test = 75/75 分层（seed=42）；t 在 train 网格调；
评审维 bootstrap B=500 报 mean±CI；配对 bootstrap 检验混合 vs 最好同质。

Usage:
  python experiments/study3_hetero_panels.py
"""
from __future__ import annotations

import json

import numpy as np

FILES = {
    "V3": "experiments/results/review/panel/debate_2024.json",
    "R1": "experiments/results/review/panel/debate_r1_sub150_2024.json",
    "GLM": "experiments/results/review/panel/debate_glm_sub150_2024.json",
}
OUT_PATH = "experiments/results/review/study3_hetero_panels.json"
STRATA = ["clear_accept", "borderline", "clear_reject"]
STANCE_MAP = {"accept": 1, "weak_accept": 1, "reject": 0, "weak_reject": 0}
B = 500
RNG = np.random.RandomState(7)


def load_all():
    """返回 {pid: {"V3": [(rating, stance01)], "R1": [...], "GLM": [...],
    stratum, y}}，仅三者交集。"""
    pools = {}
    for model, path in FILES.items():
        with open(path, encoding="utf-8") as f:
            for p in json.load(f)["papers"]:
                d = pools.setdefault(p["paper_id"], {
                    "stratum": p["stratum"], "y": int(p["decision_binary"])})
                revs = []
                for r in p["rounds"][0]["reviews"]:
                    s = STANCE_MAP.get((r["stance"] or "").lower())
                    if r["rating"] is not None and s is not None:
                        revs.append((float(r["rating"]), int(s)))
                d[model] = revs
    keep = {pid: d for pid, d in pools.items()
            if all(len(d.get(m, [])) >= 5 for m in FILES)}
    return keep


def stratified_split(pids, pools, seed=42):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for s in STRATA:
        idx = [pid for pid in pids if pools[pid]["stratum"] == s]
        idx = sorted(idx)
        rng.shuffle(idx)
        half = len(idx) // 2
        tr += idx[:half]
        te += idx[half:]
    return sorted(tr), sorted(te)


def sample_panel(pools, pid, comp, rng):
    """按组成 {model: n} 从各池有放回采样，返回 (ratings, stances)。"""
    ratings, stances = [], []
    for m, n in comp.items():
        pool = pools[pid][m]
        idx = rng.randint(0, len(pool), size=n)
        for i in idx:
            ratings.append(pool[i][0])
            stances.append(pool[i][1])
    return np.array(ratings), np.array(stances)


def tune_threshold(pools, tr, comp, n_rep=60):
    """train 上网格调 t：对 train 面板多次采样取平均一致率。"""
    ytr = np.array([pools[pid]["y"] for pid in tr])
    best_t, best_a = 6.0, -1
    for t in np.arange(4.0, 8.51, 0.1):
        accs = []
        for _ in range(n_rep):
            dec = []
            for pid in tr:
                rt, _ = sample_panel(pools, pid, comp, RNG)
                dec.append(int(rt.mean() >= t))
            accs.append((np.array(dec) == ytr).mean())
        a = float(np.mean(accs))
        if a > best_a:
            best_a, best_t = a, t
    return best_t


def eval_comp(pools, te, comp, t, b=B):
    """test 上 B 次面板采样，返回一致率数组。"""
    yte = np.array([pools[pid]["y"] for pid in te])
    out = np.zeros(b)
    for k in range(b):
        dec = []
        for pid in te:
            rt, _ = sample_panel(pools, pid, comp, RNG)
            dec.append(int(rt.mean() >= t))
        out[k] = (np.array(dec) == yte).mean()
    return out


def eval_comp_mv(pools, te, comp, b=B):
    """多数投票口径（不用校准），作对照。"""
    yte = np.array([pools[pid]["y"] for pid in te])
    out = np.zeros(b)
    for k in range(b):
        dec = []
        for pid in te:
            _, st = sample_panel(pools, pid, comp, RNG)
            dec.append(int(st.mean() > 0.5))
        out[k] = (np.array(dec) == yte).mean()
    return out


def main():
    pools = load_all()
    pids = sorted(pools.keys())
    tr, te = stratified_split(pids, pools)
    print(f"共同子集 {len(pids)} 篇 | train {len(tr)} / test {len(te)}")

    comps = {
        "5V3":          {"V3": 5},
        "5R1":          {"R1": 5},
        "5GLM":         {"GLM": 5},
        "2V3+2R1+1GLM": {"V3": 2, "R1": 2, "GLM": 1},
        "3R1+2V3":      {"R1": 3, "V3": 2},
        "2V3+2R1+2GLM": {"V3": 2, "R1": 2, "GLM": 2},
        "1V3+1R1+1GLM": {"V3": 1, "R1": 1, "GLM": 1},
        "15全员":        {"V3": 5, "R1": 5, "GLM": 5},
    }

    # ── 假设 A：组成 × 校准（均分阈值）─────────────────────
    print("\n===== 假设 A：面板组成 × 校准聚合（test, B=500 bootstrap）=====")
    print(f"{'组成':<18s}{'MV(无校准)':>12s}{'校准(均值±95%CI)':>22s}{'t*':>6s}")
    results = {"n": len(pids), "n_train": len(tr), "n_test": len(te),
               "comps": {}}
    curves = {}
    for name, comp in comps.items():
        t = tune_threshold(pools, tr, comp)
        acc_cal = eval_comp(pools, te, comp, t)
        acc_mv = eval_comp_mv(pools, te, comp)
        curves[name] = (acc_cal, acc_mv)
        ci = 1.96 * acc_cal.std()
        results["comps"][name] = {
            "comp": comp, "t_star": t,
            "cal_mean": float(acc_cal.mean()), "cal_ci": float(ci),
            "mv_mean": float(acc_mv.mean()), "mv_ci": float(1.96 * acc_mv.std())}
        print(f"{name:<18s}{acc_mv.mean():>11.1%}{acc_cal.mean():>14.1%}"
              f" ±{ci:.1%}{t:>6.1f}")

    # 配对 bootstrap：每个混合组成 vs 最强同质
    homo = max(["5V3", "5R1", "5GLM"], key=lambda k: curves[k][0].mean())
    print(f"\n最强同质 = {homo}（{curves[homo][0].mean():.1%}）")
    print("配对 bootstrap（同批采样，差值 95%CI 是否含 0）:")
    results["paired_vs_best_homo"] = {"best_homo": homo}
    for name in comps:
        if name == homo or name == "15全员":
            continue
        diff = curves[name][0] - curves[homo][0]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        sig = "✓ 显著" if lo > 0 else ("✗ 显著更差" if hi < 0 else "ns")
        results["paired_vs_best_homo"][name] = {
            "diff_mean": float(diff.mean()),
            "ci": [float(lo), float(hi)], "sig": sig}
        print(f"  {name:<18s} Δ={diff.mean():+.1%}  CI[{lo:+.1%},{hi:+.1%}]  {sig}")

    # 分层分解（最优混合组成）
    best_mix = max([k for k in comps if k not in ("5V3", "5R1", "5GLM")],
                   key=lambda k: curves[k][0].mean())
    print(f"\n最优混合 = {best_mix}，分层一致率（B=500 均值）:")
    for s in STRATA:
        te_s = [pid for pid in te if pools[pid]["stratum"] == s]
        y_s = np.array([pools[pid]["y"] for pid in te_s])
        comp = comps[best_mix]
        accs = np.zeros(100)
        for k in range(100):
            dec = []
            for pid in te_s:
                rt, _ = sample_panel(pools, pid, comp, RNG)
                dec.append(int(rt.mean() >= results["comps"][best_mix]["t_star"]))
            accs[k] = (np.array(dec) == y_s).mean()
        print(f"  {s:<14s} {accs.mean():.1%}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n→ saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
