"""
Study 1 统计检验 — 评审面板 herding 诊断的显著性验证

检验清单：
  1. 少数派翻转方向性: 二项检验（to_reject vs to_accept）
  2. 少数派消灭率: Wilson 95% CI
  3. 评分方差坍缩: Wilcoxon 符号秩（round1 std vs round3 std, 逐论文配对）
  4. 评分下移: Wilcoxon 符号秩（round1 均分 vs round3 均分, 逐论文配对）
  5. 决策一致性: McNemar（debate_r1 vs debate_r3, 被试内）+ 各条件 Wilson CI
  6. 拒稿偏置: 分层 accept 率的 Wilson CI + 卡方检验（层间差异）
  7. 效应量: Cohen's h（决策一致性关键对比）

Usage:
  python experiments/study1_stats.py --year 2024
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

PANEL_DIR = "experiments/results/review/panel"
OUT_PATH = "experiments/results/review/study1_stats.json"
STRATA = ["clear_accept", "borderline", "clear_reject"]


def _load(condition: str, year: int) -> list:
    path = os.path.join(PANEL_DIR, f"{condition}_{year}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["papers"]


def _majority_stance(reviews: list):
    votes = [r["stance"] for r in reviews if r["stance"]]
    if not votes:
        return None
    n_acc = votes.count("accept")
    if n_acc == len(votes) - n_acc:
        return "tie"
    return "accept" if n_acc > len(votes) / 2 else "reject"


def _wilson(k: int, n: int) -> dict:
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return {"k": k, "n": n, "rate": k / n, "ci_low": ci.low, "ci_high": ci.high}


def _mcnemar(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(stats.binomtest(b, b + c, 0.5).pvalue)


def _cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", type=str, default=OUT_PATH)
    args = ap.parse_args()

    parallel = _load("parallel", args.year)
    debate = _load("debate", args.year)
    star = _load("star", args.year)
    results = {"year": args.year, "n_per_condition": {
        "parallel": len(parallel), "debate": len(debate), "star": len(star)}}

    print("=" * 64)
    print("Study 1 统计检验")
    print("=" * 64)

    # ── 1+2. 少数派翻转：方向性 + 消灭率 ─────────────────────
    to_acc = to_rej = persist = 0
    for p in debate:
        r1 = [r["stance"] for r in p["rounds"][0]["reviews"] if r["stance"]]
        rN = [r["stance"] for r in p["rounds"][-1]["reviews"] if r["stance"]]
        if len(r1) != len(rN) or len(set(r1)) == 1:
            continue
        maj = "accept" if r1.count("accept") > len(r1) / 2 else "reject"
        for i, s in enumerate(r1):
            if s == maj:
                continue
            if rN[i] == maj:
                if maj == "accept":
                    to_acc += 1
                else:
                    to_rej += 1
            else:
                persist += 1

    n_flips = to_acc + to_rej
    p_dir = float(stats.binomtest(to_rej, n_flips, 0.5).pvalue) if n_flips else 1.0
    n_minority = n_flips + persist
    elim = _wilson(n_flips, n_minority)
    results["minority_flips"] = {
        "to_reject": to_rej, "to_accept": to_acc, "persist": persist,
        "direction_binomial_p": p_dir,
        "elimination_rate": elim,
    }
    print(f"\n[1] 少数派翻转方向: to_reject={to_rej} to_accept={to_acc} "
          f"二项检验 p={p_dir:.2e}")
    print(f"[2] 少数派消灭率: {elim['rate']:.1%} "
          f"95% CI [{elim['ci_low']:.1%}, {elim['ci_high']:.1%}] (n={n_minority})")

    # ── 3. 方差坍缩（Wilcoxon, 全体 + 分层）──────────────────
    var_results = {}
    for stratum in [None] + STRATA:
        std1, stdN = [], []
        for p in debate:
            if stratum and p["stratum"] != stratum:
                continue
            r1 = [r["rating"] for r in p["rounds"][0]["reviews"] if r["rating"] is not None]
            rN = [r["rating"] for r in p["rounds"][-1]["reviews"] if r["rating"] is not None]
            if len(r1) >= 2 and len(rN) >= 2:
                std1.append(np.std(r1, ddof=1))
                stdN.append(np.std(rN, ddof=1))
        if len(std1) < 10:
            continue
        w = stats.wilcoxon(std1, stdN, alternative="greater")
        key = stratum or "overall"
        var_results[key] = {
            "n": len(std1),
            "std_round1_mean": float(np.mean(std1)),
            "std_round3_mean": float(np.mean(stdN)),
            "collapse": float(1 - np.mean(stdN) / np.mean(std1)),
            "wilcoxon_W": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
        }
        print(f"[3] 方差坍缩[{key}]: {np.mean(std1):.3f} → {np.mean(stdN):.3f} "
              f"(-{100*(1-np.mean(stdN)/np.mean(std1)):.1f}%) "
              f"Wilcoxon p={w.pvalue:.2e}")
    results["variance_collapse"] = var_results

    # ── 4. 评分下移（Wilcoxon, 全体）─────────────────────────
    m1_all, mN_all = [], []
    for p in debate:
        r1 = [r["rating"] for r in p["rounds"][0]["reviews"] if r["rating"] is not None]
        rN = [r["rating"] for r in p["rounds"][-1]["reviews"] if r["rating"] is not None]
        if r1 and rN:
            m1_all.append(np.mean(r1))
            mN_all.append(np.mean(rN))
    w = stats.wilcoxon(m1_all, mN_all, alternative="greater")
    results["rating_downshift"] = {
        "n": len(m1_all),
        "mean_round1": float(np.mean(m1_all)),
        "mean_round3": float(np.mean(mN_all)),
        "delta": float(np.mean(mN_all) - np.mean(m1_all)),
        "wilcoxon_p": float(w.pvalue),
    }
    print(f"[4] 评分下移: {np.mean(m1_all):.3f} → {np.mean(mN_all):.3f} "
          f"Wilcoxon p={w.pvalue:.2e}")

    # ── 5. 决策一致性：McNemar + Wilson CI ───────────────────
    def _correct_list(cond_papers, round_idx, use_meta=False):
        out = []
        for p in cond_papers:
            if use_meta:
                ms = p.get("meta_review", {}).get("stance")
            else:
                ms = _majority_stance(p["rounds"][round_idx]["reviews"])
            if ms in ("accept", "reject"):
                out.append(int((ms == "accept") == bool(p["decision_binary"])))
        return out

    acc_par = _correct_list(parallel, 0)
    acc_d1 = _correct_list(debate, 0)
    acc_dN = _correct_list(debate, -1)
    acc_star = _correct_list(star, 0, use_meta=True)

    dec = {}
    for name, arr in [("parallel", acc_par), ("debate_r1", acc_d1),
                      ("debate_r3", acc_dN), ("star_meta", acc_star)]:
        ci = _wilson(sum(arr), len(arr))
        dec[name] = ci
        print(f"[5] {name}: {ci['rate']:.1%} 95% CI [{ci['ci_low']:.1%}, {ci['ci_high']:.1%}]")

    b = sum(1 for x, y in zip(acc_d1, acc_dN) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(acc_d1, acc_dN) if x == 0 and y == 1)
    p_mn = _mcnemar(b, c)
    dec["mcnemar_r1_vs_r3"] = {"b": b, "c": c, "p": p_mn,
                               "cohens_h": _cohens_h(np.mean(acc_d1), np.mean(acc_dN))}
    b2 = sum(1 for x, y in zip(acc_par, acc_star) if x == 1 and y == 0)
    c2 = sum(1 for x, y in zip(acc_par, acc_star) if x == 0 and y == 1)
    dec["mcnemar_parallel_vs_star"] = {"b": b2, "c": c2, "p": _mcnemar(b2, c2)}
    print(f"    McNemar r1 vs r3: b={b} c={c} p={p_mn:.4f}")
    print(f"    McNemar parallel vs star: b={b2} c={c2} p={dec['mcnemar_parallel_vs_star']['p']:.4f}")
    results["decision_accuracy"] = dec

    # ── 6. 拒稿偏置：分层 accept 率 CI + 卡方 ────────────────
    accept_rates = {}
    table = []
    for stratum in STRATA:
        stances = []
        for p in parallel:
            if p["stratum"] != stratum:
                continue
            stances.extend(r["stance"] for r in p["rounds"][0]["reviews"] if r["stance"])
        k = stances.count("accept")
        accept_rates[stratum] = _wilson(k, len(stances))
        table.append([k, len(stances) - k])
        print(f"[6] accept率[{stratum}]: {k/len(stances):.1%} "
              f"95% CI [{accept_rates[stratum]['ci_low']:.1%}, {accept_rates[stratum]['ci_high']:.1%}]")
    chi2, p_chi, _, _ = stats.chi2_contingency(table)
    accept_rates["chi2_strata"] = {"chi2": float(chi2), "p": float(p_chi)}
    print(f"    层间卡方: chi2={chi2:.1f} p={p_chi:.2e}")
    results["accept_baserate"] = accept_rates

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n→ saved to {args.out}")


if __name__ == "__main__":
    main()
