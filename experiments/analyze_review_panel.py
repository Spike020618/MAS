"""
Study 1 分析：LLM 评审面板 herding 量化

核心指标：
  1. 决策一致性: 各条件 vs 官方决定（多数投票/meta-review 立场 → binary）
  2. 评分校准:   |面板均分 - 人类均分| MAE
  3. Herding 动力学（debate 主条件）:
     - SCR (Stance Change Rate) 每次轮间转移
     - MR (Majority Ratio) 每轮
     - 少数派消灭率: round1 少数派立场在 round3 被翻转的比例
     - 评分方差坍缩: rating std round1 → round3（对标 AgentReview 的 27.2%）
     - 决策漂移: round1 多数立场 vs round3 多数立场 vs 官方决定
  4. 分层对比: borderline vs clear_accept vs clear_reject

Usage:
  python experiments/analyze_review_panel.py --year 2024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

PANEL_DIR = "experiments/results/review/panel"
OUT_PATH = "experiments/results/review/study1_analysis.json"
STRATA = ["clear_accept", "borderline", "clear_reject"]


def _load(condition: str, year: int) -> list:
    path = os.path.join(PANEL_DIR, f"{condition}_{year}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["papers"]


def _majority_stance(reviews: list) -> str | None:
    votes = [r["stance"] for r in reviews if r["stance"]]
    if not votes:
        return None
    n_acc = votes.count("accept")
    if n_acc == len(votes) - n_acc:
        return "tie"
    return "accept" if n_acc > len(votes) / 2 else "reject"


def _mean_rating(reviews: list) -> float | None:
    ratings = [r["rating"] for r in reviews if r["rating"] is not None]
    return float(np.mean(ratings)) if ratings else None


def _stance_metrics(rounds: list) -> dict:
    """debate 轮间动力学指标。"""
    scr_list, mr_list, std_list = [], [], []
    prev_stances = None
    for rnd in rounds:
        stances = [r["stance"] for r in rnd["reviews"] if r["stance"]]
        ratings = [r["rating"] for r in rnd["reviews"] if r["rating"] is not None]
        n = len(stances)
        if n:
            n_acc = stances.count("accept")
            mr_list.append(max(n_acc, n - n_acc) / n)
        if len(ratings) >= 2:
            std_list.append(float(np.std(ratings, ddof=1)))
        if prev_stances is not None and stances:
            flips = sum(1 for a, b in zip(prev_stances, stances)
                        if a and b and a != b)
            scr_list.append(flips / len(stances))
        prev_stances = stances

    # 少数派消灭率: round1 的少数派中, 最终轮转到多数派的比例
    minority_elim = None
    if len(rounds) >= 2:
        r1 = [r["stance"] for r in rounds[0]["reviews"] if r["stance"]]
        rN = [r["stance"] for r in rounds[-1]["reviews"] if r["stance"]]
        if len(r1) == len(rN) and r1:
            maj1 = "accept" if r1.count("accept") >= len(r1) / 2 else "reject"
            minority_idx = [i for i, s in enumerate(r1) if s and s != maj1]
            if minority_idx:
                flipped = sum(1 for i in minority_idx if rN[i] == maj1)
                minority_elim = flipped / len(minority_idx)

    return {
        "scr_per_transition": scr_list,
        "mr_per_round": mr_list,
        "rating_std_per_round": std_list,
        "minority_elimination": minority_elim,
    }


def analyze_year(year: int) -> dict:
    parallel = {p["paper_id"]: p for p in _load("parallel", year)}
    debate = {p["paper_id"]: p for p in _load("debate", year)}
    star = {p["paper_id"]: p for p in _load("star", year)}
    print(f"  loaded: parallel={len(parallel)} debate={len(debate)} star={len(star)}")

    results = {"year": year, "n": {"parallel": len(parallel),
                                   "debate": len(debate), "star": len(star)}}

    # ── 1. 决策一致性 & 校准（按条件 × 分层）─────────────────
    cond_acc = defaultdict(lambda: defaultdict(list))
    cond_mae = defaultdict(lambda: defaultdict(list))
    for pid, p in parallel.items():
        ms = _majority_stance(p["rounds"][0]["reviews"])
        mr_ = _mean_rating(p["rounds"][0]["reviews"])
        if ms in ("accept", "reject"):
            cond_acc["parallel"][p["stratum"]].append(
                int((ms == "accept") == bool(p["decision_binary"])))
        if mr_ is not None:
            cond_mae["parallel"][p["stratum"]].append(abs(mr_ - p["human_score_mean"]))
    for pid, p in debate.items():
        ms = _majority_stance(p["rounds"][-1]["reviews"])
        mr_ = _mean_rating(p["rounds"][-1]["reviews"])
        if ms in ("accept", "reject"):
            cond_acc["debate_rN"][p["stratum"]].append(
                int((ms == "accept") == bool(p["decision_binary"])))
        ms1 = _majority_stance(p["rounds"][0]["reviews"])
        if ms1 in ("accept", "reject"):
            cond_acc["debate_r1"][p["stratum"]].append(
                int((ms1 == "accept") == bool(p["decision_binary"])))
        if mr_ is not None:
            cond_mae["debate_rN"][p["stratum"]].append(abs(mr_ - p["human_score_mean"]))
    for pid, p in star.items():
        mrev = p.get("meta_review", {})
        if mrev.get("stance") in ("accept", "reject"):
            cond_acc["star_meta"][p["stratum"]].append(
                int((mrev["stance"] == "accept") == bool(p["decision_binary"])))

    results["decision_accuracy"] = {
        c: {"overall": float(np.mean(sum(v.values(), []))) if any(v.values()) else None,
            **{s: float(np.mean(lst)) for s, lst in v.items() if lst}}
        for c, v in cond_acc.items()}
    results["rating_mae"] = {
        c: {"overall": float(np.mean(sum(v.values(), []))) if any(v.values()) else None,
            **{s: float(np.mean(lst)) for s, lst in v.items() if lst}}
        for c, v in cond_mae.items()}

    # ── 2. Debate herding 动力学 ─────────────────────────────
    dyn = defaultdict(lambda: {"scr": [], "mr": [], "std_r1": [], "std_rN": [],
                               "minority_elim": [], "drift": []})
    for pid, p in debate.items():
        m = _stance_metrics(p["rounds"])
        stratum = p["stratum"]
        if m["scr_per_transition"]:
            dyn[stratum]["scr"].append(float(np.mean(m["scr_per_transition"])))
        if m["mr_per_round"]:
            dyn[stratum]["mr"].append(m["mr_per_round"])
        if len(m["rating_std_per_round"]) >= 2:
            dyn[stratum]["std_r1"].append(m["rating_std_per_round"][0])
            dyn[stratum]["std_rN"].append(m["rating_std_per_round"][-1])
        if m["minority_elimination"] is not None:
            dyn[stratum]["minority_elim"].append(m["minority_elimination"])
        ms1 = _majority_stance(p["rounds"][0]["reviews"])
        msN = _majority_stance(p["rounds"][-1]["reviews"])
        if ms1 in ("accept", "reject") and msN in ("accept", "reject"):
            dyn[stratum]["drift"].append(int(ms1 != msN))

    results["debate_dynamics"] = {}
    for stratum, d in dyn.items():
        if not d["scr"]:
            continue
        std_r1 = np.array(d["std_r1"]) if d["std_r1"] else np.array([np.nan])
        std_rN = np.array(d["std_rN"]) if d["std_rN"] else np.array([np.nan])
        var_collapse = (1 - np.nanmean(std_rN) / np.nanmean(std_r1)) if np.nanmean(std_r1) > 0 else None
        results["debate_dynamics"][stratum] = {
            "n": len(d["scr"]),
            "mean_SCR": float(np.mean(d["scr"])),
            "mean_MR_final": float(np.mean([mr[-1] for mr in d["mr"] if mr])),
            "minority_elimination_rate": float(np.mean(d["minority_elim"])) if d["minority_elim"] else None,
            "rating_std_round1": float(np.nanmean(std_r1)),
            "rating_std_final": float(np.nanmean(std_rN)),
            "rating_variance_collapse": float(var_collapse) if var_collapse is not None else None,
            "decision_drift_rate": float(np.mean(d["drift"])) if d["drift"] else None,
        }

    # ── 3. 深度诊断 ─────────────────────────────────────────
    # 3a. 独立评审 accept 基准率（按分层）
    accept_base = {}
    for stratum in STRATA:
        rates = []
        for p in parallel.values():
            if p["stratum"] != stratum:
                continue
            st = [r["stance"] for r in p["rounds"][0]["reviews"] if r["stance"]]
            if st:
                rates.append(st.count("accept") / len(st))
        if rates:
            accept_base[stratum] = float(np.mean(rates))
    results["accept_baserate_parallel"] = accept_base

    # 3b. debate round1 面板构成
    n_unanimous = sum(
        1 for p in debate.values()
        if len({r["stance"] for r in p["rounds"][0]["reviews"] if r["stance"]}) == 1
    )
    results["debate_round1_unanimous"] = n_unanimous
    results["debate_round1_unanimous_rate"] = n_unanimous / max(len(debate), 1)

    # 3c. 少数派翻转方向（herding 的方向性）
    flip = {"to_accept": 0, "to_reject": 0, "persist": 0}
    for p in debate.values():
        r1 = [r["stance"] for r in p["rounds"][0]["reviews"] if r["stance"]]
        rN = [r["stance"] for r in p["rounds"][-1]["reviews"] if r["stance"]]
        if len(r1) != len(rN) or len(set(r1)) == 1:
            continue
        maj = "accept" if r1.count("accept") > len(r1) / 2 else "reject"
        for i, s in enumerate(r1):
            if s == maj:
                continue
            if rN[i] == maj:
                flip["to_accept" if maj == "accept" else "to_reject"] += 1
            else:
                flip["persist"] += 1
    results["minority_flip_direction"] = flip

    # 3d. 讨论前后均分漂移（按分层）
    rating_shift = {}
    for stratum in STRATA:
        d1, dN = [], []
        for p in debate.values():
            if p["stratum"] != stratum:
                continue
            m1 = _mean_rating(p["rounds"][0]["reviews"])
            mN = _mean_rating(p["rounds"][-1]["reviews"])
            if m1 is not None and mN is not None:
                d1.append(m1); dN.append(mN)
        if d1:
            rating_shift[stratum] = {
                "mean_round1": float(np.mean(d1)),
                "mean_final": float(np.mean(dN)),
                "delta": float(np.mean(dN) - np.mean(d1)),
            }
    results["debate_rating_shift"] = rating_shift

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", type=str, default=OUT_PATH)
    args = ap.parse_args()

    results = analyze_year(args.year)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== 决策一致性 (vs 官方) ===")
    for c, v in results["decision_accuracy"].items():
        print(f"  {c}: overall={v.get('overall')}, "
              + ", ".join(f"{s}={x:.2f}" for s, x in v.items() if s != "overall" and x is not None))
    print("\n=== 评分校准 MAE ===")
    for c, v in results["rating_mae"].items():
        print(f"  {c}: overall={v.get('overall')}")
    print("\n=== Debate Herding 动力学 ===")
    for s, v in results.get("debate_dynamics", {}).items():
        print(f"  [{s}] n={v['n']} SCR={v['mean_SCR']:.3f} "
              f"minority_elim={v.get('minority_elimination_rate')} "
              f"var_collapse={v.get('rating_variance_collapse')} "
              f"drift={v.get('decision_drift_rate')}")
    print("\n=== 深度诊断 ===")
    print(f"  独立accept基准率: {results.get('accept_baserate_parallel')}")
    print(f"  round1一致面板: {results.get('debate_round1_unanimous')}/300 "
          f"({results.get('debate_round1_unanimous_rate', 0):.1%})")
    print(f"  少数派翻转方向: {results.get('minority_flip_direction')}")
    for s, v in results.get("debate_rating_shift", {}).items():
        print(f"  均分漂移[{s}]: {v['mean_round1']:.2f} → {v['mean_final']:.2f} "
              f"(Δ={v['delta']:+.2f})")
    print(f"\n→ saved to {args.out}")


if __name__ == "__main__":
    main()
