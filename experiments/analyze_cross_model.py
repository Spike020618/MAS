"""
跨模型评审面板对比 — 同一 150 篇子集上，各模型集群的 herding 行为指标

输入：
  - DeepSeek:  experiments/results/review/panel/debate_2024.json（300 篇全集，按子集切片）
  - 其他模型:  experiments/results/review/panel/debate_{tag}_2024.json（150 篇子集原生采集）

指标（每模型）：
  - accept 基准率（round1 独立评审，按分层）
  - 少数派消灭率 + 翻转方向（to_reject / to_accept / persist）
  - 评分方差坍缩（r1→r3 std）
  - 均分漂移（r1→r3）
  - 决策一致率（r1 / r3 多数立场 vs 官方）
  - 骑墙率（C 字段含 "X/Y" 双侧写法的比例，模型行为差异素材）

Usage:
  python experiments/analyze_cross_model.py --year 2024 \
      --models deepseek:experiments/results/review/panel/debate_2024.json \
               glm:experiments/results/review/panel/debate_glm_sub150_2024.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

SUBSET_PATH = "experiments/results/review/review_papers_2024_sub150.json"
STRATA = ["clear_accept", "borderline", "clear_reject"]


def _load_papers(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return {p["paper_id"]: p for p in json.load(f)["papers"]}


def _subset_ids() -> set:
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return {p["paper_id"] for p in json.load(f)["papers"]}


def _majority_stance(reviews: list):
    votes = [r["stance"] for r in reviews if r["stance"]]
    if not votes:
        return None
    n_acc = votes.count("accept")
    if n_acc == len(votes) - n_acc:
        return "tie"
    return "accept" if n_acc > len(votes) / 2 else "reject"


def analyze_model(papers: dict) -> dict:
    out = {}
    # accept 基准率（r1，按分层）
    accept_base = {}
    for stratum in STRATA:
        stances = []
        for p in papers.values():
            if p["stratum"] != stratum:
                continue
            stances.extend(r["stance"] for r in p["rounds"][0]["reviews"]
                           if r["stance"])
        accept_base[stratum] = stances.count("accept") / len(stances) if stances else None
    out["accept_baserate_r1"] = accept_base

    # 少数派命运 + 翻转方向
    to_acc = to_rej = persist = 0
    for p in papers.values():
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
    n_min = to_acc + to_rej + persist
    out["minority"] = {
        "n": n_min,
        "to_reject": to_rej, "to_accept": to_acc, "persist": persist,
        "elimination_rate": (to_acc + to_rej) / n_min if n_min else None,
        "flip_to_reject_share": to_rej / (to_acc + to_rej) if (to_acc + to_rej) else None,
    }

    # 方差坍缩 + 均分漂移
    std1, stdN, m1, mN = [], [], [], []
    unanimous_r1 = 0
    for p in papers.values():
        rt1 = [r["rating"] for r in p["rounds"][0]["reviews"] if r["rating"] is not None]
        rtN = [r["rating"] for r in p["rounds"][-1]["reviews"] if r["rating"] is not None]
        if len(rt1) >= 2 and len(rtN) >= 2:
            std1.append(np.std(rt1, ddof=1))
            stdN.append(np.std(rtN, ddof=1))
        if rt1 and rtN:
            m1.append(np.mean(rt1))
            mN.append(np.mean(rtN))
        st1 = [r["stance"] for r in p["rounds"][0]["reviews"] if r["stance"]]
        if st1 and len(set(st1)) == 1:
            unanimous_r1 += 1
    out["variance_collapse"] = float(1 - np.mean(stdN) / np.mean(std1)) if std1 else None
    out["rating_shift"] = float(np.mean(mN) - np.mean(m1)) if m1 else None
    out["r1_unanimous_rate"] = unanimous_r1 / len(papers) if papers else None

    # 决策一致率（r1 / r3）
    for tag, idx in [("r1", 0), ("r3", -1)]:
        hits = []
        for p in papers.values():
            ms = _majority_stance(p["rounds"][idx]["reviews"])
            if ms in ("accept", "reject"):
                hits.append(int((ms == "accept") == bool(p["decision_binary"])))
        out[f"decision_acc_{tag}"] = float(np.mean(hits)) if hits else None

    # 骑墙率
    hedge = tot = 0
    for p in papers.values():
        for rnd in p["rounds"]:
            for r in rnd["reviews"]:
                c = r["aeic"].get("C", "").strip().lower()
                if not c:
                    continue
                tot += 1
                if re.match(r"^(weak\s+)?(accept\s*[/|&]\s*reject|reject\s*[/|&]\s*accept)", c):
                    hedge += 1
    out["hedge_rate"] = hedge / tot if tot else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--models", nargs="+", required=True,
                    help="name:path 形式，可多个")
    ap.add_argument("--out", default="experiments/results/review/cross_model_analysis.json")
    args = ap.parse_args()

    subset = _subset_ids()
    results = {}
    for spec in args.models:
        name, path = spec.split(":", 1)
        papers = _load_papers(path)
        # 对齐到同一子集（DeepSeek 全集切片；子集原生采集的全保留）
        papers = {k: v for k, v in papers.items() if k in subset}
        print(f"  {name}: {len(papers)} papers (aligned to subset)")
        results[name] = analyze_model(papers)
        results[name]["n_papers"] = len(papers)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 对比表
    print("\n" + "=" * 78)
    names = list(results.keys())
    header = f"{'指标':<34s}" + "".join(f"{n:>16s}" for n in names)
    print(header)
    print("-" * len(header))

    def row(label, fn, fmt="{:.3f}"):
        vals = []
        for n in names:
            try:
                v = fn(results[n])
            except Exception:
                v = None
            vals.append(fmt.format(v) if v is not None else "—")
        print(f"{label:<34s}" + "".join(f"{v:>16s}" for v in vals))

    for s in STRATA:
        row(f"accept基准率[{s}]", lambda r, s=s: r["accept_baserate_r1"][s], "{:.1%}")
    row("r1一致面板率", lambda r: r["r1_unanimous_rate"], "{:.1%}")
    row("少数派消灭率", lambda r: r["minority"]["elimination_rate"], "{:.1%}")
    row("翻转朝reject占比", lambda r: r["minority"]["flip_to_reject_share"], "{:.1%}")
    row("方差坍缩", lambda r: r["variance_collapse"], "{:.1%}")
    row("均分漂移", lambda r: r["rating_shift"], "{:+.2f}")
    row("决策一致率 r1", lambda r: r["decision_acc_r1"], "{:.1%}")
    row("决策一致率 r3", lambda r: r["decision_acc_r3"], "{:.1%}")
    row("骑墙率", lambda r: r["hedge_rate"], "{:.1%}")
    print(f"\n→ saved to {args.out}")


if __name__ == "__main__":
    main()
