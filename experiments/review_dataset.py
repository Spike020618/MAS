"""
评审数据集构建 — 从 berenslab/iclr-dataset parquet 分层抽样 ICLR 论文

分层逻辑（基于官方决定 + 人类评审分数分布）：
  - clear_accept:  Oral/Spotlight（明确接收）
  - borderline:    分数与决定冲突或高方差（聚合机制的主战场）
                   · Accept(poster) 但 mean < 5.5（争议接收）
                   · Reject 但 mean >= 5.5（争议拒稿）
                   · 任何决定但 max-min >= 3（评审员严重分歧）
  - clear_reject:  Reject 且 mean <= 4.0（明确拒稿）

Usage:
  python experiments/review_dataset.py --year 2024 --n_per_stratum 100
  python experiments/review_dataset.py --year 2024 --n_per_stratum 5 --seed 0  # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARQUET_PATH = "experiments/results/review/iclr26v1.parquet"
OUT_DIR = "experiments/results/review"

# 分数区间（ICLR 1-10 评分制）
BORDERLINE_MEAN_ACCEPT_MAX = 5.5   # poster 接收但均分低于此 → 争议接收
BORDERLINE_MEAN_REJECT_MIN = 5.5   # 拒稿但均分高于此 → 争议拒稿
BORDERLINE_VAR_MIN = 3.0           # 分差大于此 → 评审员严重分歧
CLEAR_REJECT_MEAN_MAX = 4.0        # 拒稿且均分低于此 → 明确拒稿


def _valid_scores(s) -> bool:
    return isinstance(s, (list, np.ndarray)) and len(s) > 0


def assign_stratum(decision: str, scores) -> str | None:
    """按决定与分数分布给论文分层。返回 None 表示不入样（如 Withdrawn/Desk rejected）。"""
    d = str(decision).lower()
    if "withdraw" in d or "desk" in d or "workshop" in d:
        return None
    if not _valid_scores(scores):
        return None
    mean_s = float(np.mean(scores))
    var_s = float(np.max(scores) - np.min(scores))

    is_accept_top = ("oral" in d) or ("spotlight" in d) or ("notable" in d)
    is_accept_poster = "poster" in d
    is_reject = d.strip() == "reject"

    # 高分歧永远优先归入 borderline
    if var_s >= BORDERLINE_VAR_MIN and not is_accept_top:
        return "borderline"
    if is_accept_top:
        return "clear_accept"
    if is_accept_poster:
        return "borderline" if mean_s < BORDERLINE_MEAN_ACCEPT_MAX else "clear_accept"
    if is_reject:
        if mean_s >= BORDERLINE_MEAN_REJECT_MIN:
            return "borderline"
        if mean_s <= CLEAR_REJECT_MEAN_MAX:
            return "clear_reject"
        return "borderline"  # 中间地带（4.0 < mean < 5.5 的拒稿）也算边界
    return None


def build_review_dataset(year: int, n_per_stratum: int, seed: int,
                         out_path: str) -> dict:
    """主入口：分层抽样并保存。"""
    df = pd.read_parquet(PARQUET_PATH)
    sub = df[df.year == year].copy()
    print(f"  ICLR {year}: {len(sub)} submissions")

    sub["stratum"] = [assign_stratum(r.decision, r.scores) for r in sub.itertuples()]
    pooled = sub.dropna(subset=["stratum"])
    print("  分层池分布:")
    for s, cnt in pooled.stratum.value_counts().items():
        print(f"    {s}: {cnt}")

    rng = np.random.RandomState(seed)
    records = []
    for stratum in ["clear_accept", "borderline", "clear_reject"]:
        pool = pooled[pooled.stratum == stratum]
        n_take = min(n_per_stratum, len(pool))
        idx = rng.choice(pool.index, size=n_take, replace=False)
        for i in idx:
            r = pooled.loc[i]
            records.append({
                "paper_id": f"iclr{year}_{r.id}",
                "openreview_id": r.id,
                "year": int(r.year),
                "title": r.title,
                "abstract": r.abstract,
                "keywords": list(r.keywords) if isinstance(r.keywords, (list, np.ndarray)) else [],
                "decision": str(r.decision),
                "decision_binary": 1 if "accept" in str(r.decision).lower() else 0,
                "human_scores": [float(x) for x in r.scores],
                "human_score_mean": float(np.mean(r.scores)),
                "human_score_var": float(np.var(r.scores)),
                "stratum": stratum,
            })
        print(f"    {stratum}: sampled {n_take}")

    dataset = {
        "meta": {
            "year": year,
            "n_per_stratum": n_per_stratum,
            "seed": seed,
            "source": "berenslab/iclr-dataset iclr26v1.parquet",
            "strata_definition": {
                "clear_accept": "Oral/Spotlight, or Poster with mean>=5.5 and spread<3",
                "borderline": "score-decision conflict or spread>=3",
                "clear_reject": "Reject with mean<=4.0 and spread<3",
            },
        },
        "papers": records,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
    print(f"  → saved {len(records)} papers to {out_path}")
    return dataset


def load_review_papers(path: str = None, year: int = 2024) -> list:
    """供实验脚本调用的加载接口。"""
    path = path or os.path.join(OUT_DIR, f"review_papers_{year}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["papers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n_per_stratum", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--subset_from", type=str, default=None,
                    help="从已抽样文件再抽子集（取每层前 n 篇，确定性，保证是原集子集）")
    args = ap.parse_args()

    out = args.out or os.path.join(OUT_DIR, f"review_papers_{args.year}.json")
    if args.subset_from:
        with open(args.subset_from, "r", encoding="utf-8") as f:
            src = json.load(f)["papers"]
        by_stratum = {}
        for p in src:
            by_stratum.setdefault(p["stratum"], []).append(p)
        sub = []
        for stratum, lst in sorted(by_stratum.items()):
            sub.extend(lst[:args.n_per_stratum])
            print(f"  {stratum}: {min(len(lst), args.n_per_stratum)}/{len(lst)}")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"papers": sub}, f, ensure_ascii=False, indent=1)
        print(f"  → saved subset {len(sub)} papers to {out}")
        return
    build_review_dataset(args.year, args.n_per_stratum, args.seed, out)


if __name__ == "__main__":
    main()
