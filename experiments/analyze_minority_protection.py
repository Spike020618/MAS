"""
机制级少数派保护：收益上限解剖（离线，零 LLM 成本）

问题：debate r1 不一致的 90 个面板中，被消灭的少数派（78%）里有多少
本来是对的？若机制级保护少数派（聚合时保留其立场），决策指标能提升多少？

口径：stance 映射 accept/weak_accept→1, reject/weak_reject→0（骑墙除外）。
面板决策 = 多数侧 stance；保护策略 = 对"高置信少数派"保留其立场。

Usage:
  python experiments/analyze_minority_protection.py
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np

DEBATE_PATH = "results/review/panel/debate_2024.json"
STRATA = ["clear_accept", "borderline", "clear_reject"]

ANCHOR_RE = re.compile(
    r"(?i)(section\s*\d|table\s*\d|figure\s*\d|fig\.\s*\d|equation|eq\.\s*\d|"
    r"theorem|lemma|appendix|algorithm\s*\d|line\s*\d|\[\d+\])")

STANCE_MAP = {"accept": 1, "weak_accept": 1, "reject": 0, "weak_reject": 0}


def spec_of(review: dict) -> float:
    e_list = review["aeic"].get("E") or []
    return (sum(1 for e in e_list if ANCHOR_RE.search(e)) / len(e_list)
            if e_list else 0.0)


def to_bin(stance: str):
    return STANCE_MAP.get((stance or "").lower())


def main():
    with open(DEBATE_PATH, encoding="utf-8") as f:
        papers = json.load(f)["papers"]

    rows = []
    for p in papers:
        r1 = p["rounds"][0]["reviews"]
        r3 = p["rounds"][-1]["reviews"]
        s1 = [to_bin(r["stance"]) for r in r1]
        s3 = [to_bin(r["stance"]) for r in r3]
        valid1 = [s for s in s1 if s is not None]
        if len(valid1) < 4:  # 骑墙过多，跳过
            continue
        n_acc = sum(valid1)
        n_rej = len(valid1) - n_acc
        if n_acc == 0 or n_rej == 0:
            unanimous = True
            min_stance, maj_stance, n_min = None, valid1[0], 0
        else:
            unanimous = False
            if n_acc <= n_rej:
                min_stance, maj_stance, n_min = 1, 0, n_acc
            else:
                min_stance, maj_stance, n_min = 0, 1, n_rej
        # 少数派特征（仅不一致时）
        min_spec = maj_spec = np.nan
        min_rating = maj_rating = np.nan
        if not unanimous:
            min_idx = [i for i, s in enumerate(s1) if s == min_stance]
            maj_idx = [i for i, s in enumerate(s1) if s == maj_stance]
            min_spec = float(np.mean([spec_of(r1[i]) for i in min_idx]))
            maj_spec = float(np.mean([spec_of(r1[i]) for i in maj_idx]))
            mr = [r1[i]["rating"] for i in min_idx if r1[i]["rating"] is not None]
            jr = [r1[i]["rating"] for i in maj_idx if r1[i]["rating"] is not None]
            min_rating = float(np.mean(mr)) if mr else np.nan
            maj_rating = float(np.mean(jr)) if jr else np.nan
        # r3 面板决策（多数）
        valid3 = [s for s in s3 if s is not None]
        dec3 = int(np.mean(valid3) > 0.5) if valid3 else None
        rows.append({
            "paper_id": p["paper_id"], "stratum": p["stratum"],
            "y": int(p["decision_binary"]),
            "unanimous_r1": unanimous,
            "min_stance": min_stance, "maj_stance": maj_stance,
            "n_min": n_min,
            "min_spec": min_spec, "maj_spec": maj_spec,
            "min_rating": min_rating, "maj_rating": maj_rating,
            "dec_r1": int(np.mean(valid1) > 0.5),
            "dec_r3": dec3,
        })

    nonun = [r for r in rows if not r["unanimous_r1"]]
    print(f"面板 {len(rows)}，r1 不一致 {len(nonun)}"
          f"（一致 {len(rows)-len(nonun)} = {(len(rows)-len(nonun))/len(rows):.1%}）")

    # ── 1. 少数派 vs 多数派的立场正确率 ─────────────────────
    min_correct = sum(1 for r in nonun if r["min_stance"] == r["y"])
    maj_correct = sum(1 for r in nonun if r["maj_stance"] == r["y"])
    print(f"\n[少数派立场正确率] {min_correct}/{len(nonun)} = "
          f"{min_correct/len(nonun):.1%}")
    print(f"[多数派立场正确率] {maj_correct}/{len(nonun)} = "
          f"{maj_correct/len(nonun):.1%}")
    print("（两者互补：min_correct + maj_correct 应 = 面板数）")

    # 按层分解
    print("\n按层分解（少数派正确率）:")
    for s in STRATA:
        sub = [r for r in nonun if r["stratum"] == s]
        if not sub:
            continue
        mc = sum(1 for r in sub if r["min_stance"] == r["y"])
        # 少数派方向分布
        dir_min = Counter(r["min_stance"] for r in sub)
        print(f"  {s:<14s} n={len(sub):>3d}  少数派正确 {mc/len(sub):.1%}  "
              f"少数派方向 accept={dir_min.get(1,0)} reject={dir_min.get(0,0)}")

    # ── 2. 决策口径的指标对比（全集 300）─────────────────────
    y = np.array([r["y"] for r in rows])
    dec_r1 = np.array([r["dec_r1"] for r in rows])
    dec_r3 = np.array([r["dec_r3"] if r["dec_r3"] is not None else r["dec_r1"]
                       for r in rows])
    print(f"\n[全集指标] r1 多数 {(dec_r1==y).mean():.1%} | "
          f"r3 多数 {(dec_r3==y).mean():.1%}")

    # oracle 保护：不一致面板总选对的一侧
    dec_oracle = dec_r1.copy()
    for j, r in enumerate(rows):
        if not r["unanimous_r1"]:
            dec_oracle[j] = r["min_stance"] if r["min_stance"] == r["y"] \
                else r["maj_stance"]
    print(f"[oracle 保护上限] {(dec_oracle==y).mean():.1%}"
          f"（不一致面板总能选对侧）")

    # 朴素保护：不一致面板一律听少数派
    dec_allmin = np.array([r["min_stance"] if not r["unanimous_r1"]
                           else r["dec_r1"] for r in rows])
    print(f"[一律听少数派]   {(dec_allmin==y).mean():.1%}")

    # ── 3. 可学习规则的空间：什么少数派该保？────────────────
    # 特征分组看正确率：少数派人数、方向、特异性差
    print("\n[少数派特征 × 正确率]（在不一致面板上）")
    for label, key, bins in [
        ("少数派人数", "n_min", [1, 2, 3]),
        ("少数派方向", "min_stance", [0, 1]),
    ]:
        for b in bins:
            sub = [r for r in nonun if r[key] == b]
            if len(sub) >= 5:
                mc = np.mean([r["min_stance"] == r["y"] for r in sub])
                print(f"  {label}={b}: n={len(sub):>3d}  少数派正确 {mc:.1%}")
    # 特异性差
    d = [r for r in nonun
         if not (np.isnan(r["min_spec"]) or np.isnan(r["maj_spec"]))]
    hi = [r for r in d if r["min_spec"] > r["maj_spec"] + 0.1]
    lo = [r for r in d if r["min_spec"] < r["maj_spec"] - 0.1]
    eq = [r for r in d if abs(r["min_spec"] - r["maj_spec"]) <= 0.1]
    for name, sub in [("少数派特异性更高", hi), ("相当", eq), ("更低", lo)]:
        if len(sub) >= 5:
            mc = np.mean([r["min_stance"] == r["y"] for r in sub])
            print(f"  特异性 {name}: n={len(sub):>3d}  少数派正确 {mc:.1%}")

    # 评分差
    d2 = [r for r in nonun
          if not (np.isnan(r["min_rating"]) or np.isnan(r["maj_rating"]))]
    gap = [r["min_rating"] - r["maj_rating"] for r in d2]
    print(f"  评分差(min-maj) 分布: mean={np.mean(gap):+.2f} "
          f"min={np.min(gap):+.2f} max={np.max(gap):+.2f}")

    # ── 4. 简单规则候选：只保"方向=accept 的少数派"───────────
    dec_rule = dec_r1.copy()
    n_protect = 0
    for j, r in enumerate(rows):
        if not r["unanimous_r1"] and r["min_stance"] == 1:
            dec_rule[j] = 1
            n_protect += 1
    print(f"\n[规则A: 只保 accept 向少数派] 保护 {n_protect} 篇 → "
          f"{(dec_rule==y).mean():.1%}")
    dec_rule2 = dec_r1.copy()
    n_protect2 = 0
    for j, r in enumerate(rows):
        if not r["unanimous_r1"] and r["min_stance"] == 1 \
                and r["min_spec"] >= r["maj_spec"]:
            dec_rule2[j] = 1
            n_protect2 += 1
    print(f"[规则B: A+少数派特异性不低] 保护 {n_protect2} 篇 → "
          f"{(dec_rule2==y).mean():.1%}")

    # 分层指标（规则B vs r1/r3）
    print("\n分层一致率（全集 300）:")
    print(f"{'方法':<22s}{'总体':>8s}{'clear_acc':>10s}{'borderline':>12s}"
          f"{'clear_rej':>11s}")
    for name, dec in [("r1 多数", dec_r1), ("r3 多数(debate后)", dec_r3),
                      ("规则B 保护", dec_rule2), ("oracle", dec_oracle)]:
        line = f"{name:<22s}{(dec==y).mean():>8.1%}"
        for s in STRATA:
            m = np.array([r["stratum"] == s for r in rows])
            line += f"{(dec[m]==y[m]).mean():>10.1%}" if s == "clear_accept" \
                else f"{(dec[m]==y[m]).mean():>12.1%}" if s == "borderline" \
                else f"{(dec[m]==y[m]).mean():>11.1%}"
        print(line)


if __name__ == "__main__":
    main()
