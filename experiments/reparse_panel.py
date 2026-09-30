"""
重解析工具：用当前最新解析器重新推导面板缓存中的 stance/rating。
用于解析器升级（骑墙回退、截断JSON救援）后修复存量缓存，避免重新采集。

Usage:
  python experiments/reparse_panel.py experiments/results/review/panel/debate_glm_sub150_2024.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.review_prompts import parse_stance_rating


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="面板缓存 json 路径")
    args = ap.parse_args()

    for path in args.paths:
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        tot = fixed = still_none = 0
        for p in cache["papers"]:
            for rnd in p.get("rounds", []):
                for r in rnd["reviews"]:
                    tot += 1
                    old = (r.get("stance"), r.get("rating"))
                    new = parse_stance_rating(r["aeic"].get("C", ""))
                    if new != old:
                        fixed += 1
                        r["stance"], r["rating"] = new
                    if r["stance"] is None or r["rating"] is None:
                        still_none += 1
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        print(f"{path}: {tot} reviews, 修正 {fixed}, 仍缺失 {still_none} "
              f"({still_none / tot:.1%})")


if __name__ == "__main__":
    main()
