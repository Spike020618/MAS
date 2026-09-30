"""
Study 2b 受调控 vs 无调控 debate 配对分析（GLM 臂，150 篇同子集）

两层分析：
1. 全板指标对比（复用 analyze_cross_model.analyze_model）
2. 干预触发子集：minority 命运的直接因果检验
   （受调控臂中 intervention_log 非空的论文 vs 同批论文的无调控臂）

Usage:
  python experiments/analyze_regulated.py            # V3 主战场（300 篇）
  python experiments/analyze_regulated.py --unreg ...debate_glm_sub150_2024.json \
      --reg ...debate_regulated_glm_sub150_2024.json --out ...regulated_glm.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_cross_model import analyze_model, _load_papers

UNREG_PATH = "experiments/results/review/panel/debate_2024.json"
REG_PATH = "experiments/results/review/panel/debate_regulated_v3_full300_2024.json"
OUT_PATH = "experiments/results/review/regulated_v3_analysis.json"


def minority_fate(papers: dict, ids: list) -> dict:
    """指定论文子集上，r1 少数派在最终轮的分布。"""
    to_acc = to_rej = persist = 0
    for pid in ids:
        p = papers.get(pid)
        if p is None:
            continue
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
    n = to_acc + to_rej + persist
    return {
        "n_minority": n, "to_accept": to_acc, "to_reject": to_rej,
        "persist": persist,
        "persist_rate": persist / n if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unreg", default=UNREG_PATH)
    ap.add_argument("--reg", default=REG_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    unreg = _load_papers(args.unreg)
    reg = _load_papers(args.reg)
    common = sorted(set(unreg) & set(reg))
    unreg = {k: unreg[k] for k in common}
    reg = {k: reg[k] for k in common}
    print(f"配对论文: {len(common)}")

    fired = [pid for pid in common if reg[pid].get("intervention_log")]
    print(f"干预触发论文: {len(fired)}")

    results = {
        "n_papers": len(common),
        "n_intervened": len(fired),
        "full_panel": {
            "unregulated": analyze_model(unreg),
            "regulated": analyze_model(reg),
        },
        "intervened_subset": {
            "unregulated_matched": analyze_model({k: unreg[k] for k in fired}),
            "regulated": analyze_model({k: reg[k] for k in fired}),
        },
        "minority_fate_intervened": {
            "unregulated_matched": minority_fate(unreg, fired),
            "regulated": minority_fate(reg, fired),
        },
        # 对照：未触发干预的论文两臂应无差异（安慰剂检验）
        "minority_fate_non_intervened": {
            "unregulated": minority_fate(
                unreg, [k for k in common if k not in fired]),
            "regulated": minority_fate(
                reg, [k for k in common if k not in fired]),
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fp = results["full_panel"]
    print("\n===== 全板指标（GLM, n=150）=====")
    print(f"{'指标':<24s}{'无调控':>12s}{'受调控':>12s}")
    for label, key, fmt in [
            ("少数派消灭率", "minority", "{:.1%}"),
            ("方差坍缩", "variance_collapse", "{:.1%}"),
            ("均分漂移", "rating_shift", "{:+.2f}"),
            ("决策一致率 r3", "decision_acc_r3", "{:.1%}"),
            ("骑墙率", "hedge_rate", "{:.1%}")]:
        if key == "minority":
            v1 = fp["unregulated"]["minority"]["elimination_rate"]
            v2 = fp["regulated"]["minority"]["elimination_rate"]
        else:
            v1, v2 = fp["unregulated"][key], fp["regulated"][key]
        s1 = fmt.format(v1) if v1 is not None else "—"
        s2 = fmt.format(v2) if v2 is not None else "—"
        print(f"{label:<24s}{s1:>12s}{s2:>12s}")

    mf = results["minority_fate_intervened"]
    print(f"\n===== 干预触发子集少数派命运（n={len(fired)} 篇）=====")
    for arm, d in [("无调控(同批)", mf["unregulated_matched"]),
                   ("受调控", mf["regulated"])]:
        pr = f"{d['persist_rate']:.0%}" if d["persist_rate"] is not None else "—"
        print(f"  {arm:<12s} 少数派 {d['n_minority']} 人: "
              f"坚持 {d['persist']} ({pr}), 投accept {d['to_accept']}, "
              f"投reject {d['to_reject']}")

    pn = results["minority_fate_non_intervened"]
    print("\n===== 安慰剂对照（未触发论文，两臂应相似）=====")
    for arm, d in [("无调控", pn["unregulated"]), ("受调控", pn["regulated"])]:
        pr = f"{d['persist_rate']:.0%}" if d["persist_rate"] is not None else "—"
        print(f"  {arm:<12s} 少数派 {d['n_minority']} 人: 坚持率 {pr}")

    print(f"\n→ saved to {args.out}")


if __name__ == "__main__":
    main()
