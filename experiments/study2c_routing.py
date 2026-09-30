"""
Study 2c —— 按论文路由讨论协议（治"讨论一刀切"）

动机：Study 1 显示 70% 面板 round1 已一致——对这些论文跑 3 轮讨论纯浪费。
路由器用 round1 即可得的前瞻特征，为每篇论文选择协议：
  动作空间：
    theta_indep  θ*校准聚合（独立评审，成本 5 次调用）
    debate       无调控 3 轮讨论（成本 15）
    debate_reg   受调控 3 轮讨论（Study 2b，成本 15）
    meta         meta-review 综合（成本 6：5 spoke + 1 meta）
  标签：该论文在该协议下决策是否与官方一致（0/1）
  路由器：每动作一个 LogisticRegression 估计 P(correct|x)，argmax 选协议

对照：各固定协议 / 规则基线 / 随机 / oracle（上界）。
H2c：路由一致率 ≥ 最优固定协议 +2pp，且平均调用成本较全讨论降 ≥40%。

train/test 切分与 Study 2a 完全一致（seed=42 分层各半）。

Usage:
  python experiments/study2c_routing.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.study2a_calibrated_aggregation import (
    load_panel, stratified_split, panel_features, theta_aggregate, STRATA,
)

PARALLEL_PATH = "experiments/results/review/panel/parallel_2024.json"
DEBATE_PATH = "experiments/results/review/panel/debate_2024.json"
REG_PATH = "experiments/results/review/panel/debate_regulated_v3_full300_2024.json"
STAR_PATH = "experiments/results/review/panel/star_2024.json"
THETA_PATH = "experiments/results/review/study2a_theta_star.json"
OUT_PATH = "experiments/results/review/study2c_routing.json"

ACTIONS = ["theta_indep", "debate", "debate_reg", "meta"]
COST = {"theta_indep": 5, "debate": 15, "debate_reg": 15, "meta": 6}


def _majority_decision(reviews: list):
    votes = [r["stance"] for r in reviews if r.get("stance")]
    if not votes:
        return None
    n_acc = votes.count("accept")
    if n_acc == len(votes) - n_acc:
        return None  # 平局视为不可用
    return int(n_acc > len(votes) / 2)


def build_labels(rows: list) -> dict:
    """每篇论文 × 每动作 → 决策正确 0/1（None=该动作数据不可用）。"""
    with open(DEBATE_PATH, "r", encoding="utf-8") as f:
        debate = {p["paper_id"]: p for p in json.load(f)["papers"]}
    reg = {}
    if os.path.exists(REG_PATH):
        with open(REG_PATH, "r", encoding="utf-8") as f:
            reg = {p["paper_id"]: p for p in json.load(f)["papers"]}
    with open(STAR_PATH, "r", encoding="utf-8") as f:
        star = {p["paper_id"]: p for p in json.load(f)["papers"]}

    # θ* 聚合决策（复用 2a 学到的参数）
    with open(THETA_PATH, "r", encoding="utf-8") as f:
        th = json.load(f)["theta_star"]
    theta = np.array(th["w"] + [th["gamma1"], th["gamma2"],
                                th["tau"], th["threshold"]])
    R, P, S = panel_features(rows)
    _, th_dec = theta_aggregate(theta, R, P, S)

    labels = {}
    for j, row in enumerate(rows):
        pid, y = row["paper_id"], row["decision"]
        lab = {"theta_indep": int(th_dec[j] == y)}
        p = debate.get(pid)
        d = _majority_decision(p["rounds"][-1]["reviews"]) if p else None
        lab["debate"] = int(d == y) if d is not None else None
        p = reg.get(pid)
        d = _majority_decision(p["rounds"][-1]["reviews"]) if p else None
        lab["debate_reg"] = int(d == y) if d is not None else None
        p = star.get(pid)
        st = (p["meta_review"].get("stance") if p and p.get("meta_review")
              else None)
        lab["meta"] = int((st == "accept") == bool(y)) if st in ("accept", "reject") else None
        labels[pid] = lab
    return labels


def build_features(rows: list) -> np.ndarray:
    """round1 前瞻特征：立场熵、评分std/均值、证据具体性、正负证据比、I长度。"""
    X = []
    for row in rows:
        st = [s for s in row["stances"] if s]
        p_acc = st.count("accept") / len(st) if st else 0.5
        p_acc = min(max(p_acc, 1e-9), 1 - 1e-9)
        entropy = -(p_acc * np.log2(p_acc) + (1 - p_acc) * np.log2(1 - p_acc))
        rt = [sg["rating"] for sg in row["sigs"] if sg["rating"] is not None]
        spec = np.mean([sg["specificity"] for sg in row["sigs"]])
        posr = np.mean([sg["pos_ratio"] for sg in row["sigs"]])
        ilen = np.log1p(np.mean([sg["i_len"] for sg in row["sigs"]]))
        X.append([entropy,
                  float(np.std(rt, ddof=1)) if len(rt) >= 2 else 0.0,
                  float(np.mean(rt)) if rt else 5.0,
                  float(spec), float(posr), float(ilen)])
    return np.array(X)


def main():
    rows = load_panel()
    y = np.array([r["decision"] for r in rows])
    tr, te = stratified_split(rows, 42)
    X = build_features(rows)
    labels = build_labels(rows)
    pids = [r["paper_id"] for r in rows]
    print(f"train {len(tr)} / test {len(te)}；动作空间 {ACTIONS}")

    # 标签矩阵（缺失动作的论文在该动作上剔除）
    Y = np.full((len(rows), len(ACTIONS)), np.nan)
    for j, pid in enumerate(pids):
        for a, act in enumerate(ACTIONS):
            v = labels[pid][act]
            if v is not None:
                Y[j, a] = v
    avail = ~np.isnan(Y)

    # ── 训练：每动作 P(correct|x) ─────────────────────────
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)
    models = {}
    for a, act in enumerate(ACTIONS):
        m = avail[tr, a]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xs[tr][m], Y[tr][m, a])
        models[act] = clf
    P = np.column_stack([models[a].predict_proba(Xs)[:, 1]
                         for a in ACTIONS])  # (N, n_actions)

    # ── 路由策略 ──────────────────────────────────────────
    def route_argmax(j):
        """P(correct) 最大；平局/缺失回退更便宜动作。"""
        order = np.argsort(-P[j])
        for a in order:
            if avail[j, a]:
                return a
        return ACTIONS.index("theta_indep")

    def route_eps(j, eps=0.05):
        """成本敏感：比最优便宜且 P 损失 < eps 的动作优先。"""
        best = route_argmax(j)
        for a in np.argsort([COST[x] for x in ACTIONS]):
            if avail[j, a] and P[j, best] - P[j, a] < eps:
                return int(a)
        return best

    def route_rule(j):
        """规则基线：r1 一致 → 独立聚合；分歧 → train 上一致率最高的讨论协议。"""
        ent = X[j, 0]
        if ent < 1e-9:
            return ACTIONS.index("theta_indep")
        # train 上 debate / debate_reg / meta 的固定一致率
        rates = {a: np.nanmean(Y[tr, ACTIONS.index(a)])
                 for a in ["debate", "debate_reg", "meta"]}
        best_disc = max(rates, key=rates.get)
        a = ACTIONS.index(best_disc)
        return a if avail[j, a] else ACTIONS.index("theta_indep")

    # ── 评估 ──────────────────────────────────────────────
    def eval_policy(route_fn, idx):
        hits, costs, used = [], [], {a: 0 for a in ACTIONS}
        for j in idx:
            a = route_fn(j)
            act = ACTIONS[a]
            used[act] += 1
            costs.append(COST[act])
            hits.append(Y[j, a])
        hits = np.array(hits)
        return {"agreement": float(np.nanmean(hits)),
                "avg_cost": float(np.mean(costs)), "action_mix": used}

    def eval_fixed(act, idx):
        a = ACTIONS.index(act)
        m = avail[idx, a]
        return {"agreement": float(np.nanmean(Y[idx, a][m])),
                "avg_cost": float(COST[act]), "n": int(m.sum())}

    rng = np.random.RandomState(0)
    rand_hits, rand_costs = [], []
    te_avail = [j for j in te if avail[j].any()]
    for _ in range(200):
        h, c = [], []
        for j in te_avail:
            opts = np.where(avail[j])[0]
            a = rng.choice(opts)
            h.append(Y[j, a])
            c.append(COST[ACTIONS[a]])
        rand_hits.append(np.mean(h))
        rand_costs.append(np.mean(c))

    def oracle(j):
        opts = [a for a in range(len(ACTIONS)) if avail[j, a] and Y[j, a] == 1]
        if not opts:
            return route_argmax(j)
        return min(opts, key=lambda a: COST[ACTIONS[a]])

    test_res = {
        "router_argmax": eval_policy(route_argmax, te),
        "router_cost_sensitive": eval_policy(route_eps, te),
        "rule_baseline": eval_policy(route_rule, te),
        "oracle": eval_policy(oracle, te),
        "random": {"agreement": float(np.mean(rand_hits)),
                   "agreement_std": float(np.std(rand_hits)),
                   "avg_cost": float(np.mean(rand_costs))},
    }
    for act in ACTIONS:
        test_res[f"fixed_{act}"] = eval_fixed(act, te)

    # 最优固定协议（test 上一致率最高）
    fixed = {a: test_res[f"fixed_{a}"]["agreement"] for a in ACTIONS}
    best_fixed = max(fixed, key=fixed.get)
    # McNemar：router_argmax vs 最优固定
    from scipy.stats import binomtest
    a_bf = ACTIONS.index(best_fixed)
    pair = [(j, route_argmax(j)) for j in te if avail[j, a_bf]]
    b = sum(1 for j, ra in pair if Y[j, ra] == 1 and Y[j, a_bf] == 0)
    c = sum(1 for j, ra in pair if Y[j, ra] == 0 and Y[j, a_bf] == 1)
    p_mc = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) else 1.0

    results = {
        "config": {"seed": 42, "n_train": len(tr), "n_test": len(te),
                   "actions": ACTIONS, "cost": COST},
        "feature_names": ["stance_entropy", "rating_std", "rating_mean",
                          "specificity", "pos_ratio", "log_i_len"],
        "test": test_res,
        "best_fixed": best_fixed,
        "mcnemar_router_vs_best_fixed": {"b": b, "c": c, "pvalue": p_mc},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 报告 ───────────────────────────────────────────────
    print("\n===== test 集（n=150）路由 vs 固定协议 =====")
    print(f"{'策略':<22s}{'一致率':>9s}{'平均成本':>9s}")
    for name in ["fixed_theta_indep", "fixed_debate", "fixed_debate_reg",
                 "fixed_meta", "rule_baseline", "random",
                 "router_argmax", "router_cost_sensitive", "oracle"]:
        d = test_res[name]
        print(f"{name:<22s}{d['agreement']:>9.1%}{d['avg_cost']:>9.1f}")
    ra = test_res["router_argmax"]
    print(f"\nH2c 检验：路由 {ra['agreement']:.1%} vs 最优固定 "
          f"{best_fixed} {fixed[best_fixed]:.1%}"
          f"（{'达标' if ra['agreement'] >= fixed[best_fixed] + 0.02 else '未达 +2pp'}）")
    print(f"成本 vs 全讨论(15)：{ra['avg_cost']:.1f} "
          f"（降 {1 - ra['avg_cost'] / 15:.0%}，目标 ≥40%）")
    print(f"路由动作分布: {ra['action_mix']}")
    print(f"McNemar vs 最优固定: b={b}, c={c}, p={p_mc:.4f}")
    print(f"→ saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
