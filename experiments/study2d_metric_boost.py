"""
Study 2d —— 指标提升：r1 面板信号融合（少数派保护 + 校准的统一形式）

动机：2a 的 θ*（64.7%）输给均值阈值（66.7%）；少数派解剖显示 oracle
保护上限 77%。r1 面板的立场分布 + 评分 + 证据特征蕴含比"均值"更多的
决策信息。用 LogisticRegression 在 train 上学习融合规则，test 上评估。

V3：300 篇 debate r1（独立评审轮），train/test 150/150（seed=42 分层，
    与 study2a 同口径可比）。
R1：150 篇 debate r1 轮，75/75 分层切分。

Usage:
  python experiments/study2d_metric_boost.py
"""
from __future__ import annotations

import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

V3_PATH = "results/review/panel/debate_2024.json"
R1_PATH = "results/review/panel/debate_r1_sub150_2024.json"
OUT_PATH = "results/review/study2d_metric_boost.json"
STRATA = ["clear_accept", "borderline", "clear_reject"]
STANCE_MAP = {"accept": 1, "weak_accept": 1, "reject": 0, "weak_reject": 0}

ANCHOR_RE = re.compile(
    r"(?i)(section\s*\d|table\s*\d|figure\s*\d|fig\.\s*\d|equation|eq\.\s*\d|"
    r"theorem|lemma|appendix|algorithm\s*\d|line\s*\d|\[\d+\])")


def spec_of(review: dict) -> float:
    e_list = review["aeic"].get("E") or []
    return (sum(1 for e in e_list if ANCHOR_RE.search(e)) / len(e_list)
            if e_list else 0.0)


def pos_ratio_of(review: dict) -> float:
    e_list = review["aeic"].get("E") or []
    n_pos = sum(1 for e in e_list if e.strip().startswith("(+)"))
    n_neg = sum(1 for e in e_list if e.strip().startswith("(-)"))
    return n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0.5


def paper_features(p: dict) -> tuple:
    """r1 轮特征向量 + 辅助量。"""
    r1 = p["rounds"][0]["reviews"]
    stances = [STANCE_MAP.get((r["stance"] or "").lower()) for r in r1]
    ratings = [r["rating"] for r in r1]
    valid = [(s, rt, r) for s, rt, r in zip(stances, ratings, r1)
             if s is not None and rt is not None]
    if len(valid) < 4:
        return None, None
    sv = np.array([v[0] for v in valid], dtype=float)
    rv = np.array([v[1] for v in valid], dtype=float)
    n_acc = int(sv.sum())
    n_rej = len(sv) - n_acc
    unanimous = (n_acc == 0 or n_rej == 0)
    specs = np.array([spec_of(v[2]) for v in valid])
    poss = np.array([pos_ratio_of(v[2]) for v in valid])

    if unanimous:
        n_min, min_is_accept, spec_diff, rating_diff = 0, 0.5, 0.0, 0.0
    else:
        min_side = 1 if n_acc <= n_rej else 0
        n_min = min(n_acc, n_rej)
        min_mask = sv == min_side
        min_is_accept = float(min_side)
        spec_diff = float(specs[min_mask].mean() - specs[~min_mask].mean())
        rating_diff = float(rv[min_mask].mean() - rv[~min_mask].mean())

    feat = [rv.mean(), rv.std(), n_acc / len(sv), n_min, min_is_accept,
            spec_diff, rating_diff, poss.mean(), specs.mean()]
    aux = {"dec_mv": int(sv.mean() > 0.5), "mean_rating": float(rv.mean()),
           "unanimous": unanimous}
    return feat, aux


def stratified_split(rows, seed=42):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for s in STRATA:
        idx = [i for i, r in enumerate(rows) if r["stratum"] == s]
        rng.shuffle(idx)
        half = len(idx) // 2
        tr.extend(idx[:half])
        te.extend(idx[half:])
    return sorted(tr), sorted(te)


def eval_dec(dec, y, rows, idx):
    out = {"agreement": float((dec[idx] == y[idx]).mean())}
    for s in STRATA:
        m = np.array([rows[i]["stratum"] == s for i in idx])
        out[f"agree[{s}]"] = float((dec[idx][m] == y[idx][m]).mean())
    return out


def run_arm(path: str, name: str) -> dict:
    with open(path, encoding="utf-8") as f:
        papers = json.load(f)["papers"]
    rows, feats, mv, mean_r = [], [], [], []
    for p in papers:
        feat, aux = paper_features(p)
        if feat is None:
            continue
        rows.append({"stratum": p["stratum"], "y": int(p["decision_binary"])})
        feats.append(feat)
        mv.append(aux["dec_mv"])
        mean_r.append(aux["mean_rating"])
    X = np.array(feats)
    y = np.array([r["y"] for r in rows])
    mv = np.array(mv)
    mean_r = np.array(mean_r)
    tr, te = stratified_split(rows, seed=42)
    print(f"\n===== {name}（{len(rows)} 篇，train {len(tr)} / test {len(te)}）=====")

    # 均值阈值基线（train 网格）
    best_t, best_a = 6.0, -1
    for t in np.arange(4.0, 8.01, 0.1):
        a = ((mean_r[tr] >= t).astype(int) == y[tr]).mean()
        if a > best_a:
            best_a, best_t = a, t
    dec_mt = (mean_r >= best_t).astype(int)

    # LR 融合
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(X[tr]), y[tr])
    dec_lr = clf.predict(scaler.transform(X))
    cv = cross_val_score(LogisticRegression(max_iter=2000),
                         scaler.transform(X[tr]), y[tr],
                         cv=StratifiedKFold(5, shuffle=True, random_state=0))
    # 特征权重（标准化后，可解释）
    coef = dict(zip(["r1_mean", "r1_std", "acc_share", "n_min",
                     "min_is_accept", "spec_diff", "rating_diff",
                     "pos_ratio", "spec_mean"],
                    np.round(clf.coef_[0], 3).tolist()))

    # oracle（r1 不一致面板总选对侧；仅 V3 全集口径参考）
    res = {
        "n": len(rows), "n_train": len(tr), "n_test": len(te),
        "mean_threshold_t": float(best_t),
        "lr_cv_acc": float(cv.mean()),
        "lr_coef": coef,
        "test": {
            "majority_vote": eval_dec(mv, y, rows, te),
            "mean_threshold": eval_dec(dec_mt, y, rows, te),
            "lr_fusion": eval_dec(dec_lr, y, rows, te),
        },
    }
    t = res["test"]
    print(f"{'方法':<16s}{'总体':>8s}{'clear_acc':>10s}{'borderline':>12s}"
          f"{'clear_rej':>11s}")
    for label, key in [("多数投票", "majority_vote"), ("均值阈值", "mean_threshold"),
                       ("LR融合(含少数派)", "lr_fusion")]:
        d = t[key]
        print(f"{label:<16s}{d['agreement']:>8.1%}{d['agree[clear_accept]']:>10.1%}"
              f"{d['agree[borderline]']:>12.1%}{d['agree[clear_reject]']:>11.1%}")
    print(f"LR 5折CV: {cv.mean():.1%} | 均值阈值 t*={best_t:.1f}")
    print(f"LR 系数: {coef}")
    return res


def main():
    results = {"v3": run_arm(V3_PATH, "DeepSeek-V3 (debate r1 = 独立评审)")}
    results["r1"] = run_arm(R1_PATH, "DeepSeek-R1 (debate r1 轮)")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n→ saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
