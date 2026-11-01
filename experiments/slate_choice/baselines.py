"""Ordinary point, diversity and uncertainty-aware slate baselines, not a new game."""
from __future__ import annotations

import hashlib
import numpy as np


def ridge(x, y, penalty):
    # Explicit contractions avoid spurious floating-point status warnings from
    # the platform BLAS matmul path; do not silence numeric warnings globally.
    gram = np.einsum("ni,nj->ij", x, x, optimize=False)
    rhs = np.einsum("ni,n->i", x, y, optimize=False)
    return np.linalg.solve(gram + penalty * np.eye(x.shape[1]), rhs)


def select_slate(values, point, genres, k, method, mmr_lambda=0.7):
    """values: bootstrap models x candidates, all sharing the same observed input.

    SMR uses the same finite candidate set and bootstrap hypotheses as the other
    set objectives. Greedy construction has no global minimax certificate.
    """
    n = len(point)
    k = min(k, n)
    if k < 0 or not np.isfinite(values).all() or not np.isfinite(point).all():
        raise ValueError("invalid slate input")
    if values.ndim != 2 or values.shape[1] != n or len(genres) != n:
        raise ValueError("shape mismatch")
    if method == "point_topk":
        return sorted(range(n), key=lambda j: (-point[j], j))[:k]
    if method not in {"mmr", "bootstrap_expected_max", "greedy_minimax_regret"}:
        raise ValueError("unknown baseline")
    chosen = []
    current = np.zeros(values.shape[0])
    best = values.max(axis=1) if n else np.zeros(values.shape[0])
    norms = np.linalg.norm(genres, axis=1)
    normalized = genres / np.maximum(norms[:, None], 1e-12)
    redundancy = np.zeros(n)
    for _ in range(k):
        if method == "mmr":
            scores = mmr_lambda * point / 5 - (1 - mmr_lambda) * redundancy
        else:
            next_values = np.maximum(current[:, None], values)
            scores = (next_values.mean(axis=0) if method == "bootstrap_expected_max"
                      else -(best[:, None] - next_values).max(axis=0))
        eligible = [j for j in range(n) if j not in chosen]
        # Deterministic tie-breaking, then common point-order for display.
        picked = max(eligible, key=lambda j: (round(float(scores[j]), 12), float(point[j]), -j))
        chosen.append(picked)
        current = np.maximum(current, values[:, picked])
        redundancy = np.maximum(redundancy, np.einsum("ij,j->i", normalized, normalized[picked], optimize=False))
    return sorted(chosen, key=lambda j: (-point[j], j))


class Recommender:
    def __init__(self, items, cfg):
        self.cfg = cfg
        self.ids = sorted(items)
        self.positions = {mid: i for i, mid in enumerate(self.ids)}
        self.items = items
        self.vocabulary = sorted({g for item in items.values() for g in item["genres"]})
        self.genre_features = np.array([[float(g in items[mid]["genres"]) for g in self.vocabulary]
                                        for mid in self.ids])
        normalized = self.genre_features / np.maximum(1, np.linalg.norm(self.genre_features, axis=1))[:, None]
        self.x = np.column_stack((np.ones(len(items)), normalized))
        self.base = np.array([items[mid]["prior"] for mid in self.ids])
        self.prior_order = sorted(range(len(self.ids)), key=lambda j: (-self.base[j], self.ids[j]))
        self.genre_orders = {g: [j for j in self.prior_order if self.genre_features[j, g]]
                             for g in range(len(self.vocabulary))}

    def predict(self, inp, history_budget):
        # No labels parameter. Only the allowed prefix/history enters this path.
        cfg = self.cfg
        history = inp["history"][-history_budget:]
        if not history or any(r["timestamp"] >= inp["cutoff"] for r in history):
            raise ValueError("empty or post-cutoff profile")
        if any(self.items[mid]["year"] > inp["hard_constraints"]["max_year"] for mid in self.ids):
            raise ValueError("catalog violates year constraint")
        positions = [self.positions[r["movie_id"]] for r in history]
        x = self.x[positions]
        residual = np.array([r["rating"] for r in history]) - self.base[positions]
        beta = ridge(x, residual, cfg["ridge_lambda"])
        point_all = np.clip(self.base + np.einsum("ij,j->i", self.x, beta, optimize=False), 0.5, 5)
        seen = set(inp["seen_movie_ids"])
        available = [j for j, mid in enumerate(self.ids) if mid not in seen]
        selected = set(sorted(available, key=lambda j: (-point_all[j], self.ids[j]))[:cfg["candidate_point_items"]])
        selected.update([j for j in self.prior_order if self.ids[j] not in seen][:cfg["candidate_prior_items"]])
        for order in self.genre_orders.values():
            selected.update([j for j in order if self.ids[j] not in seen][:cfg["candidate_items_per_genre"]])
        pool = sorted(selected, key=lambda j: self.ids[j])
        seed = int(hashlib.sha256(f"{cfg['seed']}:{inp['user_id']}:{history_budget}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        betas = []
        for _ in range(cfg["bootstrap_models"]):
            sampled = rng.integers(0, len(history), size=len(history))
            betas.append(ridge(x[sampled], residual[sampled], cfg["ridge_lambda"]))
        values = np.clip(self.base[pool][None, :] +
                         np.einsum("md,nd->mn", np.array(betas), self.x[pool], optimize=False), 0.5, 5)
        point = point_all[pool]
        predictions = {}
        for method in cfg["methods"]:
            picked = select_slate(values, point, self.genre_features[pool], cfg["top_k"], method,
                                  cfg["mmr_lambda"])
            choices = [self.ids[pool[j]] for j in picked]
            best = values.max(axis=1) if len(pool) else np.zeros(len(values))
            chosen_value = values[:, picked].max(axis=1) if picked else np.zeros(len(values))
            predictions[method] = {"movie_ids": choices,
                                   "finite_hypothesis_max_regret": float((best - chosen_value).max()),
                                   "finite_hypothesis_expected_max": float(chosen_value.mean())}
        return {"user_id": inp["user_id"], "history_budget": history_budget,
                "candidate_ids": [self.ids[j] for j in pool], "predictions": predictions}


def score_prediction(prediction, label, catalog_ids):
    future = {r["movie_id"]: r["rating"] for r in label["future_ratings"]}
    relevant = {mid for mid, rating in future.items() if rating >= 4.0}
    pool = set(prediction["candidate_ids"])
    eligible = relevant & set(catalog_ids)
    result = {"future_positive_count": len(relevant), "eligible_positive_count": len(eligible),
              "candidate_positive_count": len(relevant & pool),
              "candidate_positive_coverage": len(relevant & pool) / len(relevant) if relevant else 0.0,
              "methods": {}}
    for method, output in prediction["predictions"].items():
        chosen = output["movie_ids"]
        if len(chosen) != len(set(chosen)) or not set(chosen) <= pool:
            raise ValueError("illegal recommendation")
        hits = [mid in relevant for mid in chosen]
        dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal = sum(1 / np.log2(rank + 2) for rank in range(min(5, len(relevant))))
        pairs = []
        for i, left in enumerate(chosen):
            for right in chosen[i + 1:]:
                a, b = set(catalog_ids[left]["genres"]), set(catalog_ids[right]["genres"])
                pairs.append(1 - len(a & b) / len(a | b) if a | b else 0.0)
        result["methods"][method] = {"hit_at_5": float(any(hits)),
                                     "recall_at_5": sum(hits) / len(relevant) if relevant else 0.0,
                                     "ndcg_at_5": float(dcg / ideal) if ideal else 0.0,
                                     "genre_distance": float(np.mean(pairs)) if pairs else 0.0,
                                     "observed_rating_coverage": sum(mid in future for mid in chosen) / max(1, len(chosen)),
                                     "observed_ratings": [future[mid] for mid in chosen if mid in future]}
    return result
