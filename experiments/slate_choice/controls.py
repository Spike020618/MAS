"""Collaborative representation and finite-hypothesis exact SMR controls."""
from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, vstack

from .baselines import Recommender
from . import baselines


class CFRecommender(Recommender):
    def __init__(self, items, cfg, feature_path):
        super().__init__(items, cfg)
        with np.load(feature_path, allow_pickle=False) as saved:
            if saved["ids"].tolist() != self.ids:
                raise ValueError("CF item alignment mismatch")
            features = saved["features"].copy()
        if not np.isfinite(features).all() or len(features) != len(self.ids):
            raise ValueError("invalid collaborative features")
        self.x = np.column_stack((np.ones(len(self.ids)), features))


def capture_values(model, inp, budget):
    """Observe the matrices actually passed to selection, without changing it."""
    captured = {}
    original = baselines.select_slate

    def inspect(values, point, genres, *args, **kwargs):
        if not captured:
            captured.update(values=values.copy(), point=point.copy(), genres=genres.copy())
        return original(values, point, genres, *args, **kwargs)

    with patch.object(baselines, "select_slate", side_effect=inspect):
        prediction = model.predict(inp, budget)
    return prediction, captured


def exact_smr(values, point, k, initial, time_limit=2.0, wall_budget=5.0, tolerance=1e-8):
    """Threshold-cover equivalence for deterministic setwise minimax regret.

    R(L)<=r iff for every hypothesis at least one selected item has gap<=r.
    MILP feasibility provides a witness; status=2 provides infeasibility. Time
    limits without valid witnesses never count as infeasibility. The objective
    of this MILP is a secondary point-score sum, not the regret itself.
    """
    if values.ndim != 2 or values.shape[1] != len(point) or not np.isfinite(values).all():
        raise ValueError("invalid objective matrix")
    n = len(point)
    if not 0 < k <= n or len(set(initial)) != k:
        raise ValueError("invalid cardinality/initial set")
    best = values.max(axis=1)
    gaps = best[:, None] - values

    def regret(chosen):
        return float((best - values[:, chosen].max(axis=1)).max())

    started = time.perf_counter()
    selected = list(initial)
    upper = regret(selected)
    thresholds = np.unique(gaps[gaps <= upper + tolerance])
    thresholds = np.unique(np.append(thresholds, upper))
    hi = int(np.searchsorted(thresholds, upper))
    lo = -1
    calls = []
    last_tie_optimal = False

    def feasible(threshold):
        remaining = wall_budget - (time.perf_counter() - started)
        if remaining <= 0:
            return "unknown", None, False
        coverage = gaps <= threshold + tolerance
        a = vstack([csr_matrix(coverage.astype(float)), csr_matrix(np.ones((1, n)))], format="csc")
        lower = np.append(np.ones(len(values)), -np.inf)
        upper_bounds = np.append(np.full(len(values), np.inf), float(k))
        result = milp(-np.asarray(point, dtype=float), integrality=np.ones(n),
                      bounds=Bounds(np.zeros(n), np.ones(n)),
                      constraints=LinearConstraint(a, lower, upper_bounds),
                      options={"time_limit": min(time_limit, remaining), "mip_rel_gap": 0.0})
        calls.append({"threshold": float(threshold), "status": int(result.status)})
        if result.x is not None:
            chosen = np.flatnonzero(result.x > 0.5).tolist()
            if len(chosen) <= k and chosen and np.all(coverage[:, chosen].any(axis=1)):
                extras = sorted((j for j in range(n) if j not in chosen), key=lambda j: (-point[j], j))
                chosen += extras[:k - len(chosen)]
                return "feasible", chosen, result.status == 0
        if result.status == 2:
            return "infeasible", None, False
        return "unknown", None, False

    zero, witness, tie = feasible(0.0)
    if zero == "feasible":
        selected, hi, lo, last_tie_optimal = witness, 0, -1, tie
    elif zero == "infeasible":
        lo = 0
        while hi - lo > 1:
            mid = (lo + hi) // 2
            status, witness, tie = feasible(float(thresholds[mid]))
            if status == "feasible":
                hi, selected, last_tie_optimal = mid, witness, tie
            elif status == "infeasible":
                lo = mid
            else:
                break
        if hi - lo <= 1:
            status, witness, tie = feasible(float(thresholds[hi]))
            if status == "feasible":
                selected, last_tie_optimal = witness, tie
    selected = sorted(selected, key=lambda j: (-point[j], j))
    final = regret(selected)
    grid_closed = (zero == "feasible" or (zero == "infeasible" and hi - lo <= 1))
    if final > upper + tolerance:
        raise ValueError("certified solver worsened known feasible upper bound")
    return {"chosen": selected, "regret": final, "initial_regret": upper,
            "global_optimum_certified_to_tolerance": bool(grid_closed),
            "secondary_tie_objective_optimal": bool(last_tie_optimal),
            "lower_bound": float(thresholds[hi]) if grid_closed else (float(thresholds[lo]) if lo >= 0 else 0.0),
            "upper_bound": final, "tolerance": tolerance,
            "solver_calls": calls, "elapsed_seconds": time.perf_counter() - started}
