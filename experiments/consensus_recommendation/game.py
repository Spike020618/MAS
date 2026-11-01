"""Restricted finite zero-sum solver. A certificate is NOT external validity."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve(loss):
    """Rows=declared preference states, columns=explicitly available slates.

    Recommender minimizes expected loss by a lottery over columns; opposing state
    selector maximizes it. Both deviation bounds concern this matrix ONLY.
    """
    a=np.asarray(loss,dtype=float)
    if a.ndim!=2 or min(a.shape)<1 or not np.isfinite(a).all():
        raise ValueError("finite nonempty payoff matrix required")
    h,s=a.shape
    primal=linprog(np.r_[np.zeros(s),1.],A_ub=np.c_[a,-np.ones(h)],b_ub=np.zeros(h),
                   A_eq=np.array([np.r_[np.ones(s),0.]]),b_eq=[1.],bounds=[(0,None)]*s+[(None,None)],method="highs")
    dual=linprog(np.r_[np.zeros(h),-1.],A_ub=np.c_[-a.T,np.ones(s)],b_ub=np.zeros(s),
                 A_eq=np.array([np.r_[np.ones(h),0.]]),b_eq=[1.],bounds=[(0,None)]*h+[(None,None)],method="highs")
    if not primal.success or not dual.success:
        raise RuntimeError("finite game LP failed")
    p,q=primal.x[:-1],dual.x[:-1]
    lower=float(np.min(q@a)); upper=float(np.max(a@p))
    value=float(q@a@p)
    return {"slate_probabilities":p.tolist(),"state_probabilities":q.tolist(),"value":value,
            "lower_bound":lower,"upper_bound":upper,"duality_gap":upper-lower,
            "recommender_deviation_gain":value-lower,"opponent_deviation_gain":upper-value,
            "deterministic_minimax_loss":float(np.min(np.max(a,axis=0))),
            "scope":"declared finite matrix and randomized slates; not all catalog slates, not strategic LLMs, not real-user utility"}
