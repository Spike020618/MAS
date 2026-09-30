"""
ConsensusConfig — 可学习的共识参数

包含bid构建(12个) + Stackelberg博弈(4个) = 16个可学习参数。
默认值 = 当前硬编码值，确保向后兼容。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import List, Tuple

import numpy as np


@dataclass
class ConsensusConfig:
    # alignment_willingness = max(aw_floor, ic^aw_ic_exp * eq^aw_eq_exp * (aw_ev_base + aw_ev_scale*ev))
    aw_floor: float = 0.05
    aw_ic_exp: float = 1.0
    aw_eq_exp: float = 0.0
    aw_ev_base: float = 0.5
    aw_ev_scale: float = 0.5

    # revision_cost = max(rc_floor, rc_scale * (1 - agreement))
    rc_floor: float = 5.0
    rc_scale: float = 30.0

    # max_revision_depth = clip(ec, mrd_lo, mrd_hi)
    mrd_lo: float = 0.1
    mrd_hi: float = 0.95

    # historical_alignment = ha_ic*ic + ha_ec*ec + ha_eq*eq + ha_ev*ev
    ha_ic: float = 0.25
    ha_ec: float = 0.25
    ha_eq: float = 0.0
    ha_ev: float = 0.25

    # Stackelberg game
    sg_lambda: float = 1.0
    sg_mu: float = 0.1
    sg_eta: float = 0.05
    sg_gamma: float = 0.1

    # Dynamic sycophancy regulation
    dyn_tau_high: float = 0.6
    dyn_tau_low: float = 0.2
    dyn_alpha: float = 0.5
    dyn_beta: float = 0.3

    _PARAM_NAMES = [f.name for f in fields(__class__)] if False else None

    @classmethod
    def param_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.param_names()])

    @classmethod
    def from_vector(cls, vec) -> "ConsensusConfig":
        names = cls.param_names()
        return cls(**{n: float(vec[i]) for i, n in enumerate(names)})

    @classmethod
    def bounds(cls) -> List[Tuple[float, float]]:
        b = {
            "aw_floor": (0.01, 0.2),
            "aw_ic_exp": (0.3, 3.0),
            "aw_eq_exp": (0.3, 3.0),
            "aw_ev_base": (0.0, 1.0),
            "aw_ev_scale": (0.0, 1.0),
            "rc_floor": (1.0, 20.0),
            "rc_scale": (5.0, 80.0),
            "mrd_lo": (0.01, 0.4),
            "mrd_hi": (0.5, 1.0),
            "ha_ic": (0.0, 1.0),
            "ha_ec": (0.0, 1.0),
            "ha_eq": (0.0, 1.0),
            "ha_ev": (0.0, 1.0),
            "sg_lambda": (0.1, 5.0),
            "sg_mu": (0.01, 1.0),
            "sg_eta": (0.005, 0.3),
            "sg_gamma": (0.01, 0.5),
            "dyn_tau_high": (0.3, 0.9),
            "dyn_tau_low": (0.0, 0.4),
            "dyn_alpha": (0.1, 0.9),
            "dyn_beta": (0.1, 0.6),
        }
        return [b[n] for n in cls.param_names()]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConsensusConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
