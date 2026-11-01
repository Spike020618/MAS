"""Explicit estimator boundary; P0 does not fit models or create real outcome labels."""
import math
class FrozenValues:
    def __init__(self,values,interactions=None):
        self.values=dict(values);self.interactions=dict(interactions or {})
        if any(not math.isfinite(v)for v in list(self.values.values())+list(self.interactions.values())):raise ValueError('finite values required')
    def __call__(self,actions):
        ids=frozenset(a.action_id for a in actions)
        return sum(self.values[id]for id in ids)+sum(v for pair,v in self.interactions.items()if set(pair)<=ids)
