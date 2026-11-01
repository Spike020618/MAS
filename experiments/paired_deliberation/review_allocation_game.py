"""Executable finite coverage game controlling review actions, not LM persuasion.

Payoffs use PUBLIC boundary-sensitivity proxies, never evaluation labels. This is
standard marginal-contribution potential construction, not novel game theory or
a guarantee that the resulting reviews improve actual recommendations.
"""
from itertools import product
import math


class ReviewGame:
    def __init__(self, targets, costs, players=3):
        self.targets = tuple(dict(t) for t in targets); self.costs = tuple(tuple(row) for row in costs)
        self.n = players
        if players < 2 or len(costs) != players or any(len(row) != len(targets) for row in costs): raise ValueError('cost shape')
        if any(not math.isfinite(c) or c <= 0 for row in costs for c in row): raise ValueError('strict positive review costs required')
        if any(not math.isfinite(t['value']) or t['value'] < 0 or type(t['owner']) is not int or not 0 <= t['owner'] < players for t in targets): raise ValueError('public target proxy')
        # Each software agent has one separate review allowance; aggregate max n.
        self.actions = tuple((None,) + tuple(q for q, t in enumerate(targets) if t['owner'] != i) for i in range(players))

    def check(self, profile):
        if len(profile) != self.n or any(a not in self.actions[i] for i, a in enumerate(profile)): raise ValueError('invalid game action')

    def coverage(self, profile): return sum(self.targets[q]['value'] for q in set(profile) if q is not None)

    def potential(self, profile):
        self.check(profile)
        return self.coverage(profile) - sum(self.costs[i][q] for i, q in enumerate(profile) if q is not None)

    def utility(self, i, profile):
        self.check(profile); q = profile[i]
        if q is None: return 0.0
        without = list(profile); without[i] = None
        return self.coverage(profile) - self.coverage(without) - self.costs[i][q]

    def deviations(self, profile):
        result = []
        for i in range(self.n):
            result.append(max(self.utility(i, tuple(profile[:i]) + (a,) + tuple(profile[i+1:])) - self.utility(i, profile) for a in self.actions[i]))
        return result

    def solve(self, eps=1e-10):
        profile = (None,) * self.n; trace = []; seen = {profile}
        while True:
            changed = False
            for i in range(self.n):
                old = profile; current_u = self.utility(i, old); best = old; best_u = current_u
                for a in self.actions[i]:
                    proposal = old[:i] + (a,) + old[i+1:]; u = self.utility(i, proposal)
                    if u > best_u + eps: best, best_u = proposal, u
                if best != old:
                    delta = self.potential(best) - self.potential(old)
                    if abs(delta - (best_u - current_u)) > 1e-9 or delta <= eps: raise AssertionError('potential identity')
                    profile = best; changed = True
                    trace.append({'player': i, 'from': list(old), 'to': list(best), 'utility_gain': best_u-current_u, 'potential_gain': delta})
                    if profile in seen: raise AssertionError('strict improvement cycle')
                    seen.add(profile)
            if not changed: break
        profiles = tuple(product(*self.actions))
        optimum = max(profiles, key=self.potential)
        residual = max(self.deviations(profile))
        return {'actions': list(profile), 'trace': trace, 'utilities': [self.utility(i, profile) for i in range(self.n)],
                'potential': self.potential(profile), 'max_unilateral_gain': residual,
                'finite_game_epsilon_nash': residual <= eps, 'epsilon': eps,
                'central_optimum_actions': list(optimum), 'central_potential_gap': self.potential(optimum)-self.potential(profile),
                'scope': 'fixed public sensitivity/effort proxy game only; not correctness equilibrium or natural LLM self-interest'}
