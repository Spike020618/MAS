"""Fixed-source-weight acquisition game. No semantic truth or LLM rationality claim."""
from itertools import product
import hashlib
import math
import time


def _nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative number required')
    return float(value)


class SourceSharingGame:
    def __init__(self, agents, bundles, source_weights, costs, budget):
        self.agents = tuple(agents)
        if not self.agents or len(set(self.agents)) != len(self.agents):
            raise ValueError('unique nonempty agents required')
        if set(bundles) != set(self.agents) or set(costs) != set(self.agents):
            raise ValueError('complete agent metadata required')
        self.budget = _nonnegative(budget)
        self.weights = {s: _nonnegative(w) for s, w in source_weights.items()}
        if self.weights and not math.isclose(sum(self.weights.values()), 1, abs_tol=1e-12):
            raise ValueError('fixed source weights must sum to one')
        self.bundles = {a: frozenset(bundles[a]) for a in self.agents}
        if any(not bundle <= self.weights.keys() for bundle in self.bundles.values()):
            raise ValueError('unknown source in bundle')
        self.costs = {a: tuple(_nonnegative(c) for c in costs[a]) for a in self.agents}
        if any(len(c) != 2 for c in self.costs.values()):
            raise ValueError('costs must specify D and R')

    def evaluate(self, profile):
        if len(profile) != len(self.agents) or any(type(a) is not int or a not in (0, 1) for a in profile):
            raise ValueError('binary complete profile required')
        selected = [self.bundles[a] if profile[i] else frozenset() for i, a in enumerate(self.agents)]
        counts = {s: sum(s in bundle for bundle in selected) for s in self.weights}
        credits = [sum(self.budget * self.weights[s] / counts[s] for s in bundle) for bundle in selected]
        costs = [self.costs[a][profile[i]] for i, a in enumerate(self.agents)]
        potential = sum(self.budget * self.weights[s] * sum(1/k for k in range(1, n+1))
                        for s, n in counts.items()) - sum(costs)
        return {'profile': list(profile), 'credits': credits, 'costs': costs,
                'followers': [p-c for p, c in zip(credits, costs)],
                'paid': sum(credits), 'potential': potential}

    def regrets(self, profile):
        current = self.evaluate(profile)['followers']
        regrets = []
        for i in range(len(self.agents)):
            alternative = list(profile)
            alternative[i] = 1 - profile[i]
            regrets.append(max(0., self.evaluate(alternative)['followers'][i] - current[i]))
        return regrets

    def equilibria(self, tolerance=1e-10):
        return [self.evaluate(p) for p in product((0, 1), repeat=len(self.agents))
                if max(self.regrets(p)) <= tolerance]

    def run_local_updates(self, initial=None, order=None, tolerance=1e-10):
        """Each agent compares only its two utilities against current public actions."""
        order = list(self.agents if order is None else order)
        if len(order) != len(self.agents) or set(order) != set(self.agents):
            raise ValueError('order must be a permutation')
        profile = list([0]*len(self.agents) if initial is None else initial)
        self.evaluate(profile)
        events = []
        visited = {tuple(profile)}
        started = time.perf_counter()
        while True:
            changed = False
            for agent in order:
                i = self.agents.index(agent)
                current = self.evaluate(profile)
                alternative = list(profile)
                alternative[i] = 1-profile[i]
                candidate = self.evaluate(alternative)
                gain = candidate['followers'][i]-current['followers'][i]
                update = gain > tolerance
                events.append({'agent': agent, 'before': list(profile), 'alternative': alternative,
                               'gain': gain, 'updated': update})
                if update:
                    if tuple(alternative) in visited:
                        raise RuntimeError('strict improvement cycle')
                    visited.add(tuple(alternative))
                    profile = alternative
                    changed = True
            if not changed:
                break
        return {'terminal': self.evaluate(profile), 'events': events,
                'regrets': self.regrets(profile), 'wall_seconds': time.perf_counter()-started,
                'implementation': 'explicit utility policy; no autonomous LLM choices'}


class AcquisitionLedger:
    """Trusted executor ledger: source bytes, identity and permissions are checked.

    Internal audit component, not a cryptographically hardened remote service.
    Receipts are issued only by acquire(); report text cannot create a receipt.
    """
    def __init__(self, task_id, game, sources):
        if set(sources) != set(game.weights):
            raise ValueError('complete source bytes required')
        if any(not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != source
               for source, data in sources.items()):
            raise ValueError('source content hash mismatch')
        self.task_id, self.game, self.sources = task_id, game, dict(sources)
        self.receipts = {}

    def acquire(self, agent, source):
        if agent not in self.game.bundles or source not in self.game.bundles[agent]:
            raise PermissionError('unauthorized acquisition')
        data = self.sources[source]
        if hashlib.sha256(data).hexdigest() != source:
            raise ValueError('source bytes changed')
        key = (agent, source)
        self.receipts[key] = {'task_id': self.task_id, 'agent': agent, 'source': source,
                              'bytes': len(data)}
        return data

    def settle(self, profile):
        self.game.evaluate(profile)
        required = {(agent, source) for i, agent in enumerate(self.game.agents) if profile[i]
                    for source in self.game.bundles[agent]}
        if set(self.receipts) != required:
            raise ValueError('actual acquisition does not match committed profile')
        return {**self.game.evaluate(profile), 'receipts': list(self.receipts.values()),
                'scope': 'acquisition provenance only; not semantic support or truth'}
