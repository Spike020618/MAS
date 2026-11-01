"""Finite quality-aware review allocation; proxy equilibrium is not task truth.

Player i chooses one query q or abstains. Coverage credits only the greatest
nonnegative g[i,q] assigned to each query, never the sum of duplicate reviews.
Phi(a) = sum_q max(0, max_{i:a_i=q} g[i,q]) - lambda sum_i c[i,a_i].
Marginal-contribution utilities form an exact potential game. A shared hard
budget restricts deviations and yields a generalized Nash equilibrium instead.
The central comparator enumerates the identical feasible action space.
"""
import argparse
import itertools
import json
import math
from pathlib import Path


class QualityReviewGame:
    def __init__(self, gains, costs, cost_weight=1., budget=None, epsilon=1e-10):
        self.gains = tuple(tuple(float(x) for x in row) for row in gains)
        self.costs = tuple(tuple(float(x) for x in row) for row in costs)
        self.n = len(self.gains)
        self.m = len(self.gains[0]) if self.n else 0
        if not self.n or not self.m or len(self.costs) != self.n:
            raise ValueError('nonempty matching player/query matrices required')
        if any(len(r) != self.m for r in self.gains + self.costs):
            raise ValueError('rectangular matching matrices required')
        if any(not math.isfinite(v) for r in self.gains + self.costs for v in r):
            raise ValueError('all entries must be finite')
        if any(v < 0 for r in self.costs for v in r):
            raise ValueError('costs must be nonnegative; zero is permitted')
        if not math.isfinite(cost_weight) or cost_weight < 0:
            raise ValueError('nonnegative finite cost weight required')
        if budget is not None and (not math.isfinite(budget) or budget < 0):
            raise ValueError('budget must be nonnegative finite or None')
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError('epsilon must be nonnegative finite')
        self.cost_weight, self.budget, self.epsilon = cost_weight, budget, epsilon
        self.actions = (None,) + tuple(range(self.m))

    def _validate(self, state):
        if len(state) != self.n or any(x is not None and (type(x) is not int or not 0 <= x < self.m) for x in state):
            raise ValueError('one valid query or None per player required')

    def cost(self, state):
        self._validate(state)
        return math.fsum(self.costs[i][q] for i, q in enumerate(state) if q is not None)

    def feasible(self, state):
        return self.budget is None or self.cost(state) <= self.budget

    def coverage(self, state):
        self._validate(state)
        return math.fsum(max([0.] + [self.gains[i][q] for i, a in enumerate(state) if a == q]) for q in range(self.m))

    def potential(self, state):
        return self.coverage(state) - self.cost_weight * self.cost(state)

    def utility(self, state, player):
        if not 0 <= player < self.n:
            raise ValueError('invalid player')
        without = list(state)
        without[player] = None
        return self.potential(state) - self.potential(without)

    def deviations(self, state, player):
        for action in self.actions:
            other = list(state)
            other[player] = action
            other = tuple(other)
            if self.feasible(other):
                yield other

    def max_unilateral_gain(self, state):
        return max([0.] + [self.utility(other, i) - self.utility(state, i)
                          for i in range(self.n) for other in self.deviations(state, i)])

    def central(self):
        # Prefer abstention when the potential is exactly tied.
        best = (None,) * self.n
        for state in itertools.product(self.actions, repeat=self.n):
            if self.feasible(state) and self._key(state) > self._key(best):
                best = state
        return best

    def _key(self, state):
        return self.potential(state), -sum(q is not None for q in state)

    def solve(self, initial=None):
        state = tuple(initial) if initial is not None else (None,) * self.n
        self._validate(state)
        if not self.feasible(state):
            raise ValueError('initial state exceeds shared budget')
        seen, trace = {state}, []
        while True:
            moved = False
            for i in range(self.n):
                before = state
                best = max(self.deviations(state, i), key=self._key)
                delta = self.potential(best) - self.potential(state)
                strict = delta > self.epsilon
                # At exact numerical equality, dropping a redundant free review
                # strictly improves the secondary finite objective (-active).
                cleanup = (best[i] is None and state[i] is not None and
                           self.potential(best) == self.potential(state))
                if strict or cleanup:
                    state = best
                    if state in seen:
                        raise RuntimeError('potential ascent revisited a state')
                    seen.add(state)
                    trace.append({'player': i, 'from': before, 'to': state,
                                  'potential_gain': delta,
                                  'utility_gain': self.utility(state, i) - self.utility(before, i),
                                  'zero_gain_abstention_cleanup': cleanup})
                    moved = True
            if not moved:
                break
        optimum = self.central()
        gain = self.max_unilateral_gain(state)
        return {'actions': state, 'potential': self.potential(state), 'cost_cny': self.cost(state),
                'utilities': [self.utility(state, i) for i in range(self.n)],
                'max_feasible_unilateral_gain': gain, 'epsilon': self.epsilon,
                'epsilon_equilibrium': gain <= self.epsilon,
                'equilibrium_kind': 'generalized_nash' if self.budget is not None else 'ordinary_nash',
                'central_actions': optimum, 'central_potential': self.potential(optimum),
                'central_gap': max(0., self.potential(optimum) - self.potential(state)),
                'trace': trace, 'visited_states': len(seen),
                'scope': 'exact potential allocation over estimated quality; not semantic truth or measured recommendation gain'}


def calibration_demo(summary_path):
    from experiments.paired_deliberation.expected_review_value import asymmetric_review_value
    data = json.loads(Path(summary_path).read_text())
    names = ('glm_flash', 'qwen_flash', 'qwen_plus')
    # Synthetic moments only: not a fitted posterior or held-out experiment.
    moments = [(-.01, .08), (-.03, .04), (0., .02)]
    gains, costs = [], []
    for name in names:
        model = data['models'][name]
        rates = model['effective_parameters']
        gains.append([asymmetric_review_value(mu, sd,
            rates['recovery_if_b_observed_more'], rates['safe_retention_if_a_observed_more'])['incremental_review_gain']
            for mu, sd in moments])
        unit = model['physical']['configured_known_cny'] / model['physical']['physical_attempts']
        costs.append([unit] * len(moments))
    return {'reviewers': names, 'synthetic_moments': moments, 'gains': gains, 'costs_cny': costs,
            'result': QualityReviewGame(gains, costs).solve(),
            'warning': 'illustration only; balanced fit-stratum action rates and synthetic Normal moments do not validate deployment utility; observed mean configured cost is not a price guarantee'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-summary', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibration_demo(args.calibration_summary), ensure_ascii=False, indent=2))
