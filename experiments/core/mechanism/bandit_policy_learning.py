"""Minimal independent REINFORCE actors for fixed-API action policies.

No differentiation through API calls, no invented utility surrogate, no
equilibrium guarantee. The caller must supply observed training utilities.
"""
from math import exp, isfinite, log
from random import Random


class IndependentActors:
    def __init__(self, action_counts, learning_rate=.05, seed=0):
        if not action_counts or any(type(n) is not int or n < 2 for n in action_counts):
            raise ValueError('at least two actions per player')
        if not isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError('bounded positive learning rate')
        self.logits = [[0.] * n for n in action_counts]
        self.learning_rate = learning_rate
        self.steps = 0
        self.baselines = [0.] * len(action_counts)
        self.frozen = False
        self.rng = Random(seed)

    def probabilities(self):
        result = []
        for values in self.logits:
            weights = [exp(v - max(values)) for v in values]
            result.append([v / sum(weights) for v in weights])
        return result

    def sample(self):
        return [self.rng.choices(range(len(p)), weights=p, k=1)[0] for p in self.probabilities()]

    @staticmethod
    def score_gradient(probabilities, action, advantage):
        if not isfinite(advantage) or action not in range(len(probabilities)):
            raise ValueError('finite observed advantage and registered action required')
        if any(not isfinite(p) or p < 0 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-10:
            raise ValueError('probability simplex required')
        return [advantage * (int(j == action) - p) for j, p in enumerate(probabilities)]

    def update(self, actions, observed_utilities):
        if self.frozen:
            raise RuntimeError('evaluation policy is frozen')
        if len(actions) != len(self.logits) or len(observed_utilities) != len(self.logits):
            raise ValueError('one action and observed utility per player')
        if any(not isfinite(v) or not -1 <= v <= 1 for v in observed_utilities):
            raise ValueError('training utilities must be bounded in[-1,1]')
        probabilities = self.probabilities()
        # Compute everything before mutation; the running baseline only uses
        # PRIOR samples and is independent of the current sampled action.
        gradients = [self.score_gradient(p, a, u - b) for p, a, u, b in
                     zip(probabilities, actions, observed_utilities, self.baselines)]
        for i, gradient in enumerate(gradients):
            self.logits[i] = [v + self.learning_rate * g for v, g in zip(self.logits[i], gradient)]
            self.baselines[i] += (observed_utilities[i] - self.baselines[i]) / (self.steps + 1)
        self.steps += 1
        return gradients

    def snapshot(self):
        return {'logits': [v[:] for v in self.logits], 'probabilities': self.probabilities(),
                'steps': self.steps, 'baselines': self.baselines[:],
                'learning_rate': self.learning_rate, 'frozen': self.frozen,
                'scope': 'independent score-function learning, not ECON reproduction or certified equilibrium'}

    def freeze(self):
        self.frozen = True
        return self.snapshot()


def empirical_deviation(payoffs, probabilities):
    """Full-table audit only; never infer unobserved counterfactual utilities.

payoffs maps complete action tuples to one utility per player. A small regret
on this table is NOT population equilibrium, generalization or optimality.
"""
    from itertools import product
    profiles = list(product(*(range(len(p)) for p in probabilities)))
    if set(payoffs) != set(profiles):
        raise ValueError('complete observed payoff table required; cannot fill missing cells')
    for p in probabilities:
        if any(not isfinite(v) or v < 0 for v in p) or abs(sum(p) - 1) > 1e-10:
            raise ValueError('invalid probability simplex')
    def expected(player, forced=None):
        total = 0.
        for profile in profiles:
            weight = 1.
            for j, a in enumerate(profile):
                weight *= int(a == forced) if forced is not None and j == player else probabilities[j][a]
            u = payoffs[profile]
            if len(u) != len(probabilities) or any(not isfinite(v) for v in u):
                raise ValueError('invalid payoff vector')
            total += weight * u[player]
        return total
    return [max(0., max(expected(i, a) for a in range(len(p))) - expected(i))
            for i, p in enumerate(probabilities)]
