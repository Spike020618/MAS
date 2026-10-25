"""Independent exhaustive checker; deliberately does not call settlement()."""
from itertools import product


def audit(contract, actions):
    profiles = list(product((0, 1), repeat=contract.n))
    values = dict(zip(profiles, contract.values))
    beta = contract.budget / (2 * contract.n)

    def payoff(profile, i):
        reference = list(profile)
        reference[i] = 0
        return beta * (1 + values[profile] - values[tuple(reference)]) - contract.costs[i][profile[i]]

    rows = []
    for profile in profiles:
        gains = []
        for i in range(contract.n):
            alternative = list(profile)
            alternative[i] = 1 - alternative[i]
            gains.append(payoff(tuple(alternative), i) - payoff(profile, i))
        rows.append({'actions': profile, 'gains': gains, 'regret': max(0.0, *gains),
                     'objective': beta * values[profile] - sum(contract.costs[i][a] for i, a in enumerate(profile))})
    selected = next(r for r in rows if r['actions'] == tuple(actions))
    return {'selected': selected, 'profiles': rows,
            'global_objective_gap': max(r['objective'] for r in rows) - selected['objective']}
