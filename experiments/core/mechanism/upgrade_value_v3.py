"""Same shrinkage estimator, with aggregation-implied invariances projected.

Projection is an input/aggregation identity, not a domain rule or reward tuning.
This does not establish a uniform error bound to true conditional task value.
"""
from collections import Counter

from .upgrade_value import fit as previous_fit, PROFILES
from .upgrade_game import Contract, digest


def invariant_to_baseline(state, profile):
    if len(state) != 3 or len(profile) != 3:
        raise ValueError('three-agent state required')
    baseline = next((v for v, count in Counter(state).items() if v != '_' and count >= 2), '_')
    kept = Counter(v for v, action in zip(state, profile) if not action and v != '_')
    fixed = next((v for v, count in kept.items() if count >= 2), None)
    if fixed is not None:
        return fixed == baseline
    # With no retained non-abstention and at most one free vote, no majority.
    return baseline == '_' and not kept and sum(profile) <= 1 or sum(profile) == 0


def fit(rows):
    model = previous_fit(rows)
    model['version'] = 'upgrade-proxy-mean-v3-logical-projection'
    for state, entry in model['states'].items():
        reference = entry['values'][0]
        entry['projected_profiles'] = []
        for i, profile in enumerate(PROFILES):
            if invariant_to_baseline(state, profile):
                entry['values'][i] = reference
                entry['projected_profiles'].append(list(profile))
    model['scope'] = 'post-development repair replay; matched complete-initial calibration only'
    return model


def contract_for(model, initial, cost=.01):
    if cost != .01:
        raise ValueError('repair holds original default cost .01 fixed, no new parameter search')
    from .upgrade_value import state_key
    state = state_key([{'C': r['conclusion']} for r in initial])
    row = model['states'].get(state)
    values = list(row['values'] if row else model['pooled'])
    # Logical identities also apply to previously unseen state patterns.
    for i, profile in enumerate(PROFILES):
        if invariant_to_baseline(state, profile):
            values[i] = values[0]
    return Contract(tuple(values), ((0, cost / 6),) * 3, 1, 'repair-v3-calibration:' + digest(model))
