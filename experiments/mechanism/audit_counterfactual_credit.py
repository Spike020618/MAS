"""Finite exact gradient/variance audit; standard identities, not new theory."""
from itertools import product
import json
from pathlib import Path
from experiments.mechanism.run_upgrade_pilot import freeze

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/results/redesign_v1/credit_theory_audit'


def statistics(table, policies, player, mode):
    profiles = list(product(*(range(len(p)) for p in policies)))
    if set(table) != set(profiles): raise ValueError('complete payoff table required')
    if mode not in ('none', 'coma', 'score_weighted'): raise ValueError('unknown baseline')
    p = policies[player]; samples = []
    for profile in profiles:
        mass = 1.
        for i, action in enumerate(profile): mass *= policies[i][action]
        values = []
        for action in range(len(p)):
            alternate = list(profile); alternate[player] = action
            score = [int(k == action) - q for k, q in enumerate(p)]
            norm2 = sum(v * v for v in score)
            values.append((p[action], table[tuple(alternate)], norm2))
        if mode == 'none': baseline = 0.
        elif mode == 'coma': baseline = sum(prob * q for prob, q, norm in values)
        else:
            denominator = sum(prob * norm for prob, q, norm in values)
            baseline = sum(prob * q * norm for prob, q, norm in values) / denominator if denominator else 0.
        gradient = [(table[profile] - baseline) * (int(k == profile[player]) - q) for k, q in enumerate(p)]
        samples.append((mass, gradient))
    mean = [sum(mass * g[k] for mass, g in samples) for k in range(len(p))]
    variance = sum(mass * sum((g[k] - mean[k]) ** 2 for k in range(len(p))) for mass, g in samples)
    return {'expected_gradient': mean, 'trace_variance': variance}


def main():
    table = {(0,): 0., (1,): 1.}; policies = [[.1, .9]]
    result = {mode: statistics(table, policies, 0, mode) for mode in ('none', 'coma', 'score_weighted')}
    for mode in result:
        assert all(abs(a - b) < 1e-12 for a, b in zip(result[mode]['expected_gradient'], result['none']['expected_gradient']))
    assert result['coma']['trace_variance'] > result['none']['trace_variance']
    assert result['score_weighted']['trace_variance'] < 1e-20
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'counterexample.json', {'scope': 'mathematical counterexample, not task efficacy or original theorem',
        'new_api_calls': 0, 'policies': policies, 'rewards': [0, 1], 'metrics': result,
        'conclusion': 'COMA-style conditional reward mean preserves gradient, but need not reduce variance; score-weighted optimum is known prior art'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
