"""Known-channel mechanism unit checks; all numbers SYNTHETIC, not LLM gains."""
from fractions import Fraction as F
from itertools import product
import json

from experiments.core.mechanism.bayesian_information_fixture import (
    normal_form, equilibria, full_strategy_regret, expected,
    correlated_duplicate_counterexample,
)
from experiments.mechanism.replay_budgeted_information import OUT
from experiments.mechanism.run_upgrade_pilot import freeze


def numerical(value):
    if isinstance(value, F): return float(value)
    if isinstance(value, dict): return {str(k): numerical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [numerical(v) for v in value]
    return value


def improve(table, start, order):
    """Strict sequential best replies; constant costs debited to their owner."""
    state = start
    path = [state]
    for _ in range(len(table)):
        changed = False
        for i in order:
            candidate = tuple(1-v if j == i else v for j, v in enumerate(state))
            if table[candidate]['expected_utilities'][i] > table[state]['expected_utilities'][i]:
                if table[candidate]['potential'] <= table[state]['potential']:
                    raise ValueError('potential did not strictly increase')
                state = candidate
                if state in path: raise ValueError('strict improvement cycle')
                path.append(state)
                changed = True
        if not changed: return state, path
    raise ValueError('did not converge within finite profiles')


def audit_game(qualities, costs, report_checks=True):
    table = normal_form(qualities, costs)
    own_table = normal_form(qualities, costs, individual=True)
    ne = equilibria(table)
    if not ne: raise ValueError('finite exact potential requires pure equilibrium')
    maximum = max(row['potential'] for row in table.values())
    # Every oriented unilateral action edge, exact rational arithmetic.
    edge_count = 0
    for a, row in table.items():
        for i in range(3):
            b = tuple(1-v if j == i else v for j, v in enumerate(a))
            if table[b]['expected_utilities'][i]-row['expected_utilities'][i] != table[b]['potential']-row['potential']:
                raise ValueError('exact potential identity failed')
            edge_count += 1
    paths = []
    for start in table:
        for order in ((0, 1, 2), (2, 1, 0), (1, 2, 0)):
            end, path = improve(table, start, order)
            if end not in ne: raise ValueError('terminal profile is not NE')
            paths.append({'start': start, 'order': order, 'end': end, 'path': path})
    max_report_regret = max((full_strategy_regret(qualities, costs, a) for a in ne), default=F(0)) if report_checks else None
    if max_report_regret not in (None, F(0)): raise ValueError('joint action/report deviation')
    return {'qualities': qualities, 'costs': costs,
        'unit': 'synthetic common-prior utility units, NOT measured tokens, API money or real question accuracy',
        'action_edges_exact': edge_count, 'equilibria': [{'actions': a, **table[a],
            'global_potential_gap': maximum-table[a]['potential']} for a in ne],
        'individual_scoring_equilibria': [{'actions': a, **own_table[a],
            'collective_potential_under_same_costs': table[a]['potential']} for a in equilibria(own_table)],
        'cooperative_global_profiles': [a for a, row in table.items() if row['potential'] == maximum],
        'cooperative_global_value': maximum,
        'full_strategy_regret_at_truthful_NE': max_report_regret,
        'convergence_paths': paths,
        'table': {str(a): row for a, row in table.items()}}


def main():
    # Entire announced small grid, not a selected positive example.
    grid = [audit_game((q,)*3, (c,)*3) for q, c in product(
        (F(3, 5), F(4, 5), F(9, 10)), (F(1, 100), F(3, 100), F(6, 100)))]
    heterogeneous = [audit_game(qs, cs) for qs, cs in [
        ((F(3, 5), F(4, 5), F(9, 10)), (F(1, 100), F(3, 100), F(6, 100))),
        ((F(4, 5), F(4, 5), F(9, 10)), (F(1, 100), F(3, 100), F(3, 100))),
        ((F(9, 10), F(4, 5), F(3, 5)), (F(6, 100), F(3, 100), F(1, 100))),
    ]]
    # Extra full 3^3 cost grid for heterogeneous channels catches equilibrium
    # selection failures. Do not merely display a favorable homogeneous game.
    sensitivity = []
    for cs in product((F(1, 100), F(3, 100), F(6, 100)), repeat=3):
        r = audit_game((F(3, 5), F(4, 5), F(9, 10)), cs, False)
        sensitivity.append({'costs': cs, 'equilibria': r['equilibria'],
                            'cooperative_global_value': r['cooperative_global_value']})
    result = {'scope': 'synthetic exact-arithmetic mechanism UNIT FIXTURE only; not new empirical research gain',
        'new_api_calls': 0, 'mechanism_novelty_claim': False,
        'assumptions': ['common binary prior 1/2 known to all agents',
            'known purchased signal channels independent conditional on hidden state',
            'acquisition decisions made BEFORE any private signal',
            'trusted resource receipt prevents claiming an unpurchased channel',
            'each acquired signal privately observed, reported as a finite symbol',
            'aggregation uses correct channel likelihoods, NOT arithmetic mean of forecasts',
            'others report honestly when checking a unilateral deviation',
            'risk-neutral software agents bear known own costs; trusted delayed outcome',
            'no coalition, dynamic reputation, unknown signal model, or learned contract'],
        'payment': 'p_i = scale/2 * [1 + S(full Bayesian posterior,Y) - S(posterior excluding i,Y)]',
        'potential': 'Phi(a) = scale/2 * E[S(full Bayesian posterior,Y)|a] - sum_i c_i*a_i',
        'proof_scope': 'finite acquisition game and truthful finite signal reporting under these known-channel assumptions',
        'homogeneous_grid': grid, 'heterogeneous_examples': heterogeneous,
        'heterogeneous_cost_grid': sensitivity,
        'correlated_duplicate_counterexample': correlated_duplicate_counterexample(),
        'limits': ['not a replacement for real-trace no-increment result',
            'payoff interaction comes from information overlap under a specified statistical model; not an arbitrary action bonus',
            'exact potential does not make all equilibria globally optimal',
            'cooperative optimum is a benchmark, not a baseline an incentive mechanism must strictly beat',
            'accuracy, prediction score and cost-adjusted potential are distinct objectives',
            'not four-field AEIC necessity, a Stackelberg solution or optimal incentive contract',
            'known-channel values here cannot be copied onto uncalibrated, correlated LLM reports']}
    result = numerical(result)
    freeze(OUT/'bayesian_fixture_verification.json', result)
    print(json.dumps({'homogeneous_games': len(grid), 'heterogeneous_examples': len(heterogeneous),
        'heterogeneous_cost_grid_games': len(sensitivity),
        'oriented_edges_exact': 24*(len(grid)+len(heterogeneous)+len(sensitivity)),
        'strict_improvement_paths': 24*(len(grid)+len(heterogeneous)+len(sensitivity)),
        'suboptimal_NE_in_heterogeneous_cost_grid': sum(e['global_potential_gap'] > 0 for r in result['heterogeneous_cost_grid'] for e in r['equilibria']),
        'synthetic_only': True}, indent=2))
    return result


if __name__ == '__main__': main()
