"""Boundary audit of restricted effort game; no new policies chosen on test."""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
import json
import random

from experiments.core.mechanism.certified_effort_game import empirical_game, equilibria, regret, select_equilibrium
from experiments.mechanism.replay_certified_effort_game import cases_for_game, numerical, SCALES
from experiments.mechanism.replay_budgeted_information import ROOT, OUT, load_observations, source
from experiments.mechanism.run_upgrade_pilot import freeze


def diversity(observations, upgrades, outcomes):
    result = {}
    n = len(observations)
    for backend in ('zhipu_flash', 'deepseek'):
        votes = {cid: [o.initial.answer if backend == 'zhipu_flash' else upgrades[(cid, i)][0].answer
                       for i, o in enumerate(panel)] for cid, panel in observations.items()}
        errors = [[int(v[i] != outcomes[cid]) for cid, v in votes.items()] for i in range(3)]
        pairs = {}
        for i, j in combinations(range(3), 2):
            pairs[str((i, j))] = {'joint_error_rate': sum(x*y for x, y in zip(errors[i], errors[j]))/n,
                'product_of_marginal_error_rates_not_a_conditional_independence_test': (sum(errors[i])/n)*(sum(errors[j])/n)}
        result[backend] = {'n': n, 'unanimous_panels': sum(len(set(v)) == 1 for v in votes.values()),
            'all_three_correct': sum(all(c == outcomes[cid] for c in v) for cid, v in votes.items()),
            'at_least_one_correct_oracle_not_selector': sum(any(c == outcomes[cid] for c in v) for cid, v in votes.items()),
            'pairs': pairs}
    return result


def bootstrap_calibration(cases, labels, reps=256):
    rng = random.Random(20261023)
    records = {str(s): {'selected_counts': Counter(), 'all_keep_NE_count': 0} for s in SCALES}
    for _ in range(reps):
        sample, sampled_labels = [], {}
        for j, row in enumerate(rng.choices(cases, k=len(cases))):
            sample.append(dict(row, id=str(j)))
            sampled_labels[str(j)] = labels[row['id']]
        for scale in SCALES:
            table = empirical_game(sample, sampled_labels, F(scale))
            records[str(scale)]['selected_counts'][str(select_equilibrium(table))] += 1
            records[str(scale)]['all_keep_NE_count'] += int((0, 0, 0) in equilibria(table))
    return {k: {'repetitions': reps, 'selected_profile_counts': dict(sorted(v['selected_counts'].items())),
                'all_keep_NE_fraction': v['all_keep_NE_count']/reps} for k, v in records.items()}


def main():
    train, train_hi, _ = load_observations(source.OUT)
    test, hi, _ = load_observations(ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1')
    cal_labels = {r['id']: r['label'] for r in json.loads((ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    labels = {r['id']: r['label'] for r in json.loads((ROOT/'experiments/results/resource_allocation/routing_holdout_data_v1/private_labels.json').read_text())}
    cases = cases_for_game(train, train_hi)
    participation = {}
    for scale in SCALES:
        table = empirical_game(cases, cal_labels, F(scale))
        chosen = select_equilibrium(table)
        participation[str(scale)] = {'selected_profile': chosen,
            'total_expected_calibration_utilities_including_initial': table[chosen]['utilities'],
            'incremental_utilities_after_sunk_initial': [table[chosen]['utilities'][i]+table[chosen]['initial_costs'][i] for i in range(3)],
            'all_keep_total_utilities': table[(0, 0, 0)]['utilities'],
            'agents_below_zero_cost_outside_option': [i for i, u in enumerate(table[chosen]['utilities']) if u < 0],
            'retainer_min_to_offset_calibration_expected_loss_only_not_a_payment_design': [max(F(0), -u) for u in table[chosen]['utilities']]}
    result = numerical({'scope': 'supplementary no-call scope/selection audit, no modified contract or results',
        'new_api_calls': 0, 'bootstrap_calibration_only': bootstrap_calibration(cases, cal_labels),
        'participation': participation,
        'diversity': {'calibration': diversity(train, train_hi, cal_labels), 'opened_evaluation': diversity(test, hi, labels)},
        'indispensable_scope': ['K/U game is conditional on enrolled actors and mandatory initial production',
            'negative total expected utilities mean voluntary entry versus a zero-cost outside option is NOT established',
            'after-sunk-cost utility and full deployment utility are different quantities, both shown',
            'retainer numbers are lower bounds against fitted mean losses only, not an implemented subsidy or true participation guarantee',
            'no private-cost elicitation or bidding protocol exists',
            'shared errors reflect task difficulty as well as dependence; marginal products are not conditional-independence tests',
            'bootstrap on opened calibration is a sensitivity audit, not prospective confirmation',
            'no recommendation retuned from evaluation accuracy or empirical regret']})
    freeze(OUT/'certified_effort_diagnostics.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__': main()
