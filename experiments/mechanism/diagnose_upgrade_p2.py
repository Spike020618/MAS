"""Post-collection diagnostics, never used to select questions or actions."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.core.mechanism.upgrade_game import improve
from experiments.core.mechanism.upgrade_value import COST_GRID, PROFILES, contract_for, values
from experiments.mechanism.calibrate_upgrade_v2 import scored_rows
from experiments.mechanism.run_upgrade_pilot import freeze


def diagnose(rows, model):
    interactions = {}
    for state, entry in {**model['states'], 'pooled_fallback': {'values': model['pooled']}}.items():
        table = dict(zip(PROFILES, entry['values']))
        differences = []
        for i in range(3):
            for j in range(i + 1, 3):
                other = next(k for k in range(3) if k not in (i, j))
                for background in (0, 1):
                    p = [0, 0, 0]; p[other] = background
                    q00 = tuple(p); p[i] = 1; q10 = tuple(p)
                    p[j] = 1; q11 = tuple(p); p[i] = 0; q01 = tuple(p)
                    differences.append(table[q11] - table[q10] - table[q01] + table[q00])
        interactions[state] = {'max_abs_pairwise_discrete_difference': max(map(abs, differences)),
                               'scope': 'learned-table nonadditivity, not proven physical/quality interaction'}
    domains = {}
    for domain in sorted({r['id'].split(':')[0] for r in rows}):
        subset = [r for r in rows if r['id'].split(':')[0] == domain]
        domains[domain] = {'n': len(subset), 'all_K_correct': sum(r['scores'][0] for r in subset),
                           'all_U_correct': sum(r['scores'][7] for r in subset),
                           'action_dependent_correctness_questions': sum(len(set(r['scores'])) > 1 for r in subset),
                           'initial_failed_slots': sum(r['initial_failures'] for r in subset),
                           'upgrade_failed_slots': sum(r['upgrade_failures'] for r in subset)}
    candidates = []
    for cost in COST_GRID:
        profiles = [tuple(improve(contract_for(model, r['initial'], cost))['actions']) for r in rows]
        changes = {}
        for group in ('all', 'all_six_reports_valid', 'at_least_one_failed_report'):
            chosen = [i for i, r in enumerate(rows) if group == 'all' or
                      (group == 'all_six_reports_valid' and r['initial_failures'] + r['upgrade_failures'] == 0) or
                      (group == 'at_least_one_failed_report' and r['initial_failures'] + r['upgrade_failures'] > 0)]
            differences = [rows[i]['scores'][PROFILES.index(profiles[i])] - rows[i]['scores'][0] for i in chosen]
            changes[group] = {'n': len(chosen), 'corrected': sum(d == 1 for d in differences),
                               'harmed': sum(d == -1 for d in differences), 'net_correct': sum(differences)}
        candidates.append({'cost_lambda': cost, 'versus_all_K_by_failure_group': changes})
    n = len(rows)
    state_delta_mse = sum(((values(model, r['initial'])[j] - values(model, r['initial'])[0]) -
                           (r['scores'][j] - r['scores'][0])) ** 2 for r in rows for j in range(8)) / (8 * n)
    pooled_delta_mse = sum(((model['pooled'][j] - model['pooled'][0]) -
                            (r['scores'][j] - r['scores'][0])) ** 2 for r in rows for j in range(8)) / (8 * n)
    return {'n': n, 'initial_state_counts': dict(Counter(r['state'] for r in rows)),
            'proxy_table_interactions': interactions,
            'domains': domains, 'candidates': candidates,
            'state_increment_prediction_mse': state_delta_mse, 'pooled_increment_prediction_mse': pooled_delta_mse,
            'scope': 'diagnostic only; all-six-valid subset conditions on potential outputs and is not a causal treatment estimate',
            'true_value_equilibrium': 'not established by residual prediction error or by zero proxy-contract regret'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = scored_rows(args.collection, args.labels)
    result = diagnose(rows, json.loads(args.model.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    freeze(args.output, result)
    print(json.dumps(result))
