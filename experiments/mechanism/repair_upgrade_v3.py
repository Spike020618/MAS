"""Zero-API repair validation. Old dev labels are already exposed: exploratory.

Old upgrade outputs are usable only if their FULL requests equal the v3 request.
Partial-initial states have changed prompts: mark unmeasured, never fake replay.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

from experiments.core.mechanism.upgrade_game import digest, improve
from experiments.core.mechanism.upgrade_value import PROFILES, state_key
from experiments.core.mechanism.upgrade_value_v3 import fit, contract_for
from experiments.core.mechanism.upgrade_protocol_v3 import decode_row, majority, request, snapshot
from experiments.core.mechanism.upgrade_coordination import recommend
from experiments.mechanism.calibrate_upgrade_v2 import audit_stage
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows


def load(directory, labels_path):
    audit = audit_stage(directory)
    cases = json.loads((directory / 'public_cases.json').read_text())
    raw = {r['key']: r for r in read_rows(directory / 'responses.jsonl')}
    rows = []
    for case in cases:
        prefix = digest(case)[:20]
        before_rows = [raw[prefix + '/initial/' + str(i)] for i in range(3)]
        selected = json.loads((directory / (prefix + '.selection.json')).read_text())['actions']
        after_rows = [raw[prefix + ('/upgrade/' if a else '/diagnostic/') + str(i)] for i, a in enumerate(selected)]
        initial = [decode_row(r, case['choices'])[0] for r in before_rows]
        upgraded = [decode_row(r, case['choices'])[0] for r in after_rows]
        frozen = snapshot(case['question'], case['choices'], initial)
        compatible = [request(frozen, i) == r['request'] for i, r in enumerate(after_rows)]
        rows.append({'id': case['id'], 'choices': case['choices'], 'initial': initial, 'upgraded': upgraded,
                     'state': state_key([{'C': r['conclusion']} for r in initial]),
                     'upgrade_request_compatible': compatible, 'all_replayable': all(compatible),
                     'initial_tokens': sum(decode_row(r, case['choices'])[1]['total_tokens'] for r in before_rows),
                     'upgrade_tokens': [decode_row(r, case['choices'])[1]['total_tokens'] for r in after_rows]})
    # Compatibility and membership established without reading gold.
    labels_list = json.loads(labels_path.read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    provenance = json.loads((directory / 'source_provenance.json').read_text())
    if provenance['private_hashes'][protocol['stage']] != digest(labels_list):
        raise ValueError('label commitment mismatch')
    labels = {r['id']: r['label'] for r in labels_list}
    if set(labels) != {r['id'] for r in rows} or len(labels) != len(labels_list):
        raise ValueError('label membership mismatch')
    for row in rows:
        row['initial_correct'] = int(majority(row['initial'], row['choices']) == labels[row['id']])
        row['scores'] = []
        for profile in PROFILES:
            if any(a and not row['upgrade_request_compatible'][i] for i, a in enumerate(profile)):
                row['scores'].append(None)
            else:
                final = [row['upgraded'][i] if a else row['initial'][i] for i, a in enumerate(profile)]
                row['scores'].append(int(majority(final, row['choices']) == labels[row['id']]))
    return rows, audit


def metrics(rows, profiles):
    scores = [r['scores'][PROFILES.index(tuple(p))] for r, p in zip(rows, profiles)]
    if any(v is None for v in scores):
        raise ValueError('unobserved changed-input outcome cannot be scored')
    return {'n': len(rows), 'correct': sum(scores), 'upgrades': sum(map(sum, profiles)),
            'deployed_calls': sum(3 + sum(p) for p in profiles),
            'tokens': sum(r['initial_tokens'] + sum(t for t, a in zip(r['upgrade_tokens'], p) if a) for r, p in zip(rows, profiles)),
            'question_scores': scores}


def run(root, output):
    calibration, ca = load(root / 'p2_calibration_v2', root / 'p2_data_v2/calibration_private.json')
    development, da = load(root / 'p2_development_v2', root / 'p2_data_v2/development_private.json')
    matched_calibration = [r for r in calibration if r['all_replayable']]
    matched_development = [r for r in development if r['all_replayable']]
    model = fit(matched_calibration)
    contracts = [contract_for(model, r['initial']) for r in matched_development]
    joint = [recommend(c) for c in contracts]
    joint_profiles = [r['actions'] for r in joint]
    local_profiles = [improve(c)['actions'] for c in contracts]
    baseline = metrics(matched_development, [(0, 0, 0)] * len(matched_development))
    coordinated = metrics(matched_development, joint_profiles)
    local = metrics(matched_development, local_profiles)
    all_u = metrics(matched_development, [(1, 1, 1)] * len(matched_development))
    simple = metrics(matched_development, [tuple(int((i - int(digest(['rotation-v1', r['id']])[:8], 16) % 3) % 3 < 2) for i in range(3))
                                          if len({p['conclusion'] for p in r['initial']}) > 1 else (0, 0, 0) for r in matched_development])
    # Bounds only, NEVER use labels to select K/U on the unmatched inputs.
    full_decisions = [recommend(contract_for(model, r['initial']))['actions'] for r in development]
    observed, unmeasured = [], []
    for r, actions in zip(development, full_decisions):
        score = r['scores'][PROFILES.index(tuple(actions))]
        (unmeasured if score is None else observed).append(r['id'] if score is None else score)
    table_gap = sum(r['audit']['global_objective_gap'] for r in joint)
    result = {'version': 'v3-offline-repair', 'api_calls_added': 0,
              'scope': 'post-development exploratory replay, NOT independent validation of new method',
              'calibration': {'total': len(calibration), 'matched': len(matched_calibration),
                              'changed_input_unmeasured': [r['id'] for r in calibration if not r['all_replayable']],
                              'C_usable_initial_correct_all_cases': sum(r['initial_correct'] for r in calibration)},
              'development': {'total': len(development), 'matched': len(matched_development),
                              'changed_input_unmeasured': [r['id'] for r in development if not r['all_replayable']],
                              'C_usable_initial_correct_all_cases': sum(r['initial_correct'] for r in development)},
              'matched_development_results': {'initial_C_majority': baseline, 'local_from_KKK': local,
                                              'joint_proxy_equilibrium': coordinated, 'full_upgrade': all_u,
                                              'simple_disagreement_gate': simple},
              'joint_selection': {'max_unilateral_regret': max(r['audit']['selected']['regret'] for r in joint),
                                  'summed_proxy_global_gap': table_gap,
                                  'all_agent_approvals': all(a['accept'] for r in joint for a in r['agent_approvals']),
                                  'equivalent_to_central_proxy_optimum_baseline': True,
                                  'true_value_or_quality_optimum': False},
              'full_development_bound': {'observed_selected_scores': len(observed),
                                         'unmeasured_changed_input_cases': unmeasured,
                                         'possible_correct_count_interval': [sum(observed), sum(observed) + len(unmeasured)],
                                         'initial_C_majority_correct': sum(r['initial_correct'] for r in development),
                                         'scope': 'offline logical score bound, NOT executed or imputed outcomes; not used by the policy'},
              'corrected_vs_initial': sum(a > b for a, b in zip(coordinated['question_scores'], baseline['question_scores'])),
              'harmed_vs_initial': sum(a < b for a, b in zip(coordinated['question_scores'], baseline['question_scores'])),
              'formal_300_started': False,
              'source_audits': {'calibration': ca, 'development': da}}
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / 'model.json', model)
    freeze(output / 'result.json', result)
    freeze(output / 'recommendations.json', [{'id': r['id'], 'recommendation': d} for r, d in zip(matched_development, joint)])
    code = ['experiments/mechanism/repair_upgrade_v3.py',
            'experiments/core/mechanism/upgrade_protocol_v3.py',
            'experiments/core/mechanism/upgrade_coordination.py',
            'experiments/core/mechanism/upgrade_execution_v3.py',
            'experiments/core/mechanism/upgrade_value_v3.py']
    repo = Path(__file__).resolve().parents[2]
    freeze(output / 'manifest.json', {
        'scope': result['scope'], 'api_calls_added': 0,
        'code_sha256': {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in code},
        'source_response_sha256': {s: hashlib.sha256((root / ('p2_' + s + '_v2') / 'responses.jsonl').read_bytes()).hexdigest()
                                   for s in ('calibration', 'development')},
        'result_hash': digest(result), 'model_hash': digest(model)})
    return {k: result[k] for k in ('scope', 'api_calls_added', 'calibration', 'development', 'joint_selection', 'corrected_vs_initial', 'harmed_vs_initial')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('experiments/results/upgrade_game'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output)))
