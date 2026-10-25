"""Versioned P2 collection, isolated scoring, frozen proxy and simple baselines."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from experiments.core.mechanism.upgrade_game import Contract, digest, freeze_state, upgrade_request, improve
from experiments.core.mechanism.upgrade_execution import select, seal_selected
from experiments.core.mechanism.upgrade_value import COST_GRID, PROFILES, contract_for, fit, outcomes, state_key, values
from experiments.core.mechanism.upgrade_decode import parse, is_failure
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, initial_request, live_transport, lock, public_cases, read_rows, timeline


def stage_contract(stage, model, initial):
    if stage == 'calibration':
        return Contract((0,) * 8, ((0, .01 / 6),) * 3, 1,
                        'calibration-full-K-output-before-counterfactuals-NOT-trained-contract')
    if stage != 'development' or model is None:
        raise ValueError('development requires frozen calibration model')
    return contract_for(model, initial)


def check_cases(cases, stage):
    limit = 120 if stage == 'calibration' else 60
    if not 1 <= len(cases) <= limit or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('stage size or duplicate IDs invalid')
    for case in cases:
        public_cases([case])


def collect(cases, directory, provenance, stage, model, transport, identity):
    check_cases(cases, stage)
    if stage == 'development' and model is None:
        raise ValueError('development model missing; no API call allowed')
    if provenance['public_hashes'][stage] != digest(cases):
        raise ValueError('screened public input mismatch')
    if not provenance.get('history_scan_hash'):
        raise ValueError('missing historical exclusion audit')
    with lock(directory):
        root = Path(__file__).resolve().parents[2]
        files = ['experiments/mechanism/calibrate_upgrade_v2.py', 'experiments/core/mechanism/upgrade_decode.py', 'experiments/core/mechanism/upgrade_value.py',
                 'experiments/core/mechanism/upgrade_game.py', 'experiments/core/mechanism/upgrade_execution.py',
                 'experiments/mechanism/audit_upgrade_game.py', 'experiments/mechanism/run_upgrade_pilot.py',
                 'experiments/mechanism/prepare_upgrade_p2.py',
                 'docs/research/升级博弈_P2预注册_2026-10-22.md',
                 'docs/research/升级博弈_P2工程修订v2_2026-10-22.md']
        protocol = {'version': 'upgrade-P2-v2-failure-slots', 'stage': stage, 'scope': provenance['scope'],
                    'public_hash': digest(cases), 'model_hash': digest(model), 'identity': identity,
                    'max_attempts': 6 * len(cases), 'online_cost': .01,
                    'code_hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}}
        freeze(directory / 'protocol.json', protocol)
        freeze(directory / 'source_provenance.json', provenance)
        freeze(directory / 'public_cases.json', cases)
        freeze(directory / 'frozen_model.json', model)
        journal = CallJournal(directory, transport, 6 * len(cases))
        models = set()
        for case in cases:
            prefix = digest(case)[:20]

            def call(key, request):
                row = journal.call(prefix + '/' + key, request)
                report, usage = parse(row, case['choices'])
                if is_failure(report):
                    freeze(directory / (digest(row['key'])[:20] + '.decode_failure.json'),
                           {'key': row['key'], 'response_hash': row['response_hash'], 'policy': 'explicit_failed_abstention_no_retry'})
                models.add(json.loads(row['response']['body'])['model'])
                if len(models) != 1:
                    raise ValueError('returned model changed; stop before more calls')
                return row, report, usage

            initial = [call('initial/' + str(i), initial_request(case, i))[1] for i in range(3)]
            frozen = freeze_state(case['question'], case['choices'], initial)
            freeze(directory / (prefix + '.state.json'), json.loads(frozen))
            timeline(directory, prefix, 'state', digest(json.loads(frozen)))
            contract = stage_contract(stage, model, initial)
            selection = select(frozen, contract)
            freeze(directory / (prefix + '.selection.json'), selection)
            timeline(directory, prefix, 'selection', digest(selection))
            events = [selection]
            for i, a in enumerate(selection['actions']):
                if a:
                    request = upgrade_request(frozen, i)
                    raw, report, usage = call('upgrade/' + str(i), request)
                    events.append({'type': 'selected_upgrade', 'agent': i, 'request_hash': request['request_hash'],
                                   'response_hash': raw['response_hash'], 'report': report,
                                   'finish_reason': 'stop', 'usage': usage})
            seal = seal_selected(frozen, contract, events)
            freeze(directory / (prefix + '.events.json'), events)
            freeze(directory / (prefix + '.sealed.json'), seal)
            timeline(directory, prefix, 'seal', digest(seal))
            for i, a in enumerate(selection['actions']):
                if not a:
                    call('diagnostic/' + str(i), upgrade_request(frozen, i))
            print('sealed and collected ' + case['id'], flush=True)
        result = audit_stage(directory)
        freeze(directory / 'collection_audit.json', result)
        return result


def audit_stage(directory):
    """Independent raw replay and total-order check, no labels."""
    protocol = json.loads((directory / 'protocol.json').read_text())
    root = Path(__file__).resolve().parents[2]
    if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != sha for p, sha in protocol['code_hashes'].items()):
        raise ValueError('frozen implementation changed; versioned audit required')
    cases = json.loads((directory / 'public_cases.json').read_text())
    model = json.loads((directory / 'frozen_model.json').read_text())
    check_cases(cases, protocol['stage'])
    if digest(cases) != protocol['public_hash'] or digest(model) != protocol['model_hash']:
        raise ValueError('frozen stage input changed')
    provenance = json.loads((directory / 'source_provenance.json').read_text())
    if provenance['public_hashes'][protocol['stage']] != digest(cases):
        raise ValueError('provenance mismatch')
    responses, attempts = read_rows(directory / 'responses.jsonl'), read_rows(directory / 'attempts.jsonl')
    raw = {r['key']: r for r in responses}
    attempted = {r['key']: r for r in attempts}
    log = read_rows(directory / 'timeline.jsonl')
    if [r['sequence'] for r in log] != list(range(len(log))):
        raise ValueError('invalid timeline sequence')
    events = {(r['key'], r['event']): r for r in log}
    if len(events) != len(log) or len(raw) != len(responses) or len(attempted) != len(attempts):
        raise ValueError('duplicate event or call')
    if set(raw) != set(attempted) or len(raw) != 6 * len(cases):
        raise ValueError('unresolved, extra or missing attempt')

    def position(key, event, commitment):
        e = events[(key, event)]
        if e['commitment'] != commitment:
            raise ValueError('timeline commitment mismatch')
        return e['sequence']

    expected_keys, models = set(), set()
    for case in cases:
        prefix = digest(case)[:20]

        def response(phase, i, request):
            key = prefix + '/' + phase + '/' + str(i)
            expected_keys.add(key)
            r = raw[key]
            if r['request'] != request or r['request_hash'] != digest(request) or r['response_hash'] != digest(r['response']):
                raise ValueError('request or raw response changed')
            if attempted[key]['request_hash'] != r['request_hash']:
                raise ValueError('attempt request mismatch')
            start, end = position(key, 'attempt', r['request_hash']), position(key, 'response', r['response_hash'])
            if start >= end:
                raise ValueError('response before attempt')
            report, usage = parse(r, case['choices'])
            models.add(json.loads(r['response']['body'])['model'])
            return r, report, usage, start, end

        initial, ends = [], []
        for i in range(3):
            _, report, _, _, end = response('initial', i, initial_request(case, i))
            initial.append(report); ends.append(end)
        frozen = freeze_state(case['question'], case['choices'], initial)
        if json.loads((directory / (prefix + '.state.json')).read_text()) != json.loads(frozen):
            raise ValueError('initial state mismatch')
        contract = stage_contract(protocol['stage'], model, initial)
        selection = select(frozen, contract)
        selection_json = json.loads(json.dumps(selection))
        saved_selection = json.loads((directory / (prefix + '.selection.json')).read_text())
        saved_events = json.loads((directory / (prefix + '.events.json')).read_text())
        if selection_json != saved_selection or saved_events[0] != saved_selection:
            raise ValueError('selection replay mismatch')
        state_pos = position(prefix, 'state', digest(json.loads(frozen)))
        select_pos = position(prefix, 'selection', digest(selection))
        if not max(ends) < state_pos < select_pos:
            raise ValueError('initial state or choice ordering invalid')
        seal = seal_selected(frozen, contract, [selection] + saved_events[1:])
        if seal != json.loads((directory / (prefix + '.sealed.json')).read_text()):
            raise ValueError('output seal mismatch')
        seal_pos = position(prefix, 'seal', digest(seal))
        selected_events = {e['agent']: e for e in saved_events[1:]}
        for i, a in enumerate(selection['actions']):
            r, report, usage, start, end = response('upgrade' if a else 'diagnostic', i, upgrade_request(frozen, i))
            if a:
                e = selected_events[i]
                if not select_pos < start < end < seal_pos or e['report'] != report or e['usage'] != usage or e['response_hash'] != r['response_hash']:
                    raise ValueError('selected upgrade unsupported or reordered')
            elif start <= seal_pos:
                raise ValueError('diagnostic before output seal')
    if expected_keys != set(raw) or len(models) != 1:
        raise ValueError('unknown jobs or model version change')
    return {'status': 'passed', 'stage': protocol['stage'], 'scope': protocol['scope'],
            'questions': len(cases), 'responses': len(raw), 'returned_models': sorted(models),
            'labels_read': False,
            'invalid_report_slots': sum(is_failure(parse(r, next(c['choices'] for c in cases if digest(c)[:20] == r['key'].split('/')[0]))[0]) for r in responses),
            'tokens': sum(json.loads(r['response']['body'])['usage']['total_tokens'] for r in responses)}


def scored_rows(directory, labels_path):
    # Must prove collection COMPLETE before reading any private label file.
    audit_stage(directory)
    cases = json.loads((directory / 'public_cases.json').read_text())
    labels_list = json.loads(labels_path.read_text())
    provenance = json.loads((directory / 'source_provenance.json').read_text())
    stage = json.loads((directory / 'protocol.json').read_text())['stage']
    if provenance.get('private_hashes', {}).get(stage) != digest(labels_list):
        raise ValueError('private labels do not match the frozen source preparation')
    labels = {r['id']: r['label'] for r in labels_list}
    if len(labels) != len(labels_list) or set(labels) != {c['id'] for c in cases}:
        raise ValueError('label set mismatch')
    raw = {r['key']: r for r in read_rows(directory / 'responses.jsonl')}
    result = []
    for case in cases:
        prefix = digest(case)[:20]
        selection = json.loads((directory / (prefix + '.selection.json')).read_text())
        initial_rows = [raw[prefix + '/initial/' + str(i)] for i in range(3)]
        upgrade_rows = [raw[prefix + ('/upgrade/' if a else '/diagnostic/') + str(i)] for i, a in enumerate(selection['actions'])]
        initial = [parse(r, case['choices'])[0] for r in initial_rows]
        upgraded = [parse(r, case['choices'])[0] for r in upgrade_rows]
        result.append({'id': case['id'], 'state': state_key(initial), 'initial': initial,
                       'scores': outcomes(initial, upgraded, case['choices'], labels[case['id']]),
                       'initial_failures': sum(is_failure(r) for r in initial),
                       'upgrade_failures': sum(is_failure(r) for r in upgraded),
                       'initial_tokens': sum(parse(r, case['choices'])[1]['total_tokens'] for r in initial_rows),
                       'upgrade_tokens': [parse(r, case['choices'])[1]['total_tokens'] for r in upgrade_rows],
                       'component_wall_seconds': [r['wall_seconds'] for r in initial_rows + upgrade_rows]})
    return result


def metric(rows, profiles):
    scores = [r['scores'][PROFILES.index(tuple(p))] for r, p in zip(rows, profiles)]
    return {'correct': sum(scores), 'n': len(rows), 'accuracy': sum(scores) / len(rows),
            'upgrades': sum(sum(p) for p in profiles), 'deployed_calls': sum(3 + sum(p) for p in profiles),
            'tokens': sum(r['initial_tokens'] + sum(t for t, a in zip(r['upgrade_tokens'], p) if a) for r, p in zip(rows, profiles)),
            'sum_observed_component_wall_seconds_NOT_end_to_end': sum(sum(r['component_wall_seconds'][:3]) + sum(t for t, a in zip(r['component_wall_seconds'][3:], p) if a) for r, p in zip(rows, profiles)),
            'question_scores': scores}


def rotation(row, k):
    offset = int(digest(['rotation-v1', row['id']])[:8], 16) % 3
    return tuple(int((i - offset) % 3 < k) for i in range(3))


def paired_diagnostic(rows, treatment, baseline):
    differences = [a - b for a, b in zip(treatment['question_scores'], baseline['question_scores'])]
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['id'].split(':')[0], []).append(i)
    rng = random.Random(20261022)
    draws = []
    for _ in range(2000):
        selected = [rng.choice(indices) for indices in groups.values() for _ in indices]
        draws.append(sum(differences[i] for i in selected) / len(selected))
    draws.sort()
    return {'corrected': sum(d == 1 for d in differences), 'harmed': sum(d == -1 for d in differences),
            'accuracy_delta': sum(differences) / len(rows), 'bootstrap_95_interval': [draws[49], draws[1949]],
            'scope': 'question-paired domain-stratified bootstrap, 2000 draws; development exploratory, not confirmatory',
            'by_domain': {domain: {'n': len(indices), 'net_correct': sum(differences[i] for i in indices)}
                          for domain, indices in groups.items()}}


def compare(calibration, development, model):
    if {r['id'] for r in calibration} & {r['id'] for r in development}:
        raise ValueError('calibration/development overlap')
    n = len(development)
    best_fixed = min(PROFILES, key=lambda p: (-sum(r['scores'][PROFILES.index(p)] for r in calibration), sum(p), p))
    best_k = min(range(4), key=lambda k: (-sum(r['scores'][PROFILES.index(rotation(r, k))] for r in calibration), k))
    simple_profiles = {'all_K': [(0, 0, 0)] * n, 'all_U': [(1, 1, 1)] * n,
                       'calibration_fixed': [best_fixed] * n,
                       'calibration_rotation_k': [rotation(r, best_k) for r in development]}
    # Include all predeclared fixed/rotation policies for a transparent frontier.
    simple_profiles.update({'fixed_' + ''.join(map(str, p)): [p] * n for p in PROFILES})
    simple_profiles.update({'rotation_' + str(k): [rotation(r, k) for r in development] for k in range(4)})
    simple = {k: metric(development, p) for k, p in simple_profiles.items()}
    mse_state = sum((values(model, r['initial'])[j] - r['scores'][j]) ** 2 for r in development for j in range(8)) / (8 * n)
    mse_pooled = sum((model['pooled'][j] - r['scores'][j]) ** 2 for r in development for j in range(8)) / (8 * n)
    candidates = []
    for cost in COST_GRID:
        contracts = [contract_for(model, r['initial'], cost) for r in development]
        selected = [tuple(improve(c)['actions']) for c in contracts]
        m = metric(development, selected)
        central = [min(PROFILES, key=lambda p: (-c.settlement(p)['potential'], sum(p), p)) for c in contracts]
        random_profiles = []
        expected_correct = 0.
        for r, p in zip(development, selected):
            same_k = [q for q in PROFILES if sum(q) == sum(p)]
            chosen = same_k[int(digest(['matched-k-random-v1', r['id']])[:8], 16) % len(same_k)]
            random_profiles.append(chosen)
            expected_correct += sum(r['scores'][PROFILES.index(q)] for q in same_k) / len(same_k)
        dominated = [name for name, s in simple.items() if s['correct'] >= m['correct'] and s['upgrades'] <= m['upgrades'] and
                     (s['correct'] > m['correct'] or s['upgrades'] < m['upgrades'])]
        variable = len(set(selected)) > 1
        qualified = (m['correct'] >= simple['all_K']['correct'] + 2 and
                     m['correct'] >= simple['all_U']['correct'] and m['upgrades'] <= .8 * 3 * n and
                     mse_state < mse_pooled and variable and not dominated)
        candidates.append({'cost_lambda': cost, 'mechanism': m, 'central_proxy_optimum': metric(development, central),
                           'matched_k_random': metric(development, random_profiles),
                           'matched_k_random_expected_correct': expected_correct,
                           'action_counts': dict(Counter(''.join(map(str, p)) for p in selected)),
                           'dominated_by': dominated, 'nonconstant_policy': variable, 'development_signal_gate': qualified,
                           'paired_diagnostics': {name: paired_diagnostic(development, m, simple[name])
                                                  for name in ('all_K', 'all_U', 'calibration_fixed', 'calibration_rotation_k')}})
    eligible = [c for c in candidates if c['development_signal_gate']]
    winner = min(eligible, key=lambda c: (-c['mechanism']['correct'], c['mechanism']['upgrades'], -c['cost_lambda'])) if eligible else None
    return {'scope': 'development selection, NOT independent final test', 'calibration_n': len(calibration),
            'development_n': n, 'calibration_fixed': best_fixed, 'calibration_rotation_k': best_k,
            'state_proxy_mse': mse_state, 'pooled_proxy_mse': mse_pooled,
            'questions_with_action_dependent_correctness': sum(len(set(r['scores'])) > 1 for r in development),
            'baselines': simple, 'candidates': candidates, 'selected_lambda': winner['cost_lambda'] if winner else None,
            'true_value_equilibrium': 'not_established: proxy error to true conditional expectations not bounded by these finite samples',
            'effect_gate': 'positive_development_signal_requires_full_scope_and_registration' if winner else 'STOP_no_qualifying_development_signal',
            'formal_300_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--cases', type=Path, required=True); c.add_argument('--provenance', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True); c.add_argument('--stage', choices=['calibration', 'development'], required=True)
    c.add_argument('--model', type=Path); c.add_argument('--execute', action='store_true')
    f = sub.add_parser('fit')
    f.add_argument('--collection', type=Path, required=True); f.add_argument('--labels', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--calibration', type=Path, required=True); e.add_argument('--calibration-labels', type=Path, required=True)
    e.add_argument('--development', type=Path, required=True); e.add_argument('--development-labels', type=Path, required=True)
    e.add_argument('--model', type=Path, required=True); e.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'collect':
        cases = json.loads(args.cases.read_text()); check_cases(cases, args.stage)
        model = json.loads(args.model.read_text()) if args.model else None
        if args.execute:
            transport, identity = live_transport()
            print(json.dumps(collect(cases, args.output, json.loads(args.provenance.read_text()), args.stage, model, transport, identity)))
        else:
            print(json.dumps({'status': 'preflight_no_API', 'questions': len(cases), 'max_attempts': 6 * len(cases)}))
    elif args.command == 'fit':
        rows = scored_rows(args.collection, args.labels)
        if json.loads((args.collection / 'protocol.json').read_text())['stage'] != 'calibration':
            raise ValueError('fit requires calibration stage')
        model = fit(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        freeze(args.output, model)
        print(json.dumps({'status': 'frozen', 'n': model['n'], 'states': {k: v['n'] for k, v in model['states'].items()}}))
    else:
        model = json.loads(args.model.read_text())
        calibration_audit = audit_stage(args.calibration)
        development_audit = audit_stage(args.development)
        if calibration_audit['returned_models'] != development_audit['returned_models']:
            raise ValueError('model changed across calibration/development; no pooled efficacy acceptance')
        calibration = scored_rows(args.calibration, args.calibration_labels)
        development = scored_rows(args.development, args.development_labels)
        if fit(calibration) != model:
            raise ValueError('model is not the registered calibration fit')
        protocol = json.loads((args.development / 'protocol.json').read_text())
        if protocol['stage'] != 'development' or protocol['model_hash'] != digest(model):
            raise ValueError('development did not execute frozen calibration model')
        result = compare(calibration, development, model)
        result['data_scope'] = protocol['scope']
        result['full_P2_completed'] = len(calibration) == 120 and len(development) == 60 and protocol['scope'] == 'complete_three_domain_P2'
        result['formal_test_authorized_by_gate'] = result['full_P2_completed'] and result['selected_lambda'] is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        freeze(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('baselines', 'candidates')}))

