"""New offline strategic-agent simulation over two ALREADY OPENED trace batches.

No API. Calibration labels fit a declared expected gain table; evaluation labels
are loaded only AFTER all actions are frozen. This is exploratory transfer,
not independent validation, Bayesian equilibrium or a novel coordination game.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

from experiments.core.mechanism.budgeted_information_sim import (
    Contract, Decision, Episode, Estimate, Observation, ReplayResource, Report,
    choose, digest, valid_probability,
)
from experiments.mechanism.audit_probability_compatibility import compatible_decode
from experiments.mechanism import probe_probability_routing as source
from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/results/resource_allocation/budgeted_information_sim_v2'
WEIGHTS = (0, 1000, 3000, 10000, 30000)


def load_observations(directory):
    """Broker assembly only. No label file, no label field accepted."""
    cases = json.loads((directory/'public_cases.json').read_text())
    rows = read_rows(directory/'responses.jsonl')
    result = json.loads((directory/'result.json').read_text())
    if len(rows) != 144 or len({r['key'] for r in rows}) != 144 or result['response_hash'] != p.digest(rows):
        raise ValueError('incomplete or changed source journal')
    index = {r['key']: r for r in rows}
    initial, upgrades, metadata = {}, {}, {}
    for case in cases:
        if set(case) - {'id', 'source_id', 'question', 'choices'}:
            raise ValueError('unexpected public case fields')
        observations = []
        for agent in range(3):
            for backend in ('zhipu_flash', 'deepseek'):
                row = index[source.key(case, agent, backend)]
                if row['response_hash'] != p.digest(row['response']) or row['request_hash'] != p.digest(row['request']):
                    raise ValueError('raw request/response modified')
                body = p.body(row)
                decoded = compatible_decode(body['choices'][0]['message']['content'], case['choices'])
                q = decoded['probabilities']
                report = Report(decoded['answer'] or 'abstain',
                                tuple(q[c] for c in case['choices']) if q is not None else None)
                tokens = body['usage']['total_tokens']
                if type(tokens) is not int or tokens <= 0:
                    raise ValueError('unknown usage cannot be zero charged')
                metadata[(case['id'], agent, backend)] = {'tokens': tokens,
                    'seconds': row['wall_seconds'], 'model': body['model'],
                    'response_hash': row['response_hash']}
                if backend == 'zhipu_flash':
                    observations.append(Observation(case['id'], agent, tuple(case['choices']), report, tokens))
                else:
                    upgrades[(case['id'], agent)] = (report, tokens, row['response_hash'])
        initial[case['id']] = observations
    return initial, upgrades, metadata


def confidence_cell(o):
    q = o.initial.probabilities
    if not valid_probability(q, len(o.choices)):
        return 'invalid'
    return 'confident' if max(q) >= .8 else 'uncertain'


def fit_table(observations, upgrades, calibration_outcomes, conditional=False):
    """Outcome-supervised historical estimates, never evaluation-outcome fitting.

    Invalid initial probabilities do not support a probability-based decision.
    Invalid HIGH probabilities are included as zero score, not selected away.
    """
    records = []
    for cid, panel in observations.items():
        for o in panel:
            if confidence_cell(o) == 'invalid':
                continue
            report, cost, _ = upgrades[(cid, o.agent)]
            delta = report.score(calibration_outcomes[cid], o.choices) - o.initial.score(calibration_outcomes[cid], o.choices)
            records.append((cid, o.agent, confidence_cell(o), delta, cost))
    table = {}
    for role in range(3):
        for cell in ('confident', 'uncertain'):
            role_records = [r for r in records if r[1] == role]
            rs = [r for r in role_records if r[2] == cell] if conditional else role_records
            fallback = conditional and len(rs) < 5
            if fallback:
                rs = role_records
            n = len(rs)
            gain = sum(r[3] for r in rs)/n if n else 0.
            cost = sum(r[4] for r in rs)/n if n else 0.
            # Range [-1,1], one-sided Hoeffding. Merely conservative sensitivity:
            # adaptivity/opened data and selection prevent a confirmatory 95% claim.
            lower = max(-1., gain-math.sqrt(2*math.log(20)/n)) if n else -1.
            table[(role, cell)] = Estimate(gain, cost, n, lower,
                ('role_fallback' if fallback else 'role_confidence' if conditional else 'role')+f':{role}:{cell}')
    return table


def estimate_for(o, table):
    return table.get((o.agent, confidence_cell(o)), Estimate(0., 0., 0, -1., 'invalid'))


def plan(observations, table, contract, conservative=False):
    # Deliberately no upgrades, metadata, question text, source domain or gold input.
    return {cid: [choose(o, estimate_for(o, table), contract, conservative) for o in panel]
            for cid, panel in observations.items()}


def execute(observations, upgrades, decisions, contract, outcomes, metadata):
    broker = ReplayResource(upgrades)
    rows = []
    for cid, panel in observations.items():
        episode = Episode(panel, contract)
        for o, decision in zip(panel, decisions[cid]):
            episode.submit(o.agent, decision, broker)
        answer = episode.aggregate()
        # Only here does the environment reveal an outcome to the settlement service.
        settlement = episode.settle(outcomes[cid])
        selected = [i for i, d in enumerate(decisions[cid]) if d.action == 'U']
        seconds = sum(metadata[(cid, i, 'zhipu_flash')]['seconds'] for i in range(3))
        seconds += sum(metadata[(cid, i, 'deepseek')]['seconds'] for i in selected)
        rows.append({'id': cid, 'answer': answer, 'correct': int(answer == outcomes[cid]),
            'actions': [asdict(d) for d in decisions[cid]], 'settlement': settlement,
            'tokens': sum(r['tokens'] for r in settlement['records']),
            'low_calls_replayed': 3, 'high_calls_replayed': len(selected),
            'sum_recorded_call_seconds': seconds})
    return summarize(rows), rows


def summarize(rows):
    records = [record for row in rows for record in row['settlement']['records']]
    return {'n': len(rows), 'correct': sum(r['correct'] for r in rows),
        'tokens_including_initial': sum(r['tokens'] for r in rows),
        'low_calls_replayed': 3*len(rows), 'high_calls_replayed': sum(r['high_calls_replayed'] for r in rows),
        'sum_recorded_call_seconds_not_end_to_end': sum(r['sum_recorded_call_seconds'] for r in rows),
        'total_virtual_credit': sum(r['credit'] for r in records),
        'mean_individual_bounded_score_including_invalid_zero': sum(r['score'] for r in records)/len(records),
        'invalid_probability_reports': sum(not valid_probability(r['report']['probabilities'], len(r['report']['probabilities'] or ())) for r in records),
        'minimum_wallet_remaining': min(r['wallet_remaining'] for r in records),
        'max_payment_excess': max(r['settlement']['total_credit']-r['settlement']['max_credit'] for r in rows)}


def forced_plan(observations, selected):
    return {cid: [Decision('U' if (cid, o.agent) in selected else 'K', 0., 'non_strategic_control') for o in panel]
            for cid, panel in observations.items()}


def matched_controls(observations, decisions):
    selected = {(cid, i) for cid, ds in decisions.items() for i, d in enumerate(ds) if d.action == 'U'}
    by_role = {role: sum(i == role for _, i in selected) for role in range(3)}
    fixed, uncertainty = set(), set()
    for role, k in by_role.items():
        eligible = [(cid, panel[role]) for cid, panel in observations.items()
                    if confidence_cell(panel[role]) != 'invalid']
        fixed.update((cid, role) for cid, _ in sorted(eligible, key=lambda x: digest(['fixed-cost-agent-v1', x[0], role]))[:k])
        uncertainty.update((cid, role) for cid, _ in sorted(eligible,
            key=lambda x: (max(x[1].initial.probabilities), digest(['uncertainty-cost-agent-v1', x[0], role])))[:k])
    return {'fixed_same_role_call_counts': forced_plan(observations, fixed),
            'uncertainty_same_role_call_counts': forced_plan(observations, uncertainty)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    first = source.OUT
    second = ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1'
    calibration, train_upgrades, _ = load_observations(first)
    test, test_upgrades, metadata = load_observations(second)
    if set(calibration) & set(test):
        raise ValueError('source batches overlap')
    train_labels_path = ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json'
    train_labels = {r['id']: r['label'] for r in json.loads(train_labels_path.read_text())}
    tables = {name: fit_table(calibration, train_upgrades, train_labels, conditional)
              for name, conditional in [('role', False), ('role_confidence', True)]}
    settings, decisions = {}, {}
    for name, table in tables.items():
        for conservative in (False, True):
            for weight in WEIGHTS:
                tag = f'{name}:lower={conservative}:weight={weight}'
                contract = Contract(weight)
                settings[tag] = (contract, table, conservative)
                decisions[tag] = plan(test, table, contract, conservative)
    plan_snapshot = {tag: {cid: [asdict(d) for d in ds] for cid, ds in ds_by_case.items()}
                     for tag, ds_by_case in decisions.items()}
    freeze(OUT/'plans_before_evaluation.json', {'scope': 'posthoc opened-data transfer simulation',
        'no_evaluation_labels_in_plan': True, 'plans': plan_snapshot,
        'calibration_tables': {name: {str(key): asdict(e) for key, e in table.items()} for name, table in tables.items()},
        'weights_all_disclosed_no_selected_contract': WEIGHTS,
        'wallet': 3000, 'upgrade_reservation': 2000,
        'calibration_labels_sha256': sha256(train_labels_path.read_bytes()).hexdigest()})
    # Evaluation outcomes are first loaded AFTER freezing every action.
    labels_path = ROOT/'experiments/results/resource_allocation/routing_holdout_data_v1/private_labels.json'
    outcomes = {r['id']: r['label'] for r in json.loads(labels_path.read_text())}
    all_slots = {(cid, i) for cid in test for i in range(3)}
    baselines, all_rows = {}, {}
    for name, chosen in [('all_keep', set()), ('all_upgrade_after_initial', all_slots)]:
        baselines[name], all_rows[name] = execute(test, test_upgrades, forced_plan(test, chosen), Contract(0), outcomes, metadata)
    # A direct all-strong deployment skips initial weak calls: do NOT hide it.
    direct = dict(baselines['all_upgrade_after_initial'])
    direct.update(tokens_including_initial=sum(v[1] for v in test_upgrades.values()),
                  low_calls_replayed=0,
                  sum_recorded_call_seconds_not_end_to_end=sum(m['seconds'] for (cid, i, backend), m in metadata.items() if backend == 'deepseek'))
    direct.pop('minimum_wallet_remaining')
    baselines['direct_strong_without_initial'] = direct
    results = {}
    for tag, ds in decisions.items():
        contract, table, conservative = settings[tag]
        summary, rows = execute(test, test_upgrades, ds, contract, outcomes, metadata)
        # An informed cooperative controller can apply the EXACT same rule.
        central = plan(test, table, contract, conservative)
        assert central == ds
        comparisons = {}
        for name, control in matched_controls(test, ds).items():
            s, rs = execute(test, test_upgrades, control, contract, outcomes, metadata)
            corrections = sum(a['correct'] > b['correct'] for a, b in zip(rows, rs))
            harms = sum(a['correct'] < b['correct'] for a, b in zip(rows, rs))
            comparisons[name] = dict(s, mechanism_corrections=corrections, mechanism_harms=harms)
        estimated_regrets = []
        for cid, panel in test.items():
            for o, d in zip(panel, ds[cid]):
                eligible = confidence_cell(o) != 'invalid' and contract.wallet-o.initial_tokens >= contract.upgrade_reservation
                estimated_regrets.append(max(0., d.expected_upgrade_advantage) if eligible and d.action == 'K'
                                         else max(0., -d.expected_upgrade_advantage) if d.action == 'U' else 0.)
        results[tag] = {'summary': summary, 'matched_controls': comparisons,
            'central_same_information_actions_identical': True,
            'unmodeled_initial_probability_slots_no_effort_optimality_claim': sum(confidence_cell(o) == 'invalid' for panel in test.values() for o in panel),
            'complete_strategic_equilibrium_claim': False,
            'max_estimated_table_unilateral_regret_not_true_bayesian_regret': max(estimated_regrets)}
        all_rows[tag] = rows
    inputs = [first/'responses.jsonl', first/'public_cases.json', second/'responses.jsonl', second/'public_cases.json', train_labels_path, labels_path]
    output = {'scope': 'NEW environment over OLD opened traces; not independent validation or online execution',
        'version': 'budgeted-information-simulation-v2',
        'preflight_v1_preserved': 'budgeted_information_sim_v1: superseded before acceptance; v2 adds rejected-C accounting and explicitly unmodeled action slots',
        'new_api_calls': 0, 'questions_calibration': len(calibration), 'questions_evaluation': len(test),
        'actions_frozen_sha256': digest(plan_snapshot), 'inputs_sha256': {str(f.relative_to(ROOT)): sha256(f.read_bytes()).hexdigest() for f in inputs},
        'baselines': baselines, 'scenarios': results,
        'limits': ['programmatic agents maximize estimated utilities, not demonstrated rational LLMs',
            'private ledgers and protocol-level visibility, not process isolation or separate provider billing',
            'outcomes delayed and trusted by assumption; gold is offline settlement only',
            'score estimates learned on previous batch; both batches already opened before this design',
            'score weight is virtual tokens per score, no external economic calibration or optimal weight claim',
            'own absolute scoring is separable, not a new nontrivial coordination game',
            'report truthfulness conditional on fixed subjective beliefs; effort table is not a coherent Bayesian signal model',
            'all-role fallback and confidence boundary are exploratory choices, not preregistered',
            'invalid probabilities score zero; format success can contribute to estimated score gain',
            'same-role call counts do not imply identical token costs; actual token totals disclosed',
            'fixed single draws do not identify acquisition outcome distributions or equilibrium expectations',
            'recorded call-duration sum is not online wall latency',
            'no AEIC four-field value, Stackelberg equilibrium or global optimal contract established']}
    freeze(OUT/'result.json', output)
    freeze(OUT/'episode_traces.json', all_rows)
    print(json.dumps({'baselines': baselines, 'scenarios': {k: v for k, v in results.items()}}, ensure_ascii=False, indent=2))
    return output


if __name__ == '__main__':
    main()
