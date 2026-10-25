"""Restricted effort game with authenticated saved tool paths and delayed pay.

Old, already opened data. Route policy is a fixed ex-ante batch reservation,
not adaptive question routing; labels never enter current-question decisions.
No new API and no free-form report strategy or natural-LLM-rationality claim.
"""
from dataclasses import asdict
from fractions import Fraction as F
from hashlib import sha256
import json

from experiments.core.mechanism.certified_effort_game import (
    PROFILES, empirical_game, equilibria, pivotal_credits, regret,
    select_equilibrium, strict_reply_path, commit_recommendation, majority,
)
from experiments.core.mechanism.budgeted_information_sim import Contract, digest
from experiments.mechanism.replay_budgeted_information import (
    ROOT, OUT, execute, forced_plan, load_observations, source,
)
from experiments.mechanism.run_upgrade_pilot import freeze

SCALES = (0, 1000, 3000, 10000)


def numerical(value):
    if isinstance(value, F): return float(value)
    if isinstance(value, dict): return {str(k): numerical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [numerical(x) for x in value]
    return value


def cases_for_game(observations, upgrades):
    return [{'id': cid, 'choices': panel[0].choices,
             'low': [o.initial.answer for o in panel],
             'high': [upgrades[(cid, i)][0].answer for i in range(3)],
             'low_tokens': [o.initial_tokens for o in panel],
             'high_tokens': [upgrades[(cid, i)][1] for i in range(3)]}
            for cid, panel in observations.items()]


def certify_potential(table):
    edges = 0
    for a in PROFILES:
        for i in range(3):
            b = tuple(1-v if j == i else v for j, v in enumerate(a))
            if table[b]['utilities'][i]-table[a]['utilities'][i] != table[b]['potential']-table[a]['potential']:
                raise ValueError('declared payment, costs and potential do not align')
            edges += 1
    return edges


def settle_traces(rows, profile, scale, outcomes, choices, observations, upgrades):
    """Trusted environment only; agents cannot supply or alter these records."""
    settled = []
    for row in rows:
        votes = [r['accepted_answer'] for r in row['settlement']['records']]
        payments = pivotal_credits(votes, choices[row['id']], outcomes[row['id']], scale)
        ledgers = []
        for i, record in enumerate(row['settlement']['records']):
            if record['action'] != ('U' if profile[i] else 'K'):
                raise ValueError('action receipt does not match committed reservation')
            if profile[i] and record['receipt'] is None:
                raise ValueError('uncertified paid path')
            o = observations[row['id']][i]
            expected_report = upgrades[(row['id'], i)][0] if profile[i] else o.initial
            expected_tokens = o.initial_tokens+(upgrades[(row['id'], i)][1] if profile[i] else 0)
            if record['report'] != asdict(expected_report) or record['tokens'] != expected_tokens:
                raise ValueError('forwarded tool artifact or debit was edited')
            if profile[i] and record['receipt']['source_response_hash'] != upgrades[(row['id'], i)][2]:
                raise ValueError('resource certificate origin mismatch')
            # Gateway controls the accepted conclusion as well as the raw text.
            from experiments.core.mechanism.budgeted_information_sim import valid_probability
            expected_answer = ('abstain' if valid_probability(expected_report.probabilities, len(o.choices))
                               and not expected_report.valid(o.choices) else expected_report.answer)
            if record['accepted_answer'] != expected_answer or record['initial_tokens'] != o.initial_tokens:
                raise ValueError('accepted conclusion or initial debit was edited')
            if record['wallet_remaining'] != 3000-expected_tokens or record['agent'] != i:
                raise ValueError('private ledger edited')
            ledgers.append({'agent': i, 'action': record['action'], 'receipt': record['receipt'],
                'tokens': record['tokens'], 'initial_tokens': record['initial_tokens'],
                'accepted_answer': record['accepted_answer'], 'raw_report': record['report'],
                'credit': payments[i], 'utility': payments[i]-record['tokens'],
                'wallet_remaining': record['wallet_remaining']})
        if row['answer'] != majority(votes, choices[row['id']]) or row['correct'] != int(row['answer'] == outcomes[row['id']]):
            raise ValueError('aggregation or outcome score edited')
        settled.append({'id': row['id'], 'answer': row['answer'], 'correct': row['correct'],
                        'sealed_events_sha256': row['settlement']['aggregate_commitment'],
                        'credits_total': sum(payments), 'credit_cap': 2*scale,
                        'ledgers': ledgers})
    return settled


def evaluate_profile(obs, upgrades, metadata, profile, outcomes, scale):
    if any(o.initial_tokens > 1000 for panel in obs.values() for o in panel):
        raise ValueError('initial production exceeds public 1000-token reservation')
    chosen = {(cid, i) for cid in obs for i, a in enumerate(profile) if a}
    summary, rows = execute(obs, upgrades, forced_plan(obs, chosen), Contract(0), outcomes, metadata)
    settlements = settle_traces(rows, profile, scale, outcomes, {cid: panel[0].choices for cid, panel in obs.items()}, obs, upgrades)
    summary['total_virtual_credit'] = sum(r['credits_total'] for r in settlements)
    return summary, settlements


def simple_baselines(observations, upgrades, metadata, outcomes, calibration, cal_high, cal_labels):
    result = {}
    for backend in ('zhipu_flash', 'deepseek'):
        calibration_counts = [sum((panel[i].initial if backend == 'zhipu_flash' else cal_high[(cid, i)][0]).answer == cal_labels[cid]
                                  for cid, panel in calibration.items()) for i in range(3)]
        # Frozen public tie: lowest role ID, not chosen from evaluation outcomes.
        selected_role = min(range(3), key=lambda i: (-calibration_counts[i], i))
        for i in range(3):
            correct = sum((panel[i].initial if backend == 'zhipu_flash' else upgrades[(cid, i)][0]).answer == outcomes[cid]
                          for cid, panel in observations.items())
            result[f'single:{backend}:role={i}'] = {'correct': correct,
                'calls_replayed': len(observations), 'tokens': sum(metadata[(cid, i, backend)]['tokens'] for cid in observations),
                'sum_recorded_call_seconds': sum(metadata[(cid, i, backend)]['seconds'] for cid in observations),
                'selected_on_calibration': i == selected_role,
                'calibration_correct': calibration_counts[i]}
    return result


def main():
    calibration, cal_high, _ = load_observations(source.OUT)
    test, high, metadata = load_observations(ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1')
    cal_labels_path = ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json'
    cal_labels = {r['id']: r['label'] for r in json.loads(cal_labels_path.read_text())}
    train_cases = cases_for_game(calibration, cal_high)
    plans = {}
    tables = {}
    confirmations = {}
    for scale in SCALES:
        table = empirical_game(train_cases, cal_labels, F(scale))
        edges = certify_potential(table)
        ne = equilibria(table)
        selected = select_equilibrium(table)
        menus, replies = commit_recommendation(table, selected)
        confirmations[str(scale)] = {'recommended': selected, 'individual_replies': replies,
                                    'private_menus': numerical([asdict(menu) for menu in menus])}
        own_ne = equilibria(table, 'individual_utilities')
        # Individual payoffs separate. Fixed lex tie disclosed, not test-chosen.
        own_selected = min(own_ne)
        plans[str(scale)] = {'selected': selected, 'all_calibration_NE': ne,
            'individual_same_cap_selected': own_selected, 'individual_same_cap_NE': own_ne,
            'calibration_table': numerical(table), 'action_edges_exact': edges,
            'strict_reply_paths': {str(a): strict_reply_path(table, a) for a in PROFILES},
            'all_keep_strict_reply_terminal': strict_reply_path(table)[-1],
            'selected_calibration_regret': float(regret(table, selected))}
        tables[scale] = table
    freeze(OUT/'certified_effort_plans_before_evaluation.json', {
        'scope': 'opened-data ex-ante batch effort game, NOT preregistration',
        'all_scales_no_selected_contract': SCALES, 'plans': numerical(plans),
        'calibration_labels_sha256': sha256(cal_labels_path.read_bytes()).hexdigest(),
        'policy_uses_no_current_question_or_initial_output': True,
        'evaluation_outcomes_not_loaded_before_freeze': True,
        'strategy_space': 'K/U capacity reservation before task/output; no free-form report deviations',
        'recommendation': 'max calibration potential among NE, then lower incremental cost, then lex',
        'payment': 'scale*(correct full 2-of-3 majority - correct quorum after removing i)',
        'budget': '0<=p_i<=scale; sum p_i<=2*scale; actual token debit independent of credit',
        'trusted_gateway': 'resource purchase irrevocably forwards original cached tool output; no strategic editing, withholding, copying or oracle access',
        'no_novel_difference_reward_theory_claim': True})
    freeze(OUT/'certified_effort_follower_confirmations_before_evaluation.json', {
        'scope': 'programmatic follower acceptance, not natural LLM economic behavior',
        'current_question_and_gold_are_not_follower_inputs': True,
        'confirmations': confirmations})
    labels_path = ROOT/'experiments/results/resource_allocation/routing_holdout_data_v1/private_labels.json'
    outcomes = {r['id']: r['label'] for r in json.loads(labels_path.read_text())}
    test_cases = cases_for_game(test, high)
    profiles = {}
    for profile in PROFILES:
        s, rows = evaluate_profile(test, high, metadata, profile, outcomes, 0)
        profiles[str(profile)] = s
    scenarios, traces = {}, {}
    for scale in SCALES:
        info = plans[str(scale)]
        selected = info['selected']
        table = empirical_game(test_cases, outcomes, F(scale))
        certify_potential(table)
        summary, rows = evaluate_profile(test, high, metadata, selected, outcomes, scale)
        # Settlement replay MUST equal the payoff table, including ALL initial costs.
        for i in range(3):
            mean_utility = sum(r['ledgers'][i]['utility'] for r in rows)/len(rows)
            if abs(mean_utility-float(table[selected]['utilities'][i])) > 1e-8:
                raise ValueError('online-style ledger and game utilities differ')
        scenarios[str(scale)] = {'selected_profile': selected,
            'selected_summary': summary,
            'selected_calibration_NE': regret(tables[scale], selected) == 0,
            'selected_opened_evaluation_empirical_regret': float(regret(table, selected)),
            'evaluation_potential_global_gap_not_contract_optimality': float(max(r['potential'] for r in table.values())-table[selected]['potential']),
            'all_calibration_NE_evaluation_correct_counts': {str(a): profiles[str(a)]['correct'] for a in info['all_calibration_NE']},
            'individual_same_cap_selected_profile': info['individual_same_cap_selected'],
            'individual_same_cap_selected_summary': profiles[str(info['individual_same_cap_selected'])],
            'individual_same_cap_actual_virtual_payment': float(sum(table[info['individual_same_cap_selected']]['individual_payments_same_cap'])*len(test)),
            'all_evaluation_empirical_NE': numerical(equilibria(table)),
            'evaluation_table_offline_only': numerical(table)}
        traces[str(scale)] = rows
    simple = simple_baselines(test, high, metadata, outcomes, calibration, cal_high, cal_labels)
    # Mixed fixed routes bypass redundant initial calls when a role goes direct
    # to high. Outside original K/U-after-initial protocol, but relevant deployment.
    direct_mixes = {}
    for a in PROFILES:
        direct_mixes[str(a)] = {'correct': profiles[str(a)]['correct'], 'calls_replayed': 72,
            'tokens': sum(metadata[(cid, i, 'deepseek' if a[i] else 'zhipu_flash')]['tokens'] for cid in test for i in range(3))}
    result = {'scope': 'restricted empirical verifiable-production game over already opened native MC traces',
        'new_api_calls': 0, 'calibration_questions': 24, 'opened_evaluation_questions': 24,
        'payment_uses_true_outcome_only_after_all_seals': True,
        'calibration_plan_sha256': digest(numerical(plans)),
        'profiles_under_mandatory_initial': profiles, 'direct_mixed_profiles_without_redundant_initial': direct_mixes,
        'single_agent_baselines': simple, 'scenarios_all_scales': scenarios,
        'source_inputs_sha256': {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in [
            source.OUT/'responses.jsonl', ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1/responses.jsonl',
            cal_labels_path, labels_path]},
        'what_is_proved': ['exact action-potential identity for the declared authenticated-output K/U game',
            'nonnegative pivotal-credit ceiling for fixed quorum 2 of 3',
            'finite empirical calibration NE computed with exact rational arithmetic',
            'saved-path execution, receipt, per-agent costs and delayed payoff agree'],
        'not_proved': ['truthful free-form or subjective probability reports',
            'known true population utility table or population equilibrium of the selected profile',
            'optimal incentive coefficient or a Stackelberg solution',
            'benefit over fixed-route cooperative baselines or single strong model',
            'AEIC four-field necessity, online execution, or real human/LLM economic behavior',
            'globally optimal learned routes; enumerating a fitted 8-entry table is not population optimization'],
        'limits': ['actions reserved ex ante before task/private outputs, NOT adaptive request routing',
            'trusted gateway enforces immutable forwarding; allowing report edits changes this game',
            'mandatory initial weak calls charged even if selected route is high',
            'scale is virtual credit per pivotal correct outcome, no externally calibrated money conversion',
            'common empirical task distribution; actors cannot look up current evaluation labels',
            'all scales and all calibration NE evaluation outcomes disclosed; no winning profile chosen from evaluation',
            'central recommendation enumerates 8 profiles; not a decentralized equilibrium-discovery algorithm',
            'difference reward is established literature, not claimed original theorem']}
    freeze(OUT/'certified_effort_result.json', numerical(result))
    freeze(OUT/'certified_effort_episode_traces.json', numerical(traces))
    print(json.dumps({'scenarios': {k: {x: v for x, v in s.items() if 'table' not in x} for k, s in result['scenarios_all_scales'].items()},
        'single_agent_baselines': simple}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__': main()
