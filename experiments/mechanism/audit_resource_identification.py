"""Audit existing compute contrasts and reward proxies without any API calls."""
from collections import Counter, defaultdict
import json
from math import comb
from pathlib import Path
from unittest.mock import patch

from experiments.mechanism.collect import generate_one
from experiments.core.mechanism.contracts import CONTRACTS, scale_rewards
from experiments.core.mechanism.controller import MechanismController
from experiments.core.mechanism.behavioral_metrics import evidence_novelty, unsupported_flip
from experiments.core.mechanism.schema import AEICReport, AgentAction

ROOT = Path(__file__).resolve().parents[2]


def execution_messages(action, fixed=False, matched=False):
    """Capture current generation path with a fake transport; never create client."""
    seen = []
    def fake_call(client, messages, temperature, max_tokens, stage):
        seen.append({'stage': stage, 'messages': messages, 'max_tokens': max_tokens})
        raw = json.dumps({'ACTION': action}) if stage == 'choose' else json.dumps(
            {'A': 'premise', 'E': ['observation'], 'I': 'inference', 'C': 'Yes'})
        return raw, {'stage': stage, 'prompt_tokens': 1, 'completion_tokens': 1,
                     'latency_s': 0, 'status': 200, 'backend': 'fake', 'model': 'fake'}
    profile = {'backend': 'deepseek', 'max_tokens_per_call': 1000,
               'execution_calls': 2 if action == 'verify' else 1} if matched else None
    with patch('experiments.mechanism.collect.client_for', return_value=object()), \
         patch('experiments.mechanism.collect._call_json', side_effect=fake_call):
        generate_one('Generic question?', 0, 'deepseek', 'Fixed contract',
                     'No peers', 0, False, resource_tiered=not fixed,
                     fixed_resources=fixed, deepseek_only_tiers=True, matched_profile=profile)
    return [r for r in seen if r['stage'] != 'choose']


def exact_p(b, c):
    n = b+c
    return min(1., 2*sum(comb(n, i) for i in range(min(b, c)+1))/2**n) if n else 1.


def contrast(left, right):
    if left.keys() != right.keys(): raise ValueError('contrast requires paired question sets')
    pairs = [(left[k], right[k]) for k in sorted(left)]
    b = sum(l['final_decision'] != l['ground_truth'] and r['final_decision'] == r['ground_truth'] for l, r in pairs)
    c = sum(l['final_decision'] == l['ground_truth'] and r['final_decision'] != r['ground_truth'] for l, r in pairs)
    return {'n': len(pairs), 'wrong_to_correct': b, 'correct_to_wrong': c,
            'descriptive_exact_mcnemar_p_unadjusted': exact_p(b, c),
            'scope': 'recomputed existing contrast; not a new confirmatory endpoint'}


def summarize(rows):
    turns = [t for row in rows for t in row['rounds']]
    return {'questions': len(rows),
        'correct': sum(r['final_decision'] == r['ground_truth'] for r in rows),
        'agent_rounds': len(turns),
        'actions': dict(Counter(t['chosen_action'] for t in turns)),
        'tokens': sum(t['token_input']+t['token_output'] for t in turns),
        'calls': sum(len(t['action_trace']) for t in turns),
        'sum_call_latency_seconds': sum(t['latency_s'] for t in turns)}


def proxy_counterexamples():
    spec = scale_rewards(CONTRACTS['static_aeic'], .2)
    false = AEICReport('unsupported premise', ['fabricated observation'], 'invalid inference', 'Yes')
    payout = MechanismController(1).settle(0, spec.to_offer('fixed'), AgentAction.LOW_EFFORT, false, 'no')
    previous = AEICReport('a', ['same evidence'], 'i', 'Yes')
    current = AEICReport('a', ['same evidence'], 'i', 'No')
    novel = AEICReport('a', ['zxq_unrelated_947'], 'i', 'No')
    return {'wrong_complete_report_credit': payout['credit'],
        'wrong_complete_report_consistency_signal': payout['component_consistency'],
        'irrelevant_unseen_token_novelty': evidence_novelty(novel, ['familiar evidence']),
        'flip_without_external_text_flagged': unsupported_flip(previous, current, 'yes', 'no', []),
        'same_flip_with_irrelevant_nonempty_text_flagged': unsupported_flip(previous, current, 'yes', 'no', ['unrelated text']),
        'scope': 'constructed logic counterexamples, not measured prevalence or model exploitation rates'}


def audit():
    path = ROOT / 'experiments/results/mechanism/decomposition/mechanism_decomposition_v1.jsonl'
    groups = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        key = (row['dataset'], row['question_id'])
        suffix = row['condition'].split('__variant@')[-1]
        if key in groups[suffix]: raise ValueError('duplicate condition question')
        groups[suffix][key] = row
    schedule = json.loads((path.parent / 'matched_compute_schedule_v1.json').read_text())['profiles']
    checks = Counter()
    for key, full in groups['full_mechanism'].items():
        matched = groups['matched_compute'][key]
        own = {(t['round_id'], t['agent_id']): t for t in matched['rounds']}
        for turn in full['rounds']:
            other = own[(turn['round_id'], turn['agent_id'])]
            pkey = f"{full['dataset']}|{full['question_id']}|{turn['round_id']}|{turn['agent_id']}"
            profile = schedule[pkey]
            # Formatting recovery is separately logged, not part of the main
            # scheduled execution count.
            full_trace = [t for t in turn['action_trace'] if t['stage'] != 'choose' and not t['stage'].startswith('format_repair')]
            matched_trace = [t for t in other['action_trace'] if t['stage'] != 'choose' and not t['stage'].startswith('format_repair')]
            checks['agent_slots'] += 1
            checks['source_actions_match'] += profile['source_action'] == turn['chosen_action']
            checks['execution_call_counts_match'] += len(full_trace) == len(matched_trace) == profile['execution_calls']
            checks['backend_sequences_match'] += [t['backend'] for t in full_trace] == [t['backend'] for t in matched_trace]
            checks['model_sequences_match'] += [t['model'] for t in full_trace] == [t['model'] for t in matched_trace]
    fixed_low = execution_messages('low_effort', fixed=True)
    fixed_high = execution_messages('aeic_effort', fixed=True)
    comparisons = {}
    for action in ['low_effort', 'aeic_effort', 'verify', 'defer_imitate']:
        a, b = execution_messages(action), execution_messages(action, matched=True)
        comparisons[action] = {'call_counts_same': len(a) == len(b),
            'report_prompt_sequence_same': [x['messages'] for x in a] == [x['messages'] for x in b],
            'scope': 'current source path with mocked transport, not archived raw request reconstruction'}
    contrasts = {}
    for left, right in [('no_contract_fixed', 'contract_fixed'),
                        ('contract_fixed', 'full_mechanism'), ('matched_compute', 'full_mechanism')]:
        contrasts[f'{right}_vs_{left}'] = contrast(groups[left], groups[right])
    return {'no_api_calls': True, 'summary': {k: summarize(list(v.values())) for k, v in groups.items()},
        'paired_contrasts': contrasts, 'schedule_checks': dict(checks),
        'current_fixed_resource_path': {'output_caps_same': fixed_low[0]['max_tokens'] == fixed_high[0]['max_tokens'],
            'report_prompts_same': fixed_low[0]['messages'] == fixed_high[0]['messages']},
        'current_matched_compute_prompt_checks': comparisons,
        'proxy_counterexamples': proxy_counterexamples(),
        'identification_limits': ['fixed model and token cap do not imply fixed report instruction',
            'matched schedules bind backend/call count/cap, not realized token count or every report instruction',
            'choice rates alone do not prove actual effort when execution schedule is forced',
            'report novelty/completeness are not factual usefulness or truthfulness',
            'sum_call_latency_seconds is not concurrent end-to-end latency']}


if __name__ == '__main__':
    result = audit()
    out = ROOT / 'experiments/results/mechanism/synthesis/resource_identification_2026-10-21.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
