"""Read-only attribution diagnostics on historical trajectories, never causal mediation.

All subsets/alternative aggregations are post-hoc replays of saved reports, not
new executions or deployable gold-informed policies. No LLM/network dependency.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
OUT = ROOT / 'experiments/results/audit/legacy_gain_attribution_2026-10-22'


def vote(values):
    """Historical plurality: first encountered label breaks ties."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def transitions(left, right):
    if len(left) != len(right):
        raise ValueError('paired outcomes required')
    return {'n': len(left), 'left_correct': sum(left), 'right_correct': sum(right),
            'wrong_correct': sum(not a and b for a, b in zip(left, right)),
            'correct_wrong': sum(a and not b for a, b in zip(left, right))}


def trace_kind(turn):
    main = [x for x in turn['action_trace'] if x['stage'] != 'choose'
            and not x['stage'].startswith('format_repair')]
    return '+'.join(x['backend'] + ':' + x['model'] for x in main)


def record_view(row):
    rounds = defaultdict(list)
    for turn in row['rounds']:
        rounds[turn['round_id']].append(turn)
    if sorted(rounds) != [1, 2, 3]:
        raise ValueError('three full rounds required')
    truth = row['ground_truth']
    result = []
    for rid, turns in sorted(rounds.items()):
        turns.sort(key=lambda x: x['agent_id'])
        if [t['agent_id'] for t in turns] != list(range(5)):
            raise ValueError('five unique registered agents required')
        counts = Counter(t['stance'] for t in turns)
        decision = vote([t['stance'] for t in turns])
        result.append({'round': rid, 'decision': decision, 'correct': decision == truth,
                       'correct_votes': counts[truth], 'counts': dict(counts),
                       'any_correct': counts[truth] > 0,
                       'unanimous': len(counts) == 1,
                       'actions': dict(Counter(t['chosen_action'] for t in turns)),
                       'routes': dict(Counter(trace_kind(t) for t in turns)),
                       'tokens': sum(t['token_input'] + t['token_output'] for t in turns),
                       'calls': sum(len(t['action_trace']) for t in turns)})
    if result[-1]['decision'] != row['final_decision']:
        raise ValueError('historical aggregation does not replay')
    return result


def summarize(rows):
    views = [record_view(r) for r in rows]
    routes = defaultdict(list)
    actions = defaultdict(list)
    rounds = []
    repairs = []
    stages = defaultdict(Counter)
    for rid in range(1, 4):
        turns = [t for r in rows for t in r['rounds'] if t['round_id'] == rid]
        cells = [v[rid-1] for v in views]
        rounds.append({'round': rid, 'questions': len(rows),
            'correct': sum(x['correct'] for x in cells),
            'any_correct': sum(x['any_correct'] for x in cells),
            'unanimous_correct': sum(x['unanimous'] and x['correct'] for x in cells),
            'unanimous_wrong': sum(x['unanimous'] and not x['correct'] for x in cells),
            'mean_correct_votes': statistics.mean(x['correct_votes'] for x in cells),
            'agent_correct': sum(t['stance'] == r['ground_truth'] for r in rows
                                 for t in r['rounds'] if t['round_id'] == rid),
            'tokens': sum(x['tokens'] for x in cells), 'calls': sum(x['calls'] for x in cells),
            'actions': dict(Counter(t['chosen_action'] for t in turns))})
    for r in rows:
        for t in r['rounds']:
            for call in t['action_trace']:
                stage = ('choice' if call['stage'] == 'choose' else
                         'format_repair' if call['stage'].startswith('format_repair') else 'execution')
                stages[stage]['calls'] += 1
                stages[stage]['tokens'] += call.get('prompt_tokens', 0) + call.get('completion_tokens', 0)
            cell = {'correct': t['stance'] == r['ground_truth'],
                    'tokens': t['token_input'] + t['token_output'],
                    'neutral': t['stance'] == 'neutral',
                    'repair': any(x['stage'].startswith('format_repair') for x in t['action_trace'])}
            routes[trace_kind(t)].append(cell)
            actions[t['chosen_action']].append(cell)
            if cell['repair']:
                repairs.append({'dataset': r['dataset'], 'question_id': r['question_id'],
                                'round': t['round_id'], 'agent': t['agent_id'], **cell})
    def grouped(mapping):
        return {k: {'n': len(v), 'correct': sum(x['correct'] for x in v),
                     'neutral': sum(x['neutral'] for x in v),
                     'repairs': sum(x['repair'] for x in v),
                     'mean_tokens': statistics.mean(x['tokens'] for x in v)}
                for k, v in sorted(mapping.items())}
    # Fixed, label-blind replay policies; these do not recover cost of rerunning
    # the sequence with early stopping / changed reputation evolution.
    policies = {}
    for name, selector in [('first_round', lambda r: [t for t in r['rounds'] if t['round_id'] == 1]),
                           ('last_round', lambda r: [t for t in r['rounds'] if t['round_id'] == 3]),
                           ('all_15', lambda r: r['rounds'])]:
        outcomes = [vote([t['stance'] for t in selector(r)]) == r['ground_truth'] for r in rows]
        policies[name] = {'correct': sum(outcomes), 'questions': len(rows)}
    subsets = []
    for size in [1, 3, 5]:
        for ids in combinations(range(5), size):
            for rid in [1, 3]:
                correct = sum(vote([t['stance'] for t in r['rounds']
                                   if t['round_id'] == rid and t['agent_id'] in ids]) == r['ground_truth']
                              for r in rows)
                subsets.append({'round': rid, 'agents': ids, 'correct': correct, 'n': len(rows)})
    return {'n': len(rows), 'rounds': rounds,
            'record_attempt_metadata': {
                'retried_records': sum(r.get('attempt_count', 1) > 1 for r in rows),
                'prior_error_entries': sum(len(r.get('prior_errors', [])) for r in rows),
                'known_extra_unit_attempts': sum(r.get('attempt_count', 1)-1 for r in rows),
                'scope': 'logged final completed trajectories exclude unrecovered usage from failed attempts and hidden transport retries'},
            'final_confusion': {truth: dict(Counter(r['final_decision'] for r in rows if r['ground_truth'] == truth))
                                for truth in sorted({r['ground_truth'] for r in rows})},
            'by_execution_route_observational': grouped(routes),
            'by_action_observational': grouped(actions), 'repair_slots': repairs,
            'stage_costs': {k: dict(v) for k,v in sorted(stages.items())},
            'first_to_last': transitions([v[0]['correct'] for v in views], [v[2]['correct'] for v in views]),
            'fixed_replay_policies': policies, 'all_fixed_subsets_not_selected': subsets,
            'truth_counts': dict(Counter(r['ground_truth'] for r in rows))}


def analyze(rows):
    groups = defaultdict(list)
    index = defaultdict(dict)
    for r in rows:
        if not r['completed'] or r.get('error'):
            raise ValueError('frozen source must contain only complete records')
        condition = r['condition'].split('__')[1].split('@')[0]
        key = (r['dataset'], str(r['question_id']))
        if condition in index[key]:
            raise ValueError('duplicate paired key')
        index[key][condition] = r
        groups[condition].append(r)
    cases = []
    for (dataset, qid), pair in sorted(index.items()):
        if set(pair) != {'no_mechanism', 'outcome_only', 'static_aeic'}:
            raise ValueError('missing paired condition')
        if len({r['ground_truth'] for r in pair.values()}) != 1:
            raise ValueError('inconsistent paired gold')
        views = {c: record_view(r) for c, r in pair.items()}
        left, right = views['no_mechanism'], views['static_aeic']
        label = ('correct' if left[-1]['correct'] else 'wrong') + '_' + ('correct' if right[-1]['correct'] else 'wrong')
        cases.append({'dataset': dataset, 'question_id': qid, 'final_transition': label,
                      'truth': pair['no_mechanism']['ground_truth'], 'views': views,
                      'baseline_first_has_correct': left[0]['any_correct'],
                      'static_first_correct': right[0]['correct'],
                      'static_first_unanimous': right[0]['unanimous'],
                      'static_last_correct_without_first_correct': right[-1]['correct'] and not right[0]['correct']})
    flows = {}
    for label in ['wrong_correct', 'correct_wrong', 'correct_correct', 'wrong_wrong']:
        selected = [x for x in cases if x['final_transition'] == label]
        flows[label] = {'n': len(selected),
            'baseline_first_has_correct': sum(x['baseline_first_has_correct'] for x in selected),
            'static_first_correct': sum(x['static_first_correct'] for x in selected),
            'static_first_unanimous_correct': sum(x['static_first_unanimous'] and x['static_first_correct'] for x in selected),
            'late_static_corrections': sum(x['static_last_correct_without_first_correct'] for x in selected),
            'by_dataset': dict(Counter(x['dataset'] for x in selected))}
    visibility = Counter()
    for r in rows:
        for t in r['rounds']:
            for p in t.get('visible_peer_reports', []):
                visibility['serialized_peer_reports'] += 1
                visibility['serialized_peer_evidence_items'] += len(p.get('E', []))
    return {'scope': 'post-hoc frozen-trace audit, not causal resource/format decomposition',
        'no_api_calls': True, 'conditions': {k: summarize(v) for k, v in sorted(groups.items())},
        'by_domain': {d: {c: summarize([r for r in rs if r['dataset'] == d]) for c, rs in sorted(groups.items())}
                      for d in sorted({r['dataset'] for r in rows})},
        'final_flows': flows, 'serialized_visibility_not_actual_prompt': dict(visibility),
        'case_rows': cases,
        'limitations': ['same realized reports reused: alternative aggregation is not a new execution',
            'action and execution backend are confounded; action strata are self-selected',
            'report-level samples are clustered within questions and sequential reputation histories',
            'metadata peer entries do not prove exposure in independent prompt',
            'old aliases do not establish exact model weights or identity with current endpoints',
            'tokens mix tokenizers/backends; no historical monetary cost reconstructed',
            'no hidden draft retained for Verify: cannot attribute its revision effect']}


def main():
    rows = [json.loads(l) for l in SOURCE.read_text().splitlines() if l.strip()]
    result = analyze(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    paths = [SOURCE, Path(__file__).resolve()]
    (OUT / 'manifest.json').write_text(json.dumps({
        'inputs_and_code_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths},
        'output_sha256': sha256((OUT / 'analysis.json').read_bytes()).hexdigest(),
        'no_api_calls': True}, indent=2) + '\n')
    print(json.dumps({'out': str(OUT), 'flows': result['final_flows'],
        'summary': {c: {'rounds': s['rounds'], 'first_to_last': s['first_to_last']}
                    for c, s in result['conditions'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
