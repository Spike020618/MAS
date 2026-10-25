"""Post-seal cost/quality accounting. Never calls models or fits on test."""
import json
from experiments.mechanism import run_sampling_coordination_screen as r


def profile_metric(outputs, labels, profiles):
    n = len(labels)
    return {'correct': sum(r.outcome(outputs[cid], p) == labels[cid] for cid, p in profiles.items()), 'n': n,
            'tokens': sum(outputs[cid][f'{i}/{a}']['tokens'] for cid, p in profiles.items() for i, a in enumerate(p)),
            'calls': 3 * n,
            'sum_http_seconds': sum(outputs[cid][f'{i}/{a}']['wall_seconds'] for cid, p in profiles.items() for i, a in enumerate(p))}


def main():
    r.verify(); summary = {'new_analysis_calls': 0, 'gate': r.base.read(r.OUT / 'action_gate.json'),
        'status': r.base.read(r.OUT / 'screen_status.json'), 'splits': {}}
    policy = r.base.read(r.OUT / 'frozen_policy.json') if (r.OUT / 'frozen_policy.json').exists() else None
    for split in ('train', 'test'):
        if not (r.OUT / (split + '_sealed_outputs.json')).exists(): continue
        outputs = r.base.read(r.OUT / (split + '_sealed_outputs.json'))
        labels = r.base.read(r.OUT / (split + '_labels.json'))
        allfixed = {''.join(map(str, p)): profile_metric(outputs, labels, {cid: p for cid in labels}) for p in r.PROFILES}
        cells = {'all_fixed_profiles': allfixed, 'uniform_random_expected': {
            k: sum(v[k] for v in allfixed.values()) / 27 for k in ('correct', 'tokens', 'calls', 'sum_http_seconds')}}
        cells['individual_slots'] = {key: {'correct': sum(v[key]['answer'] == labels[c] for c, v in outputs.items()),
            'n': len(labels), 'calls': len(labels), 'tokens': sum(v[key]['tokens'] for v in outputs.values())}
            for key in [f'{i}/{a}' for i in range(3) for a in range(3)] + ['thinking']}
        cells['invalid_slots'] = sum(not v['valid'] for s in outputs.values() for v in s.values())
        cells['questions_with_multiple_report_answers'] = sum(len({s[f'{i}/{a}']['answer'] for i in range(3) for a in range(3)}) > 1 for s in outputs.values())
        if policy:
            cells['centralized_erm'] = allfixed[''.join(map(str, policy['centralized_erm']))]
            cells['train_selected_single'] = cells['individual_slots'][policy['train_selected_single']]
            weights = {}
            for p in r.PROFILES:
                weight = 1.
                for i, a in enumerate(p): weight *= policy['probabilities'][i][a]
                weights[''.join(map(str, p))] = weight
            cells['frozen_game_expected'] = {k: sum(weights[p] * v[k] for p, v in allfixed.items())
                                             for k in ('correct', 'tokens', 'calls', 'sum_http_seconds')}
            if split == 'test':
                profiles = {cid: r.base.read(r.OUT / split / r.sha256(cid.encode()).hexdigest() / 'selected_before_calls.json')['profile'] for cid in labels}
                cells['online_game_selected'] = profile_metric(outputs, labels, profiles)
        summary['splits'][split] = cells
    raw = [v for p in r.OUT.rglob('responses.jsonl') for v in r.read_rows(p)]
    usage = [json.loads(v['response']['body'])['usage'] for v in raw]
    summary.update(physical_calls=len(raw), physical_tokens=sum(v['total_tokens'] for v in usage),
        physical_prompt_tokens=sum(v['prompt_tokens'] for v in usage), physical_completion_tokens=sum(v['completion_tokens'] for v in usage),
        sum_http_seconds=sum(v['wall_seconds'] for v in raw), invocations=r.read_rows(r.OUT / 'invocations.jsonl'),
        scope='finite sampling menu feasibility, not ECON reproduction or game efficacy; expected controls reuse correlated slots')
    r.freeze(r.OUT / 'analysis_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'splits'}, ensure_ascii=False, indent=2))
    for split, cell in summary['splits'].items():
        print(split, json.dumps({'fixed_correct': {p: v['correct'] for p, v in cell['all_fixed_profiles'].items()},
            'slot_correct': {p: v['correct'] for p, v in cell['individual_slots'].items()},
            'invalid': cell['invalid_slots'], 'diverse_questions': cell['questions_with_multiple_report_answers']}, ensure_ascii=False))


if __name__ == '__main__': main()
