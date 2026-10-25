"""Post-seal accounting; no transport or model calls."""
import json
from pathlib import Path
from experiments.mechanism import run_full_context_challenge_pilot2 as pilot
from experiments.mechanism import run_full_context_challenge as parent
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows


def main():
    parent.verify(parent.OUT)
    result = parent.read(pilot.OUT / 'result.json')
    records = result['records']
    raw = [r for p in pilot.OUT.rglob('responses.jsonl') for r in read_rows(p)]
    attempts = [r for p in pilot.OUT.rglob('attempts.jsonl') for r in read_rows(p)]
    assert len(raw) == len(attempts) == 90
    usage = [json.loads(r['response']['body'])['usage'] for r in raw]
    changes = {}
    def correct(row, method):
        return row[method] in ('accept', 'reject') and (row[method] == 'accept') == bool(row['truth'])
    for baseline in ('native', 'ordinary0', 'ordinary2', 'fixed_judge', 'challenge_judge', 'nonadaptive'):
        changes[baseline] = {
            'corrected': sum(correct(r, 'interactive') and not correct(r, baseline) for r in records),
            'harmed': sum(not correct(r, 'interactive') and correct(r, baseline) for r in records),
            'changed_candidates': [{'id': r['id'], 'candidate': r['candidate'],
                                    'baseline': r[baseline], 'interactive': r['interactive']}
                                   for r in records if r[baseline] != r['interactive']]}
    summary = {'independent_source_groups': 2, 'candidate_claims': 8,
        'physical_attempts': len(attempts), 'physical_responses': len(raw),
        'prompt_tokens': sum(v['prompt_tokens'] for v in usage),
        'completion_tokens': sum(v['completion_tokens'] for v in usage),
        'total_tokens': sum(v['total_tokens'] for v in usage),
        'sum_http_seconds': sum(r['wall_seconds'] for r in raw),
        'peak_all_cache_miss_usd_upper_rate_proxy': sum((v['prompt_tokens'] * .3 + v['completion_tokens'] * 1.2) / 1e6 for v in usage),
        'invocations': read_rows(pilot.OUT / 'invocations.jsonl'),
        'metrics': result['metrics'], 'interactive_vs': changes,
        'scientific_scope': 'two source groups only, exploratory; no significance, generality or equilibrium claim',
        'automatic_expansion': False,
        'models': sorted({json.loads(r['response']['body'])['model'] for r in raw}),
        'fingerprints': sorted({json.loads(r['response']['body'])['system_fingerprint'] for r in raw})}
    freeze(pilot.OUT / 'pilot_summary.json', summary)
    paths = [p for p in pilot.OUT.rglob('*') if p.is_file() and p.name not in ('run.lock', 'pilot_acceptance.json')]
    paths += [Path(__file__), Path(pilot.__file__)]
    freeze(pilot.OUT / 'pilot_acceptance.json', {'status': 'two_source_pilot_completed_no_expansion',
        'physical_calls': 90, 'sha256': {str(p.relative_to(parent.ROOT)): parent.h(p) for p in paths}})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
