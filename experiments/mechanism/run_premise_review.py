"""Bounded 24-case native-QA pilot, sealed labels and resumable raw journal."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from experiments.mechanism import premise_review as method
from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows

OUT = p.ROOT / 'experiments/results/resource_allocation/premise_review_v1'


def prepare():
    from experiments.mechanism.prepare_routing_holdout import PINS
    from experiments.mechanism.prepare_upgrade_p2 import pool, native_mc
    from experiments.mechanism.prepare_upgrade_pilot import canonical, strings
    from experiments.mechanism.build_upgrade_history import build
    if (OUT / 'provenance.json').exists():
        prov = json.loads((OUT / 'provenance.json').read_text())
        for name, digest in prov['files'].items():
            assert sha256((OUT / name).read_bytes()).hexdigest() == digest
        return
    dependency = Path('/private/tmp/mas_p2_dependencies')
    if dependency.is_dir():
        sys.path.insert(0, str(dependency))
    args = SimpleNamespace(
        strategyqa=p.ROOT/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json',
        truthfulqa=p.ROOT/'experiments/data/raw/upgrade_p2_v1/truthfulqa_mc_authors.json',
        mmlu=p.ROOT/'experiments/data/raw/upgrade_p2_v1/mmlu_test.parquet')
    for domain, expected in PINS.items():
        assert sha256(getattr(args, domain).read_bytes()).hexdigest() == expected
    mmlu = args.mmlu
    args.mmlu = None
    candidates, sources = pool(args)
    import pyarrow.parquet as parquet
    for row in parquet.read_table(mmlu).to_pylist():
        if len(set(row['choices'])) == len(row['choices']):
            candidates.append(native_mc('mmlu', p.digest([row['subject'], row['question']])[:24],
                                        row['question'], list(row['choices']), int(row['answer'])))
    history = build(p.ROOT)
    if history['errors']:
        raise ValueError('historical parse errors remain')
    used = {canonical(v['text']) for v in history['texts'].values()}
    ranked = []
    for domain in ('strategyqa', 'mmlu', 'truthfulqa'):
        seen = set()
        for r in sorted((r for r in candidates if r['domain'] == domain),
                        key=lambda r: p.digest(['premise-review-2026-10-24-v1', r['source_id']])):
            stem = canonical(r['base_question'])
            if stem in seen or stem in used:
                continue
            seen.add(stem)
            ranked.append(r)
            if len(seen) == 80:
                break
    blocked, coverage = {}, []
    for file in history['files']:
        if file.get('role') == 'verified_official_raw_pool_not_model_execution' or file.get('duplicate_of'):
            continue
        path = p.ROOT / file['path']
        raw = path.read_bytes()
        assert sha256(raw).hexdigest() == file['sha256']
        data = [json.loads(s) for s in raw.decode().splitlines() if s.strip()] if path.suffix == '.jsonl' else json.loads(raw)
        leaves = list(strings(data))
        ids, content = set(leaves), '\n'.join(canonical(s) for s in leaves)
        for r in ranked:
            key = r['domain'] + ':' + r['source_id']
            if key not in blocked and (r['source_id'] in ids or canonical(r['base_question']) in content):
                blocked[key] = file['path']
        coverage.append({'path': file['path'], 'sha256': file['sha256']})
    selected = []
    for domain in ('strategyqa', 'mmlu', 'truthfulqa'):
        available = [r for r in ranked if r['domain'] == domain and r['domain'] + ':' + r['source_id'] not in blocked]
        if len(available) < 8:
            raise ValueError('insufficient unused cases: ' + domain)
        selected.extend(available[:8])
    assert len({canonical(r['base_question']) for r in selected}) == 24
    public = [{'id': r['domain'] + ':' + r['source_id'], 'source_id': r['source_id'],
               'question': r['question'], 'choices': r['choices']} for r in selected]
    public.sort(key=lambda r: p.digest(['premise-execution-order-v1', r['id']]))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in [('public_cases.json', public),
                        ('private_labels.json', [{'id': r['domain'] + ':' + r['source_id'], 'label': r['label']} for r in selected]),
                        ('history_snapshot.json', history), ('backstop_coverage.json', coverage)]:
        freeze(OUT / name, value)
    freeze(OUT / 'provenance.json', {'n': 24, 'per_domain': 8, 'source_pins': PINS,
           'history_files': len(history['files']), 'backstop_files': len(coverage), 'collisions': blocked,
           'selection': 'input-only hash order; exact normalized source-stem/ID isolation, not semantic novelty',
           'files': {n: sha256((OUT/n).read_bytes()).hexdigest() for n in
                     ('public_cases.json', 'private_labels.json', 'history_snapshot.json', 'backstop_coverage.json')}})


def initial_request(case, agent, model):
    return {'backend': 'deepseek', 'model': model, 'temperature': .2, 'max_tokens': 800,
            'response_format': {'type': 'json_object'}, 'messages': [
                {'role': 'system', 'content': 'Answer independently and carefully using available knowledge. '
                 'Reason sufficiently, respect the original question scope and answer criterion. '
                 'No external tools or sources have been accessed. Return JSON with answer '
                 '(one registered choice or abstain) and reasoning (text).'},
                {'role': 'user', 'content': json.dumps({'question': case['question'], 'choices': case['choices'],
                                                     'replicate': agent}, sort_keys=True)}]}


def parse_row(row, choices):
    response = row['response']
    try:
        value = json.loads(response['body'])
    except ValueError:
        value = {}
    usage = value.get('usage', {})
    known = all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    valid_http = response['http_status'] == 200
    choice = value.get('choices', [{}])[0]
    raw = choice.get('message', {}).get('content') or ''
    obs = method.decode(raw, choices)
    complete = valid_http and choice.get('finish_reason') == 'stop' and obs['valid']
    return {'raw': raw, 'answer': obs['answer'] if complete else 'abstain',
            'reasoning': obs['reasoning'] if complete else 'Unavailable: incomplete or invalid initial output.',
            'valid': complete, 'tokens': usage['total_tokens'] if known else None,
            'finish_reason': choice.get('finish_reason'), 'model': value.get('model'),
            'http_status': response['http_status'], 'wall_seconds': row['wall_seconds'], 'calls': 1}


def vote(reports):
    c = Counter(r['answer'] for r in reports if r['answer'] != 'abstain')
    return c.most_common(1)[0][0] if c and c.most_common(1)[0][1] >= 2 else 'abstain'


def execute():
    from experiments.core.llm import LLM_BACKENDS
    cfg = LLM_BACKENDS['deepseek']
    if not cfg['api_key']:
        raise ValueError('DeepSeek API configuration missing')
    with lock(OUT):
        prepare()
        cases = json.loads((OUT/'public_cases.json').read_text())
        protocol = {'version': 'premise-review-v1', 'cases_hash': p.digest(cases), 'model': cfg['model'],
                    'base_url': cfg['base_url'], 'max_attempts': 240, 'successful_planned_calls': 120,
                    'temperature': .2, 'output_cap': 800, 'initial_replicates': 3,
                    'retry': 'at most one identical retry for completed HTTP429/5xx only; transport ambiguity stops',
                    'failed_or_truncated': 'retained; no selective prompt repair; missing cost unknown',
                    'scope': '24-case exploratory screen, no equilibrium or confirmatory efficacy claim',
                    'code_sha256': {str(f.relative_to(p.ROOT)): sha256(f.read_bytes()).hexdigest()
                                    for f in [Path(__file__), Path(method.__file__)]}}
        freeze(OUT/'protocol.json', protocol)
        journal = CallJournal(OUT, p.transport, 240)
        def call(key, req, choices):
            row = journal.call(key, req)
            if row['response']['http_status'] == 429 or row['response']['http_status'] >= 500:
                row = journal.call(key + '/retry1', req)
            if row['response']['http_status'] in (401, 402, 403):
                raise RuntimeError('API authorization/balance error; raw response saved')
            return parse_row(row, choices)
        # Seal all independent outputs before exposing any reports to reviewers.
        initial = {}
        for case in cases:
            initial[case['id']] = [call(case['id']+'/initial/'+str(i), initial_request(case, i, cfg['model']), case['choices']) for i in range(3)]
            print('initial saved', len(initial), '/24', flush=True)
        freeze(OUT/'sealed_initial.json', initial)
        reviews = {}
        for case in cases:
            reports = [{'answer': r['answer'], 'reasoning': r['reasoning']} for r in initial[case['id']]]
            reviews[case['id']] = {}
            for arm in sorted(method.ARMS, key=lambda a: p.digest([case['id'], a])):
                req = {'backend': 'deepseek', **method.request(case['question'], case['choices'], reports, arm, cfg['model'])}
                reviews[case['id']][arm] = call(case['id']+'/review/'+arm, req, case['choices'])
            print('paired reviews saved', len(reviews), '/24', flush=True)
        freeze(OUT/'sealed_reviews.json', reviews)
        # Labels are read only after both review arms are sealed.
        labels = {r['id']: r['label'] for r in json.loads((OUT/'private_labels.json').read_text())}
        records = []
        for case in cases:
            init = initial[case['id']]
            record = {'id': case['id'], 'choices': case['choices'], 'gold': labels[case['id']],
                      'initial_answer': vote(init), 'initial_calls': 3,
                      'initial_tokens': sum(r['tokens'] for r in init) if all(r['tokens'] is not None for r in init) else None,
                      'reviews': {a: {**r, 'raw': r['raw'] if r['valid'] else ''} for a, r in reviews[case['id']].items()}}
            records.append(record)
        freeze(OUT/'scoring_records.json', records)
        costs_known = all(r['initial_tokens'] is not None and all(v['tokens'] is not None for v in r['reviews'].values()) for r in records)
        summary = method.paired_summary(records) if costs_known else {'cost_unknown': True,
            'n': 24, 'correct': {a: sum(reviews[c['id']][a]['answer'] == labels[c['id']] for c in cases) for a in method.ARMS}}
        raw_rows = read_rows(OUT/'responses.jsonl')
        summary.update({'actual_http_attempts': len(read_rows(OUT/'attempts.jsonl')),
                        'returned_models': sorted({r['model'] for rs in initial.values() for r in rs if r['model']} |
                                                  {r['model'] for rs in reviews.values() for r in rs.values() if r['model']}),
                        'initial_total_tokens': sum(r['initial_tokens'] for r in records) if costs_known else None,
                        'initial_correct': sum(r['initial_answer'] == r['gold'] for r in records),
                        'sum_http_wall_seconds_all_attempts': sum(r['wall_seconds'] for r in raw_rows),
                        'serial_path_wall_seconds': {a: sum(sum(r['wall_seconds'] for r in initial[c['id']]) + reviews[c['id']][a]['wall_seconds'] for c in cases) for a in method.ARMS},
                        'response_journal_hash': p.digest(raw_rows), 'no_automatic_expansion': True})
        freeze(OUT/'result.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        execute()
    else:
        with lock(OUT):
            prepare()
        print('24 source-screened cases prepared; no API calls')
