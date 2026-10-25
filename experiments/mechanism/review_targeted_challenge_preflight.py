"""Offline-only review and dormant run manifest. No transport or model imports."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/results/resource_allocation/targeted_challenge_v1'
OUT = ROOT / 'experiments/results/audit/targeted_challenge_preflight_2026-10-25'

# Calls physically collected once per candidate. Final methods share dependencies.
STAGES = ('author', 'ordinary0', 'ordinary1', 'ordinary2', 'fixed_judge',
          'challenge', 'challenge_judge', 'restatement', 'reply',
          'nonadaptive', 'interactive')
DEPENDENCIES = {
    'ordinary': ('author', 'ordinary0', 'ordinary1', 'ordinary2'),
    'fixed': ('author', 'fixed_judge'),
    'challenge_only': ('author', 'challenge', 'challenge_judge'),
    'nonadaptive': ('author', 'challenge', 'restatement', 'nonadaptive'),
    'interactive': ('author', 'challenge', 'reply', 'interactive'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing artifact differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def public_payload(case, candidate, stage, history, windows):
    """Proposed interface, synthetic-testable; cannot send requests."""
    if stage not in STAGES or candidate not in 'ABCD':
        raise ValueError('Unregistered stage/candidate')
    allowed = {
        'author': (), 'ordinary0': ('author',),
        'ordinary1': ('author', 'ordinary0'),
        'ordinary2': ('author', 'ordinary0', 'ordinary1'),
        'fixed_judge': ('author',), 'challenge': ('author',),
        'challenge_judge': ('author', 'challenge'),
        'restatement': ('author',), 'reply': ('author', 'challenge'),
        'nonadaptive': ('author', 'challenge', 'restatement'),
        'interactive': ('author', 'challenge', 'reply'),
    }[stage]
    if set(history) != set(allowed):
        raise ValueError('History permissions mismatch')
    if set(case['choices']) != set('ABCD'):
        raise ValueError('Four native candidates required')
    if not all(isinstance(v, str) for v in history.values()):
        raise ValueError('History must contain substantive text only')
    for w in windows:
        if w['text'] != case['article'][w['start']:w['end']]:
            raise ValueError('Window is not an exact source slice')
    # Reject rather than silently forward extra label or diagnostic metadata.
    if set(case) != {'question', 'choices', 'article'}:
        raise ValueError('Unexpected case metadata')
    return {'question': case['question'], 'choices': dict(case['choices']),
            'registered_candidate': candidate, 'article': case['article'],
            'history': dict(history),
            'source_windows': [{k: w[k] for k in ('start', 'end', 'text')} for w in windows]}


def manifest():
    n = 8
    return {
        'status': 'NOT_APPROVED_NO_PAID_EXECUTION',
        'new_api_calls': 0, 'proposed_source_groups': n,
        'candidates_per_group': 4, 'physical_stages_per_candidate': list(STAGES),
        'native_calls_per_group': 1,
        'physical_calls_per_group': 1 + 4 * len(STAGES),
        'proposed_physical_call_limit': n * (1 + 4 * len(STAGES)),
        'max_output_tokens_per_call': 1000,
        'output_token_ceiling': n * (1 + 4 * len(STAGES)) * 1000,
        'method_dependencies': {k: list(v) for k, v in DEPENDENCIES.items()},
        'selected_path_calls_per_group': {'native': 1, **{k: 4 * len(v) for k, v in DEPENDENCIES.items()}},
        'model': 'deepseek-flash', 'temperature': 0.2, 'thinking': False,
        'qwen_allowed': False, 'transport_implemented': False,
        'required_before_start': [
            'user approval of protocol and monetary cap',
            'final prompts and data provenance frozen with hashes',
            'repository-wide previous-exposure scan and source-hash separation',
            'label-isolated selection, sealed labels and public request audit',
            'actual selected-input token estimate and current price verification',
            'backend fingerprint precondition, journal and no-automatic-retry checks',
            'synthetic full-DAG replay, scoring and failure-path tests',
        ],
        'scientific_gate': 'No observed positive increment; optional confound-resolution screen, not recommended automatic expansion',
        'scope': '8 source groups are exploratory, not a powered efficacy study; no game-learning or equilibrium claim',
    }


def coverage_review():
    """Posthoc diagnostic probes only; never retrieval rules or scoring labels."""
    cases = {c['id']: c for c in read(SOURCE / 'public_cases.json')}
    probes = (
        ('quality:0005fe38057f5ae5298dd9b6', 'D', 'Thursday'),
        ('quality:0005fe38057f5ae5298dd9b6', 'D', 'Refuse Removal'),
        ('quality:005370fa329e8e123696215d', 'A', 'group-mind'),
        ('quality:005370fa329e8e123696215d', 'A', 'Project Warp'),
        ('quality:009f922b625504ba2b95b83a', 'D', 'my interest was piqued'),
    )
    results = []
    for cid, candidate, term in probes:
        article = cases[cid]['article']
        windows = read(SOURCE / cid.replace(':', '_') / f'{candidate}_retrieval.json')['windows']
        matches = []
        for match in re.finditer(re.escape(term), article, re.I):
            lo, hi = max(0, match.start() - 140), min(len(article), match.end() + 220)
            matches.append({'start': match.start(), 'end': match.end(),
                            'covered': any(w['start'] <= match.start() and match.end() <= w['end'] for w in windows),
                            'context_start': lo, 'context_end': hi, 'context': article[lo:hi]})
        results.append({'id': cid, 'candidate': candidate, 'posthoc_term': term,
                        'matches': matches})
    value = {'status': 'posthoc_illustrative_not_exhaustive_semantic_coverage',
             'new_api_calls': 0, 'probes': results}
    write_once(OUT / 'source_coverage_review.json', value)
    return value


def audit():
    paths = [p for p in SOURCE.rglob('*') if p.is_file() and p.name != 'run.lock']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    accepted = read(SOURCE / 'acceptance.json')
    for rel, digest in accepted['sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f'Frozen evidence changed: {rel}')
    outputs = read(SOURCE / 'sealed_outputs.json')
    cases = {c['id']: c for c in read(SOURCE / 'public_cases.json')}
    labels = read(SOURCE / 'private_labels.json')
    changes = []
    token_max = []
    calls = 0
    for cid, result in outputs.items():
        directory = SOURCE / cid.replace(':', '_')
        for line in (directory / 'responses.jsonl').read_text().splitlines():
            row = json.loads(line)
            body = json.loads(row['response']['body'])
            token_max.append(body['usage']['prompt_tokens'])
            calls += 1
        for candidate, stages in result['claims'].items():
            if stages['nonadaptive']['verdict'] == stages['interactive']['verdict']:
                continue
            retrieval = read(directory / f'{candidate}_retrieval.json')
            c = cases[cid]
            for w in retrieval['windows']:
                assert w['text'] == c['article'][w['start']:w['end']]
            changes.append({
                'id': cid, 'candidate': candidate, 'official_best': labels[cid],
                'question': c['question'], 'choices': c['choices'],
                'direction': 'correction' if (stages['interactive']['verdict'] == 'accept') == (candidate == labels[cid]) else 'harm',
                'stages': {s: {k: stages[s][k] for k in ('argument', 'reason', 'challenge', 'query', 'verdict', 'citations') if k in stages[s]}
                           for s in ('author', 'challenge', 'restatement', 'reply', 'nonadaptive', 'interactive')},
                'retrieved_windows': retrieval['windows'],
                'source_article_sha256': c['article_sha256'],
                'interpretation_status': 'posthoc_trace_not_causal_attribution',
            })
    assert calls == 222 and len(changes) == 3
    assert sum(x['direction'] == 'correction' for x in changes) == 1
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    plan = manifest()
    # Historical envelope only: new input sizes may exceed this reference.
    plan['historical_max_prompt_tokens'] = max(token_max)
    plan['historical_max_input_envelope_not_bound'] = max(token_max) * plan['proposed_physical_call_limit']
    result = {'new_api_calls': 0, 'old_physical_calls_verified': calls,
              'changed_candidates': changes, 'preserved_source_files': before,
              'all_acceptance_hashes_verified': len(accepted['sha256']),
              'proposed_manifest': plan}
    write_once(OUT / 'trace_review.json', result)
    write_once(OUT / 'dormant_run_manifest.json', plan)
    coverage_review()
    print(json.dumps({'new_api_calls': 0, 'changed_candidates': len(changes),
                      'old_physical_calls_verified': calls,
                      'proposed_calls_NOT_started': plan['proposed_physical_call_limit'],
                      'historical_max_prompt_tokens': max(token_max)}, ensure_ascii=False))


if __name__ == '__main__':
    audit()
