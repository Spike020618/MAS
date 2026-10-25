"""Versioned output-integrity repair. V1 artifacts and scorer remain unchanged.

Same worlds: this is a development regression, NOT a fresh held-out evaluation.
No content retries, partial JSON recovery, or answer-dependent configuration.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from experiments.mechanism import pilot_abstract_conflicts as v1
from experiments.mechanism.validate_locked_source_game import Caller, freeze, valid_report

LIMIT = 1600
BOUNDS = (' Output exactly one complete JSON object without markdown. Put C first. '
          'Keep each A/E/I string at most 180 characters; use at most two citations, '
          'each quote at most 120 characters. Keep each dispute string at most 120 characters. '
          'These are output-length limits, not changes to the reasoning rules.')


def report_status(row, structured=False):
    if row.get('finish_reason') != 'stop':
        return 'incomplete_generation'
    if not valid_report(row):
        return 'invalid_aeic'
    citations = row['parsed'].get('citations')
    if not isinstance(citations, list) or any(not isinstance(c, dict) or
            not isinstance(c.get('source_id'), str) or not isinstance(c.get('quote'), str)
            for c in citations):
        return 'invalid_citations_schema'
    if structured and not v1.valid_dispute(row):
        return 'invalid_dispute_schema'
    return 'valid'


def vote(row, structured=False):
    return row['parsed']['C'] if report_status(row, structured) == 'valid' else None


def consensus(votes):
    """None is a failed report, never an epistemic abstention or a tie-break."""
    return next((c for c in ['yes', 'no', 'abstain'] if votes.count(c) >= 2), None)


def packet(row, sources):
    status = report_status(row)
    return {'status': status, 'report': row['parsed'] if status == 'valid' else None,
            'source_packets': sources}


def setup(directory):
    cases, private = v1.generate()
    protocol = {'version': 'abstract-conflict-smoke-v2', 'expected_calls': 108,
        'scope': 'same-world development regression; not independent heldout evidence',
        'max_tokens': LIMIT, 'no_content_retries': True,
        'initial_prompt': v1.INITIAL + BOUNDS,
        'revision_prompts': {'ordinary': v1.ORDINARY + BOUNDS, 'structured': v1.STRUCTURED + BOUNDS},
        'aggregation': 'two valid explicit equal votes; otherwise failure, including gold abstain',
        'failure_denominator': 'all planned questions; invalid consensus is incorrect',
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'v1_dependency_sha256': hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest(),
        'generator_sha256': v1.digest(cases), 'private_score_sha256': v1.digest(private)}
    for name, value in [('protocol', protocol), ('public_cases', cases), ('private_scoring', private)]:
        freeze(directory / f'{name}.json', value)
    return protocol, cases


def execute(directory, case, protocol, caller):
    qid = case['question_id']
    out = directory / 'runs' / f'{qid}.json'
    if out.exists():
        return json.loads(out.read_text())
    paid = {hashlib.sha256(d['text'].encode()).hexdigest(): d for d in case['paid_documents']}
    ids = list(paid)
    bundles = {'0': ids[:1], '1': ids[1:2], '2': ids}
    game = v1.SourceSharingGame(['0', '1', '2'], bundles,
        {s: 1 / len(ids) for s in ids}, {str(i): [0, .75] for i in range(3)}, 1)
    decision = game.run_local_updates()
    profile = decision['terminal']['profile']
    freeze(directory / 'commitments' / f'{qid}.json',
        {'profile': profile, 'events': decision['events'], 'protocol_sha256': v1.digest(protocol)})
    ledger = v1.AcquisitionLedger(qid, game, {s: d['text'].encode() for s, d in paid.items()})
    initial, own_sources = [], []
    for i in range(3):
        sources = [case['local_documents'][i]]
        if profile[i]:
            for sid in bundles[str(i)]:
                ledger.acquire(str(i), sid)
                sources.append(paid[sid])
        body = {'question': case['question'], 'rules': case['rules'], 'sources': sources}
        initial.append(caller.call(f'{qid}/initial/{i}', [
            {'role': 'system', 'content': protocol['initial_prompt']},
            {'role': 'user', 'content': json.dumps(body, sort_keys=True)}], max_tokens=LIMIT))
        own_sources.append(sources)
    freeze(directory / 'initial_rounds' / f'{qid}.json',
        {'response_keys': [r['key'] for r in initial], 'response_sha256': [v1.digest(r['response']) for r in initial]})
    branches = {}
    for name in sorted(protocol['revision_prompts'], key=lambda n: hashlib.sha256((qid+n).encode()).hexdigest()):
        revised = []
        for i in range(3):
            body = {'question': case['question'], 'rules': case['rules'],
                'own': packet(initial[i], own_sources[i]),
                'peers': [{'agent_id': str(j), **packet(initial[j], own_sources[j])} for j in range(3) if j != i]}
            revised.append(caller.call(f'{qid}/{name}/{i}', [
                {'role': 'system', 'content': protocol['revision_prompts'][name]},
                {'role': 'user', 'content': json.dumps(body, sort_keys=True)}], max_tokens=LIMIT))
        votes = [vote(r, name == 'structured') for r in revised]
        sources = {s['source_id']: s for own in own_sources for s in own}
        branches[name] = {'keys': [r['key'] for r in revised], 'votes': votes,
            'consensus': consensus(votes),
            'statuses': [report_status(r, name == 'structured') for r in revised],
            'citation_checks': [v1.validate_citations(r, list(sources.values())) for r in revised],
            'tokens_including_shared_initial': sum((r.get('usage') or {}).get('total_tokens', 0) for r in initial + revised)}
    result = {'question_id': qid, 'initial_votes': [vote(r) for r in initial],
        'initial_statuses': [report_status(r) for r in initial], 'branches': branches,
        'resource_regrets': game.regrets(profile), 'settlement': ledger.settle(profile)}
    freeze(out, result)
    return result


def analyze(directory):
    private = json.loads((directory / 'private_scoring.json').read_text())
    summary = {}
    for split in ['calibration', 'heldout_seed']:
        subset = [s for s in private if s['split'] == split]
        summary[split] = {}
        for name in ['ordinary', 'structured']:
            pairs = [(s, json.loads((directory / 'runs' / f'{s["question_id"]}.json').read_text())['branches'][name]) for s in subset]
            summary[split][name] = {'n': len(pairs),
                'correct': sum(b['consensus'] == s['ground_truth'] for s, b in pairs),
                'failed_consensus': sum(b['consensus'] is None for _, b in pairs),
                'invalid_reports': sum(status != 'valid' for _, b in pairs for status in b['statuses'])}
    result = {'scope': 'same-world development regression, not fresh heldout evaluation', 'summary': summary}
    freeze(directory / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.out_dir / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, cases = setup(args.out_dir)
    if not args.collect:
        print('Protocol frozen; no API calls. Collection requires --collect (108 calls).')
        return
    for name in ['runs', 'commitments', 'initial_rounds']:
        (args.out_dir / name).mkdir(exist_ok=True)
    caller = Caller(args.out_dir)
    for case in cases:
        execute(args.out_dir, case, protocol, caller)
    if len(caller.calls) != 108:
        raise ValueError('call count mismatch')
    print(json.dumps(analyze(args.out_dir), indent=2))


if __name__ == '__main__':
    main()
