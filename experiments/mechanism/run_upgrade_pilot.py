"""Bounded engineering pilot, not a trained policy or an efficacy experiment.

No labels accepted. Full raw responses persist before parsing. Unknown transport
outcomes stop automatic resume to avoid silently issuing a duplicate paid call.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

from experiments.core.mechanism.upgrade_game import Contract, digest, freeze_state, upgrade_request, validate_report
from experiments.core.mechanism.upgrade_execution import select, seal_selected


def utc():
    return datetime.now(timezone.utc).isoformat()


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path):
    if not path.exists():
        return []
    # Torn final lines fail closed, never truncate historical logs on resume.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timeline(directory, key, event, commitment):
    path = directory / 'timeline.jsonl'
    rows = read_rows(path)
    matching = [r for r in rows if r['key'] == key and r['event'] == event]
    if matching:
        if len(matching) != 1 or matching[0]['commitment'] != commitment:
            raise ValueError('timeline commitment mismatch')
        return
    append(path, {'sequence': len(rows), 'key': key, 'event': event,
                  'commitment': commitment, 'utc': utc()})


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError('frozen artifact mismatch: ' + path.name)
        return
    # Exclusive creation: incomplete files fail closed on the next resume.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class CallJournal:
    def __init__(self, directory, transport, max_calls=72):
        self.directory = directory
        self.transport = transport
        self.max_calls = max_calls

    def call(self, key, request):
        request_hash = digest(request)
        attempts = read_rows(self.directory / 'attempts.jsonl')
        responses = read_rows(self.directory / 'responses.jsonl')
        matching = [r for r in responses if r['key'] == key]
        if len(matching) > 1:
            raise ValueError('duplicate response key')
        if matching:
            row = matching[0]
            if row['request_hash'] != request_hash or row['request'] != request:
                raise ValueError('resume request mismatch')
            if row['response_hash'] != digest(row['response']):
                raise ValueError('raw response commitment mismatch')
            started = [a for a in attempts if a['key'] == key]
            if len(started) != 1 or started[0]['request_hash'] != request_hash:
                raise ValueError('response has no unique matching prior attempt')
            timeline(self.directory, key, 'response', row['response_hash'])
            return row
        if any(a['key'] == key for a in attempts):
            raise RuntimeError('unresolved attempt; reconcile manually before any retry: ' + key)
        if len(attempts) >= self.max_calls:
            raise RuntimeError('pilot HTTP attempt budget exhausted')
        append(self.directory / 'attempts.jsonl', {'key': key, 'request_hash': request_hash,
                                                  'started_utc': utc()})
        timeline(self.directory, key, 'attempt', request_hash)
        started = time.perf_counter()
        try:
            response = self.transport(request)
        except Exception as exc:
            append(self.directory / 'errors.jsonl', {'key': key, 'error_type': type(exc).__name__,
                   'wall_seconds': time.perf_counter() - started, 'utc': utc()})
            raise RuntimeError('transport failed; automatic retry disabled: ' + key) from None
        row = {'key': key, 'request': request, 'request_hash': request_hash,
               'response': response, 'response_hash': digest(response),
               'wall_seconds': time.perf_counter() - started, 'completed_utc': utc()}
        append(self.directory / 'responses.jsonl', row)
        timeline(self.directory, key, 'response', row['response_hash'])
        return row


def parse(row, choices):
    response = row['response']
    if response['http_status'] != 200:
        raise ValueError('HTTP failure preserved in raw response journal')
    body = json.loads(response['body'])
    choice = body['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('unfinished generation retained; no automatic repair')
    report = json.loads(choice['message']['content'])
    validate_report(report, choices)
    usage = {k: body.get('usage', {}).get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    if any(type(v) is not int or v < 0 for v in usage.values()):
        raise ValueError('provider usage missing or invalid')
    if usage['prompt_tokens'] + usage['completion_tokens'] != usage['total_tokens']:
        raise ValueError('token usage inconsistent')
    if not body.get('model'):
        raise ValueError('returned model metadata missing')
    return report, usage


def initial_request(case, agent):
    return {'messages': [{'role': 'system', 'content':
             'Answer independently using your knowledge. Do not invent external verification. '
             'Return one concise complete JSON object: C first (a registered choice or abstain), '
             'A and I strings, E a list of evidence strings. Stay within 400 output tokens.'},
             {'role': 'user', 'content': json.dumps({'question': case['question'],
                'choices': case['choices'], 'agent': agent}, sort_keys=True)}],
            'max_tokens': 600}


def public_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ValueError('engineering pilot accepts 1 to 12 cases')
    ids = set()
    for case in cases:
        if set(case) != {'id', 'question', 'choices', 'source_id'}:
            raise ValueError('public case schema: no labels or extra metadata')
        if not isinstance(case['id'], str) or case['id'] in ids:
            raise ValueError('unique case IDs required')
        ids.add(case['id'])
        freeze_state(case['question'], case['choices'], [{'A': '', 'E': [], 'I': '', 'C': 'abstain'}])


def engineering_contract():
    # Nonadditive synthetic table exercises selection and diagnostic paths.
    # NO calibration labels used; NO empirical quality meaning.
    return Contract((.2, .3, .3, .35, .5, .55, .55, .6), ((0, .02),) * 3,
                    1, 'engineering-fixture-v1-NOT-LEARNED-NOT-EFFICACY')


def run(cases, directory, transport, transport_identity):
    public_cases(cases)
    with lock(directory):
        contract = engineering_contract()
        root = Path(__file__).resolve().parents[2]
        code_paths = ['experiments/core/mechanism/upgrade_game.py',
                      'experiments/core/mechanism/upgrade_execution.py',
                      'experiments/mechanism/audit_upgrade_game.py',
                      'experiments/mechanism/run_upgrade_pilot.py']
        import hashlib
        protocol = {'version': 'upgrade-engineering-pilot-v1', 'case_hash': digest(cases),
                    'contract_hash': contract.commitment(), 'transport': transport_identity,
                    'max_http_attempts': 6 * len(cases), 'labels_permitted': False,
                    'purpose': 'engineering only; synthetic proxy is not a trained method',
                    'code_hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in code_paths}}
        freeze(directory / 'protocol.json', protocol)
        freeze(directory / 'public_cases.json', cases)
        calls = CallJournal(directory, transport, 6 * len(cases))
        seals = []
        for case in cases:
            prefix = digest(case)[:20]
            initial = []
            for i in range(3):
                raw = calls.call(prefix + '/initial/' + str(i), initial_request(case, i))
                report, _ = parse(raw, case['choices'])
                initial.append(report)
            frozen = freeze_state(case['question'], case['choices'], initial)
            freeze(directory / (prefix + '.state.json'), json.loads(frozen))
            timeline(directory, prefix, 'state', digest(json.loads(frozen)))
            selection = select(frozen, contract)
            freeze(directory / (prefix + '.selection.json'), selection)
            timeline(directory, prefix, 'selection', digest(selection))
            events = [selection]
            for i, action in enumerate(selection['actions']):
                if not action:
                    continue
                request = upgrade_request(frozen, i)
                raw = calls.call(prefix + '/upgrade/' + str(i), request)
                report, usage = parse(raw, case['choices'])
                events.append({'type': 'selected_upgrade', 'agent': i,
                               'request_hash': request['request_hash'], 'response_hash': raw['response_hash'],
                               'report': report, 'finish_reason': 'stop', 'usage': usage})
            seal = seal_selected(frozen, contract, events)
            freeze(directory / (prefix + '.events.json'), events)
            freeze(directory / (prefix + '.sealed.json'), seal)
            timeline(directory, prefix, 'seal', digest(seal))
            # Only after final output is durably sealed, collect missing potentials.
            for i, action in enumerate(selection['actions']):
                if action:
                    continue
                raw = calls.call(prefix + '/diagnostic/' + str(i), upgrade_request(frozen, i))
                parse(raw, case['choices'])
            seals.append({'case_id': case['id'], 'output_hash': seal['output_hash']})
        rows = read_rows(directory / 'responses.jsonl')
        models = sorted({json.loads(r['response']['body'])['model'] for r in rows})
        if len(models) != 1:
            raise ValueError('model version changed within pilot; investigate before acceptance')
        usage = [json.loads(r['response']['body'])['usage'] for r in rows]
        phase_costs = {}
        for phase in ('initial', 'upgrade', 'diagnostic'):
            phase_rows = [r for r in rows if r['key'].split('/')[1] == phase]
            phase_costs[phase] = {
                'calls': len(phase_rows),
                'tokens': sum(json.loads(r['response']['body'])['usage']['total_tokens'] for r in phase_rows),
                'summed_call_wall_seconds': sum(r['wall_seconds'] for r in phase_rows)}
        summary = {'purpose': protocol['purpose'], 'cases': len(cases), 'responses': len(rows),
                   'returned_models': models, 'seals': seals,
                   'total_tokens': sum(u['total_tokens'] for u in usage),
                   'phase_costs': phase_costs,
                   'deployed_calls': phase_costs['initial']['calls'] + phase_costs['upgrade']['calls'],
                   'diagnostic_calls': phase_costs['diagnostic']['calls'],
                   'summed_call_wall_seconds': sum(r['wall_seconds'] for r in rows),
                   'status': 'engineering_completed_not_efficacy_validated'}
        freeze(directory / 'summary.json', summary)
        return summary


def live_transport():
    # Configuration import only. No legacy DNS monkeypatch / insecure TLS fallback.
    from experiments.core.llm import LLM_BACKENDS
    import requests
    config = LLM_BACKENDS['deepseek']
    if not config['api_key']:
        raise ValueError('DeepSeek API key unavailable')
    model = config['model']
    base = config['base_url'].rstrip('/')
    url = base + '/chat/completions' if base.endswith(('/v1', '/v2', '/v3', '/v4')) else base + '/v1/chat/completions'
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + config['api_key']})

    def transport(request):
        payload = {'model': model, 'messages': request['messages'], 'max_tokens': request['max_tokens'],
                   'temperature': .3, 'response_format': {'type': 'json_object'}}
        response = session.post(url, json=payload, timeout=(10, 45))
        return {'http_status': response.status_code, 'body': response.text,
                'requested_model': model, 'received_utc': utc()}

    return transport, {'provider': 'configured-deepseek', 'model': model, 'temperature': .3,
                       'endpoint_hash': digest(url), 'tls_verify': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, help='mandatory screened provenance for live execution')
    parser.add_argument('--execute', action='store_true', help='explicit bounded paid-call opt-in')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    public_cases(cases)
    if not args.execute:
        print(json.dumps({'status': 'preflight_only_no_API', 'cases': len(cases), 'max_attempts': len(cases) * 6}))
    else:
        if args.provenance is None:
            raise ValueError('live pilot requires screened source/history provenance')
        provenance = json.loads(args.provenance.read_text())
        if provenance.get('public_cases_hash') != digest(cases) or provenance.get('n') != len(cases):
            raise ValueError('public cases do not match screened provenance')
        if not provenance.get('history_scan_hash') or not provenance.get('source_sha256'):
            raise ValueError('missing source or history commitment')
        args.output.mkdir(parents=True, exist_ok=True)
        freeze(args.output / 'source_provenance.json', provenance)
        transport, identity = live_transport()
        print(json.dumps(run(cases, args.output, transport, identity)))
