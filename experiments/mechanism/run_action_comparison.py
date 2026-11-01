"""Bounded paired action collector. No gold loading or scoring in this module.

Default CLI verifies only. Paid execution requires --execute-paid and a separately
sealed same-day official-price commitment. Unknown transport outcomes fail closed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from experiments.core.mechanism import action_target_interface as interface
from experiments.core.mechanism.evidence_acquisition import Corpus, Ledger
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows, digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/results/action_comparison_v1'
INDEX = ROOT / 'experiments/results/redesign_v1/retrieval_env/wiki_fts.sqlite'
FP = 'e72d3c7f663d3e1383b00b3884cb4501'
ARMS = ('retrieve', 'check', 'review', 'resample')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(docs):
    unique, origins = {}, {}
    for d in docs:
        if sha_text(d['text']) != d['text_sha256']:
            raise ValueError('source text hash mismatch')
        key = 'source:' + digest([d['title'], d['text']])
        unique[key] = {**{k: d[k] for k in ('title', 'text', 'text_sha256')}, 'docid': key}
        origins.setdefault(key, set()).add(d['docid'])
    ordered = [unique[k] for k in sorted(unique)]
    if sum(len(d['text'].encode()) for d in ordered) > 40000:
        raise ValueError('whole documents over 40KB; no truncation')
    return ordered, {k: sorted(v) for k, v in origins.items()}


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def public(cases):
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('unique public cases required')
    for c in cases:
        if set(c) != {'id', 'question', 'native_documents'}:
            raise ValueError('private or unknown case fields')
        if not isinstance(c['id'], str) or not c['id'].isalnum():
            raise ValueError('unsafe case ID')
        for d in c['native_documents']:
            if set(d) != {'docid', 'title', 'text', 'text_sha256'}:
                raise ValueError('private source fields')
        canonical(c['native_documents'])


def freeze_protocol():
    cases = read(OUT / 'public_cases.json')
    public(cases)
    if len(cases) != 24:
        raise ValueError('24 fixed cases required')
    files = [Path(__file__), ROOT / 'experiments/mechanism/prepare_action_comparison.py',
        ROOT / 'experiments/mechanism/score_action_comparison.py',
        ROOT / 'experiments/tests/test_action_comparison.py',
        ROOT / 'experiments/tests/test_action_targets.py',
        ROOT / 'experiments/core/mechanism/action_target_interface.py',
        ROOT / 'experiments/core/mechanism/evidence_acquisition.py',
        ROOT / 'experiments/mechanism/run_upgrade_pilot.py',
        ROOT / 'docs/research/action_targets_v1/三行动验证方案.md',
        OUT / 'public_cases.json', OUT / 'splits.json', OUT / 'selection.json',
        INDEX.parent / 'index_manifest.json']
    freeze(OUT / 'protocol.json', {'n': 24, 'maximum_physical_calls': 192,
        'model': interface.MODEL, 'fingerprint': FP, 'usd_proxy_cap': 2.0,
        'concurrency': 1, 'version': 'action-comparison-v1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'no_training': True, 'sha256': {str(p.relative_to(ROOT)): sha(p) for p in files}})


def verify():
    protocol = read(OUT / 'protocol.json')
    for rel, value in protocol['sha256'].items():
        if sha(ROOT / rel) != value:
            raise ValueError('frozen artifact changed: ' + rel)
    public(read(OUT / 'public_cases.json'))
    return protocol


def valid_prices(prices, today=None):
    today = today or datetime.now(timezone.utc).date().isoformat()
    if prices.get('checked_utc_date') != today:
        raise ValueError('same-day price review required')
    if not prices.get('official_url', '').startswith('https://api-docs.deepseek.com/'):
        raise ValueError('official pricing source required')
    for key in ('input_usd_per_million', 'output_usd_per_million'):
        value = prices.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('finite positive rate required')


def usage(row):
    if row['response']['http_status'] != 200:
        raise RuntimeError('HTTP error preserved; cost unknown, automatic resume blocked')
    body = json.loads(row['response']['body'])
    if (body.get('model'), body.get('system_fingerprint')) != (interface.MODEL, FP):
        raise RuntimeError('backend identity changed')
    u = body['usage']
    if any(type(u.get(k)) is not int or u[k] < 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
        raise RuntimeError('invalid usage')
    if u['prompt_tokens'] + u['completion_tokens'] != u['total_tokens']:
        raise RuntimeError('inconsistent usage')
    return u


def cost(row, prices):
    u = usage(row)
    return (u['prompt_tokens'] * prices['input_usd_per_million'] +
            u['completion_tokens'] * prices['output_usd_per_million']) / 1e6


def parse(row, kind, docs):
    u = usage(row)
    result = {'valid': False, 'answer': 'invalid', 'reasoning': '', 'docids': [], 'usage': u}
    try:
        ch = json.loads(row['response']['body'])['choices'][0]
        if ch['finish_reason'] != 'stop':
            return result
        value = json.loads(ch['message']['content'])
        if not isinstance(value, dict):
            return result
        if kind == 'final':
            valid = (set(value) == {'answer', 'reasoning', 'docids'} and isinstance(value['answer'], str)
                     and bool(value['answer'].strip()) and isinstance(value['reasoning'], str)
                     and isinstance(value['docids'], list)
                     and all(isinstance(k, str) and k in {d['docid'] for d in docs} for k in value['docids']))
        elif kind == 'retrieve':
            valid = set(value) == {'query'} and isinstance(value['query'], str) and 0 < len(value['query'].strip()) <= 200
        elif kind == 'review':
            valid = set(value) == {'note'} and isinstance(value['note'], str) and 0 < len(value['note'].split()) <= 120
        else:
            valid = interface.check_finding(value, docs)
        if valid:
            result.update(value, valid=True)
    except (ValueError, TypeError, KeyError):
        pass
    return result


def ledger_rows(out):
    rows = []
    folders = {p.parent for p in out.glob('*/attempts.jsonl')} | {p.parent for p in out.glob('*/responses.jsonl')}
    for folder in sorted(folders):
        path = folder / 'attempts.jsonl'
        attempts, responses = read_rows(path), read_rows(path.with_name('responses.jsonl'))
        if len({r['key'] for r in responses}) != len(responses):
            raise ValueError('duplicate physical responses')
        if {a['key'] for a in attempts} != {r['key'] for r in responses} or len(attempts) != len(responses):
            raise RuntimeError('unknown pending outcome; no automatic retries')
        for row in responses:
            if digest(row['request']) != row['request_hash'] or digest(row['response']) != row['response_hash']:
                raise ValueError('journal altered')
            match = [a for a in attempts if a['key'] == row['key']]
            if len(match) != 1 or match[0]['request_hash'] != row['request_hash']:
                raise ValueError('invalid attempt linkage')
            usage(row)
        rows.extend(responses)
    return rows


def collect(case, out, transport, prices, corpus_factory, cap=2.0, maximum_calls=192):
    public([case])
    folder = out / case['id']
    with lock(folder):
        journal = CallJournal(folder, transport, 8)
        def call(key, req, docs, kind):
            all_rows = ledger_rows(out)
            existing = read_rows(folder / 'responses.jsonl')
            identical = next((r for r in existing if r['request'] == req), None)
            if identical is not None:
                row = journal.call(identical['key'], req)
            else:
                if len(all_rows) >= maximum_calls:
                    raise RuntimeError('global request cap exhausted')
                input_bound = len(json.dumps(req, ensure_ascii=False).encode()) + 4096
                reserve = (input_bound * prices['input_usd_per_million'] + req['max_tokens'] * prices['output_usd_per_million']) / 1e6
                if sum(cost(r, prices) for r in all_rows) + reserve > cap:
                    raise RuntimeError('budget reservation would exceed cap')
                row = journal.call(key, req)
                u = usage(row)
                if u['prompt_tokens'] > input_bound or u['completion_tokens'] > req['max_tokens']:
                    raise RuntimeError('usage exceeds reservation; stop')
            return {'source_key': row['key'], 'parsed': parse(row, kind, docs)}
        docs, origins = canonical(case['native_documents'])
        initial = call('initial', interface.final_request(case, docs), docs, 'final')
        initial_input = {k: initial['parsed'][k] for k in ('answer', 'reasoning', 'docids')}
        arms = {}
        for arm in sorted(ARMS, key=lambda x: digest([case['id'], x, 'action-order-v1'])):
            current, note, action, retrieval = docs, None, None, None
            if arm != 'resample':
                action = call(arm + '/action', interface.action_request(arm, case, docs, initial_input), docs, arm)
                parsed = action['parsed']
                if parsed['valid']:
                    if arm == 'retrieve':
                        path = folder / 'retrieval.json'
                        if path.exists():
                            retrieval = read(path)
                            if retrieval['query'] != parsed['query']:
                                raise ValueError('retrieval query changed on resume')
                        else:
                            corpus = corpus_factory()
                            try:
                                ledger = Ledger(max_searches=1, max_unique_bytes=40000)
                                ledger.acquire(corpus, parsed['query'], 2)
                                retrieval = {'query': parsed['query'], 'documents': ledger.public_documents(),
                                             'event': ledger.searches[0]}
                                freeze(path, retrieval)
                            finally:
                                corpus.close()
                        try:
                            if retrieval['event']['skipped_budget_ids']:
                                raise ValueError('retrieved source over cap')
                            current, _ = canonical(docs + retrieval['documents'])
                        except ValueError:
                            arms[arm] = {'action': action, 'retrieval': retrieval, 'final': None,
                                         'failure': 'whole_source_over_cap'}
                            continue
                        note = {'query': parsed['query']}
                    elif arm == 'check':
                        note = {k: parsed[k] for k in ('status', 'document_id', 'quote', 'note')}
                    else:
                        note = {'note': parsed['note']}
            final = call(arm + '/final', interface.final_request(case, current, initial_input, note), current, 'final')
            arms[arm] = {'action': action, 'retrieval': retrieval, 'final': final,
                         'documents': current, 'invalid_action_note_dropped': action is not None and not action['parsed']['valid']}
        sealed = {'id': case['id'], 'initial': initial, 'initial_documents': docs,
                  'origins': origins, 'arms': arms}
        freeze(folder / 'sealed_outputs.json', sealed)
        return sealed


def replay_case(case, output, rows):
    """Rebuild every allowed request and parse from journal, without network."""
    raw = {row['key']: row for row in rows}
    docs, origins = canonical(case['native_documents'])
    if output['initial_documents'] != docs or output['origins'] != origins:
        raise ValueError('initial material changed')
    def check(saved, req, material, kind):
        row = raw[saved['source_key']]
        if row['request'] != req or saved['parsed'] != parse(row, kind, material):
            raise ValueError('request permissions/parsed output changed')
    check(output['initial'], interface.final_request(case, docs), docs, 'final')
    initial = {k: output['initial']['parsed'][k] for k in ('answer', 'reasoning', 'docids')}
    for arm in ARMS:
        branch = output['arms'][arm]
        current, note = docs, None
        action = branch['action']
        if arm != 'resample':
            check(action, interface.action_request(arm, case, docs, initial), docs, arm)
            parsed = action['parsed']
            if parsed['valid']:
                if arm == 'retrieve':
                    env = branch['retrieval']
                    if env['query'] != parsed['query']:
                        raise ValueError('source query changed')
                    try:
                        if env['event']['skipped_budget_ids']:
                            raise ValueError('retrieval over cap')
                        current, _ = canonical(docs + env['documents'])
                    except ValueError:
                        if branch['final'] is None and branch['failure'] == 'whole_source_over_cap':
                            continue
                        raise
                    note = {'query': parsed['query']}
                elif arm == 'check':
                    note = {k: parsed[k] for k in ('status', 'document_id', 'quote', 'note')}
                else:
                    note = {'note': parsed['note']}
        elif action is not None:
            raise ValueError('resample must not acquire a note')
        if branch['documents'] != current:
            raise ValueError('branch material changed')
        check(branch['final'], interface.final_request(case, current, initial, note), current, 'final')


def run():
    protocol = verify()
    prices = read(OUT / 'prices.json')
    valid_prices(prices)
    from urllib.parse import urlparse
    from experiments.core.llm import LLM_BACKENDS
    cfg = LLM_BACKENDS['deepseek']
    if urlparse(cfg['base_url']).hostname != 'api.deepseek.com' or not cfg['api_key']:
        raise RuntimeError('official endpoint/API configuration required')
    from experiments.mechanism.study_explicit_thinking_control import transport
    with lock(OUT):
        started = time.perf_counter()
        freeze(OUT / 'price_commitment.json', {'sha256': sha(OUT / 'prices.json'), 'prices': prices})
        ledger_rows(OUT)
        outputs = {}
        for case in read(OUT / 'public_cases.json'):
            outputs[case['id']] = collect(case, OUT, transport, prices, lambda: Corpus(INDEX),
                cap=protocol['usd_proxy_cap'], maximum_calls=protocol['maximum_physical_calls'])
            print('sealed', len(outputs), '/24', flush=True)
        freeze(OUT / 'sealed_outputs.json', outputs)
        rows = ledger_rows(OUT)
        receipt = {'calls': len(rows), 'tokens': sum(usage(r)['total_tokens'] for r in rows),
            'price_commitment_sha256': sha(OUT / 'prices.json'), 'usd_proxy': sum(cost(r, prices) for r in rows),
            'not_fresh_full_e2e_if_resumed': True,
            'sealed_outputs_sha256': sha(OUT / 'sealed_outputs.json')}
        freeze(OUT / 'collection_receipt.json', receipt)
        from experiments.mechanism.run_upgrade_pilot import append
        append(OUT / 'invocations.jsonl', {'elapsed_seconds': time.perf_counter() - started, 'completed': len(outputs)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze_protocol()
    elif args.execute_paid:
        run()
    else:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
