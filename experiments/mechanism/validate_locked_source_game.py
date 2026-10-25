"""Prospective, locked-contract online check for explicit utility agents.

No test-label-based choice or autonomous LLM rationality claim. The LLM executes
reports only. Source acquisitions and settlement occur within each online run.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import quote

import requests

from experiments.core.llm import LLMClient
from experiments.core.mechanism.source_sharing import AcquisitionLedger, SourceSharingGame
from experiments.mechanism.pilot_external_evidence import ParagraphParser
from experiments.mechanism.pilot_source_sharing import digest, journal
from experiments.mechanism.pilot_source_sharing_neutral import prepare as neutral_prepare

ROOT = Path(__file__).resolve().parents[2]
DATA_URL = 'https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json'
QUERY_PROMPT = ('Return JSON with queries: a list of exactly two short encyclopedia entity/topic titles. '
                'Each query must contain at most four words. Cover the main entities independently. '
                'Avoid question wording, long conjunctions and complete claims. Do not answer or judge truth.')


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'frozen artifact mismatch: {path.name}')
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def walk_questions(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'question' and isinstance(item, str):
                yield canonical(item)
            elif isinstance(item, (dict, list)):
                yield from walk_questions(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_questions(item)


def audit_history():
    seen, manifest = set(), []
    for path in sorted((ROOT/'experiments/results').rglob('*')):
        if not path.is_file() or path.suffix not in ('.json', '.jsonl'):
            continue
        data = path.read_bytes()
        manifest.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest()})
        try:
            if path.suffix == '.json':
                seen.update(walk_questions(json.loads(data)))
            else:
                for line in data.splitlines():
                    if line.strip(): seen.update(walk_questions(json.loads(line)))
        except (ValueError, UnicodeError):
            raise ValueError(f'history parse failure: {path}') from None
    return seen, manifest


class Caller:
    def __init__(self, directory):
        self.directory = directory
        self.path = directory/'calls.jsonl'
        self.calls = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if row['key'] in self.calls: raise ValueError('duplicate successful job')
                self.calls[row['key']] = row
        self.client = None

    def call(self, key, messages, max_tokens=700):
        prompt_hash = digest(messages)
        if key in self.calls:
            row = self.calls[key]
            if row['prompt_sha256'] != prompt_hash or row['max_tokens'] != max_tokens:
                raise ValueError('resume prompt mismatch')
            return row
        if self.client is None:
            self.client = LLMClient(backend='deepseek', max_tokens=700)
            if self.client.model != 'deepseek-chat': raise ValueError('model alias changed')
        payload = {'model': self.client.model, 'messages': messages, 'temperature': .3,
                   'max_tokens': max_tokens, 'response_format': {'type': 'json_object'}}
        base = self.client.base_url.rstrip('/')
        url = base+'/chat/completions' if base.endswith(('/v1', '/v2', '/v3', '/v4')) else base+'/v1/chat/completions'
        started, response = time.perf_counter(), None
        journal(self.directory/'attempts.jsonl', {'key': key, 'event': 'request_started', 'utc': utc()})
        try:
            response = self.client.session.post(url, json=payload, timeout=45)
            response.raise_for_status()
            data = response.json()
            choice = data['choices'][0]
            content = choice['message'].get('content') or ''
            try: parsed = json.loads(content)
            except ValueError: parsed = None
            row = {'key': key, 'messages': messages, 'prompt_sha256': prompt_hash,
                   'max_tokens': max_tokens, 'response': content, 'parsed': parsed,
                   'usage': data.get('usage'), 'returned_model': data.get('model'),
                   'requested_model': self.client.model, 'response_id': data.get('id'),
                   'fingerprint': data.get('system_fingerprint'), 'finish_reason': choice.get('finish_reason'),
                   'wall_seconds': time.perf_counter()-started, 'utc': utc()}
            journal(self.path, row)
            self.calls[key] = row
            print(f'API {len(self.calls)} {key}', flush=True)
            return row
        except Exception as exc:
            journal(self.directory/'errors.jsonl', {'key': key, 'error_type': type(exc).__name__,
                    'http_status': response.status_code if response is not None else None,
                    'wall_seconds': time.perf_counter()-started, 'utc': utc()})
            raise RuntimeError(f'stopped safely: {key}: {type(exc).__name__}') from None


def lock_protocol(directory, n):
    analysis_path = ROOT/'experiments/results/mechanism/stackelberg/source_sharing_pilot_v3/analysis.json'
    analysis = json.loads(analysis_path.read_text())
    choice = next(r for r in analysis['leader_sensitivity']
                  if r['token_weight'] == .01 and r['payment_weight'] == .01)
    if choice['model_optimal_budgets'] != [1]: raise ValueError('development selection changed')
    _, report_jobs = neutral_prepare()
    prompts = [next(j['messages'][0]['content'] for j in report_jobs if j['agent'] == i) for i in range(3)]
    files = [Path(__file__), ROOT/'experiments/core/mechanism/source_sharing.py',
             ROOT/'experiments/mechanism/pilot_source_sharing.py',
             ROOT/'experiments/mechanism/pilot_source_sharing_neutral.py',
             ROOT/'experiments/mechanism/pilot_external_evidence.py']
    protocol = {'version': 'locked-source-game-online-v1', 'n_questions': n,
        'source': DATA_URL, 'sampling_seed': 'MAS-prospective-2026-10-19-v1',
        'selected_budget': 1, 'comparison_budget': 0, 'budget_grid': [0, .5, 1, 2, 4],
        'development_analysis_sha256': hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        'leader_weights': {'tokens_per_1000': .01, 'payment': .01},
        'selection_scope': 'exploratory development model, locked before prospective evaluation',
        'agents': ['0', '1', '2'], 'costs': {str(i): [0, .75] for i in range(3)},
        'cost_semantics': 'explicit-agent resource-account units, not actual API costs',
        'initial': [0, 0, 0], 'order': ['0', '1', '2'],
        'actor': 'explicit local utility policy; LLM produces reports, not choices',
        'query_prompt': QUERY_PROMPT, 'report_prompts': prompts, 'temperature': .3,
        'max_report_tokens': 700, 'source_failure_policy': 'keep question, empty bank or available subset, no replacement',
        'source_extraction': 'v2 paragraph parser; query-keyword ranking, four paragraphs, 1300 chars each',
        'n_source_slots': 2, 'within_question_condition_order': 'sha256(question_id) parity',
        'primary': ['online_execution_integrity', 'terminal_unilateral_regret'],
        'secondary': ['paired_team_correctness', 'report_tokens', 'payment', 'latency'],
        'quality_claim': 'small prospective check, not powered superiority study',
        'script_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    freeze(directory/'protocol.json', protocol)
    return protocol


def prepare_cases(directory, protocol):
    manifest_path = directory/'cases.json'
    if manifest_path.exists():
        cases = json.loads(manifest_path.read_text())
        if len(cases) != protocol['n_questions']: raise ValueError('case count changed')
        return cases
    # Snapshot the old records before downloading or selecting any new question.
    history_path = directory/'history_audit.json'
    if history_path.exists():
        seen_hashes = set(json.loads(history_path.read_text())['canonical_question_sha256'])
    else:
        seen, history = audit_history()
        seen_hashes = {hashlib.sha256(q.encode()).hexdigest() for q in seen}
        freeze(history_path, {'scope': 'all historical result JSON/JSONL question fields; not model pretraining audit',
            'n_unique_questions': len(seen), 'files': history,
            'canonicalization': 'NFKC casefold alphanumeric', 'canonical_question_sha256': sorted(seen_hashes)})
    raw_path = directory/'official_train.json'
    if not raw_path.exists():
        response = requests.get(DATA_URL, timeout=45)
        response.raise_for_status()
        raw_path.write_bytes(response.content)
    data = json.loads(raw_path.read_text())
    provenance_path = directory/'dataset_provenance.json'
    if not provenance_path.exists():
        freeze(provenance_path, {'url': DATA_URL, 'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            'n_examples': len(data), 'utc': utc(), 'ignored_fields': ['term', 'description', 'facts', 'decomposition', 'evidence', 'answer']})
    if json.loads(provenance_path.read_text())['sha256'] != hashlib.sha256(raw_path.read_bytes()).hexdigest():
        raise ValueError('official dataset changed')
    public = [{'question_id': 'StrategyQA-official/'+r['qid'], 'question': r['question']}
              for r in data if hashlib.sha256(canonical(r['question']).encode()).hexdigest() not in seen_hashes]
    public.sort(key=lambda r: hashlib.sha256((protocol['sampling_seed']+canonical(r['question'])).encode()).hexdigest())
    unique, chosen = set(), []
    for row in public:
        key = canonical(row['question'])
        if key not in unique:
            chosen.append(row); unique.add(key)
        if len(chosen) == protocol['n_questions']: break
    if len(chosen) != protocol['n_questions']: raise ValueError('insufficient unexposed questions')
    freeze(manifest_path, chosen)
    freeze(directory/'exposure_certificate.json', {'n_selected': len(chosen), 'prior_canonical_overlap': 0,
        'cases_sha256': digest(chosen), 'historical_audit_sha256': hashlib.sha256((directory/'history_audit.json').read_bytes()).hexdigest(),
        'limitation': 'repository result records only; public benchmark may be in model pretraining'})
    return chosen


def retrieve(directory, case, caller):
    cache = directory/'banks'/f"{case['question_id'].split('/')[-1]}.json"
    if cache.exists(): return json.loads(cache.read_text())
    query = caller.call(case['question_id']+'/query', [
        {'role': 'system', 'content': QUERY_PROMPT}, {'role': 'user', 'content': case['question']}], 100)
    query_object = query.get('parsed')
    queries = query_object.get('queries', []) if isinstance(query_object, dict) else []
    if not isinstance(queries, list): queries = []
    queries = [q for q in queries[:2] if isinstance(q, str)]
    errors, sources, http = [], {}, []
    started = time.perf_counter()
    session = requests.Session()
    session.headers['User-Agent'] = 'MASResearchValidation/1.0 (academic reproducibility)'
    last = 0.

    def get(url, **kwargs):
        nonlocal last
        delay = 4-(time.monotonic()-last)
        if delay > 0: time.sleep(delay)
        last = time.monotonic()
        response = session.get(url, timeout=25, **kwargs)
        http.append({'url': response.url, 'status': response.status_code,
                     'sha256': hashlib.sha256(response.content).hexdigest()})
        response.raise_for_status()
        return response

    for text in queries[:2]:
        if not isinstance(text, str) or len(text.split()) > 4:
            errors.append('invalid query schema'); continue
        try:
            response = get('https://en.wikipedia.org/w/rest.php/v1/search/page', params={'q': text, 'limit': 1})
            pages = response.json().get('pages', [])[:1]
            for page in pages:
                content = get('https://en.wikipedia.org/w/rest.php/v1/page/'+quote(page['key'], safe='')+'/html')
                source_hash = hashlib.sha256(content.content).hexdigest()
                (directory/'snapshots'/f'{source_hash}.html').write_bytes(content.content)
                parser = ParagraphParser(); parser.feed(content.text)
                terms = set(re.findall(r'[a-z]{3,}', ' '.join(queries).lower()))
                ranked = sorted(enumerate(p for p in parser.paragraphs if len(p)>60),
                    key=lambda pair: (-sum(t in pair[1].lower() for t in terms), pair[0]))[:4]
                excerpt = '\n'.join(value[:1300] for _, value in ranked)
                if excerpt:
                    sources[source_hash] = {'source_hash': source_hash, 'title': page['title'],
                        'url': 'https://en.wikipedia.org/wiki/'+quote(page['key'], safe=''), 'text': excerpt,
                        'excerpt_sha256': hashlib.sha256(excerpt.encode()).hexdigest()}
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(type(exc).__name__)
    row = {'question_id': case['question_id'], 'sources': list(sources.values()), 'errors': errors,
           'http': http, 'wall_seconds': time.perf_counter()-started, 'utc': utc()}
    freeze(cache, row)
    return row


def valid_report(row):
    report = row.get('parsed')
    return (isinstance(report, dict) and report.get('C') in ['yes', 'no', 'abstain']
            and all(isinstance(report.get(k), str) for k in ['A', 'E', 'I']))


def run_case(directory, protocol, case, bank, caller, budget):
    qid = case['question_id']
    key = f'{qid}/B{budget}'
    run_path = directory/'runs'/f"{qid.split('/')[-1]}_B{budget}.json"
    source_map = {s['source_hash']: s for s in bank['sources']}
    for source in source_map.values():
        if hashlib.sha256(source['text'].encode()).hexdigest() != source['excerpt_sha256']:
            raise ValueError('source excerpt changed')
    ids = list(source_map)  # Frozen retrieval rank, not hash/order selected from answers.
    bundles = {'0': ids[:1], '1': ids[1:2], '2': ids}
    game = SourceSharingGame(protocol['agents'], bundles,
        {s: 1/len(ids) for s in ids} if ids else {}, protocol['costs'], budget)
    decision = game.run_local_updates(protocol['initial'], protocol['order'])
    commitment = {'request_id': key, 'question_id': qid, 'budget': budget,
        'profile': decision['terminal']['profile'], 'events': decision['events'],
        'bundles': bundles, 'protocol_sha256': digest(protocol), 'bank_sha256': digest(bank)}
    commitment_path = directory/'commitments'/f"{qid.split('/')[-1]}_B{budget}.json"
    # Commitment exists and is fsynced before report API calls; labels are not loaded.
    if not commitment_path.exists():
        freeze(commitment_path, commitment)
        with commitment_path.open('r') as handle:
            import os
            os.fsync(handle.fileno())
    else: freeze(commitment_path, commitment)
    if run_path.exists():
        result = json.loads(run_path.read_text())
        if result['commitment_sha256'] != digest(commitment): raise ValueError('run commitment changed')
        return result
    started = time.perf_counter()
    data = {s: (directory/'snapshots'/f'{s}.html').read_bytes() for s in ids}
    ledger = AcquisitionLedger(key, game, data)
    reports = []
    resumed = False
    for i, agent in enumerate(protocol['agents']):
        selected = []
        if commitment['profile'][i]:
            for source in bundles[agent]:
                ledger.acquire(agent, source)
                selected.append(source_map[source])
        messages = [{'role': 'system', 'content': protocol['report_prompts'][i]},
                    {'role': 'user', 'content': json.dumps({'question': case['question'], 'sources': selected}, ensure_ascii=False)}]
        resumed = resumed or key+'/'+agent in caller.calls
        response = caller.call(key+'/'+agent, messages)
        reports.append(response)
    settlement = ledger.settle(commitment['profile'])
    votes = [r['parsed']['C'] if valid_report(r) else 'abstain' for r in reports]
    count = Counter(votes)
    stance = next((s for s in ['yes', 'no'] if count[s] >= 2), 'abstain')
    equilibria = {tuple(row['profile']) for row in game.equilibria()}
    passed = (tuple(commitment['profile']) in equilibria and max(game.regrets(commitment['profile']))<=1e-10
              and settlement['paid']<=budget+1e-10)
    result = {'request_id': key, 'commitment_sha256': digest(commitment), 'decision': decision,
        'settlement': settlement, 'terminal_regrets': game.regrets(commitment['profile']),
        'equilibrium_check_passed': passed, 'stance': stance,
        'report_keys': [r['key'] for r in reports], 'schema_failures': sum(not valid_report(r) for r in reports),
        'report_tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in reports),
        'report_wall_sum': sum(r['wall_seconds'] for r in reports),
        'completion_segment_wall_seconds': time.perf_counter()-started,
        'resumed_reports': resumed,
        'utc': utc()}
    freeze(run_path, result)
    return result


def score(directory, protocol, cases, caller):
    raw = json.loads((directory/'official_train.json').read_text())
    gold = {'StrategyQA-official/'+r['qid']: 'yes' if r['answer'] else 'no' for r in raw}
    rows = []
    for case in cases:
        qid = case['question_id']
        pair = []
        for budget in [0, 1]:
            path = directory/'runs'/f"{qid.split('/')[-1]}_B{budget}.json"
            if not path.exists(): raise ValueError('cannot score incomplete run')
            run = json.loads(path.read_text())
            pair.append({**run, 'correct': run['stance'] == gold[qid]})
        rows.append({'question_id': qid, 'no_credit': pair[0], 'locked_contract': pair[1]})
    n = len(rows)
    from math import comb
    improved = sum(not r['no_credit']['correct'] and r['locked_contract']['correct'] for r in rows)
    worsened = sum(r['no_credit']['correct'] and not r['locked_contract']['correct'] for r in rows)
    discordant = improved+worsened
    pvalue = min(1., 2*sum(comb(discordant, k) for k in range(min(improved, worsened)+1))/2**discordant) if discordant else 1.
    models = sorted({str(r.get('returned_model')) for r in caller.calls.values()})
    result = {'scope': 'prospective locked-contract online validation of explicit utility agents; no autonomous LLM equilibrium claim',
        'n_questions': n, 'n_online_runs': 2*n, 'successful_llm_calls': len(caller.calls),
        'all_online_equilibrium_checks_passed': all(r[c]['equilibrium_check_passed'] for r in rows for c in ['no_credit', 'locked_contract']),
        'max_terminal_regret': max(max(r[c]['terminal_regrets']) for r in rows for c in ['no_credit', 'locked_contract']),
        'accuracy': {c: sum(r[c]['correct'] for r in rows)/n for c in ['no_credit', 'locked_contract']},
        'mean_report_tokens': {c: sum(r[c]['report_tokens'] for r in rows)/n for c in ['no_credit', 'locked_contract']},
        'paired': {'improved': improved, 'worsened': worsened, 'exact_mcnemar_p': pvalue},
        'schema_failures': sum(r[c]['schema_failures'] for r in rows for c in ['no_credit', 'locked_contract']),
        'returned_models': models, 'truncated_calls': sum(r['finish_reason']=='length' for r in caller.calls.values()),
        'missing_usage': sum(not r.get('usage') for r in caller.calls.values()),
        'actual_tokens_all_calls': sum((r.get('usage') or {}).get('total_tokens',0) for r in caller.calls.values()),
        'actual_request_seconds_sum': sum(r['wall_seconds'] for r in caller.calls.values()),
        'cold_source_seconds_sum': sum(json.loads(p.read_text())['wall_seconds'] for p in (directory/'banks').glob('*.json')),
        'contract_retuned_on_test': False, 'real_resource_reward_account': False,
        'claim_boundary': 'finite development-estimated contract optimum; executed follower equilibrium under explicit fees; not a real-distribution global optimum',
        'rows': rows}
    freeze(directory/'final_evaluation.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--n-questions', type=int, default=24)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    directory = args.out_dir
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ['banks','snapshots','runs','commitments']: (directory/name).mkdir(exist_ok=True)
    protocol = lock_protocol(directory, args.n_questions)
    print('Protocol locked; explicit agents only', flush=True)
    if not args.run: return
    cases = prepare_cases(directory, protocol)
    caller = Caller(directory)
    for i, case in enumerate(cases):
        bank = retrieve(directory, case, caller)
        budgets = [0, 1] if int(hashlib.sha256(case['question_id'].encode()).hexdigest(),16)%2 == 0 else [1, 0]
        for budget in budgets: run_case(directory, protocol, case, bank, caller, budget)
        print(f'Completed paired online question {i+1}/{len(cases)} sources={len(bank["sources"])}', flush=True)
    score(directory, protocol, cases, caller)


if __name__ == '__main__': main()
