"""Mock historical client semantics; measurement limits, not real API results."""
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from types import SimpleNamespace

from experiments.core.llm import LLMClient
from experiments.mechanism.attribute_legacy_gain import ROOT, OUT


def client_probe():
    captured = []
    def post(url, json, timeout):
        captured.append(json)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {'model': 'offline-returned-model', 'id': 'offline-response-id',
                'system_fingerprint': 'offline-fingerprint', 'created': 123,
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3},
                'choices': [{'message': {'content': '{"C":"yes"}'}, 'finish_reason': 'stop'}]})
    client = LLMClient(api_key='offline-fixture-not-a-secret', base_url='https://offline.invalid/v1',
                       model='offline-requested-model', backend='deepseek')
    client._session = SimpleNamespace(post=post, close=lambda: None)
    client.chat([{'role': 'user', 'content': 'offline test'}], seed=99)
    return {'transport_fully_mocked': True, 'payload_fields': sorted(captured[0]),
        'seed_forwarded_even_if_explicitly_supplied': 'seed' in captured[0],
        'client_model_after_different_returned_model': client.model,
        'response_model_retained_as_client_model': client.model == 'offline-returned-model',
        'returned_response_id_retained': hasattr(client, 'last_response_id'),
        'scope': 'current client mock only; no real model/version experiment'}


def analyze():
    source = ROOT/'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl'
    groups = defaultdict(Counter)
    trace_keys, row_keys = Counter(), Counter()
    for line in source.read_text().splitlines():
        r = json.loads(line); c = r['condition']
        groups[c]['records'] += 1
        groups[c]['retried_records'] += r.get('attempt_count',1)>1
        groups[c]['prior_error_entries'] += len(r.get('prior_errors',[]))
        for key in r: row_keys[key] += 1
        for t in r['rounds']:
            for call in t['action_trace']:
                for key in call: trace_keys[key] += 1
    collect_path = ROOT/'experiments/mechanism/collect.py'
    llm_path = ROOT/'experiments/core/llm.py'
    tree = ast.parse(collect_path.read_text())
    generate = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='generate_one')
    call_keywords = sorted({kw.arg for node in ast.walk(generate) if isinstance(node,ast.Call)
                            for kw in node.keywords if kw.arg})
    return {'no_api_calls': True, 'client_probe': client_probe(),
        'formal_record_metadata': {c:dict(v) for c,v in groups.items()},
        'observed_trace_keys': dict(trace_keys), 'observed_record_keys': dict(row_keys),
        'generate_one_keyword_argument_names': call_keywords,
        'conclusions': {
            'historical_seed_names_do_not_establish_controlled_API_random_seeds': True,
            'logged_model_alias_does_not_identify_server_returned_weights': True,
            'completed_trace_call_counts_are_not_all_network_attempts_or_all_billed_calls': True,
            'sum_call_latency_is_not_parallel_end_to_end_wall_time': True,
            'old_absolute_execution_UTC_timing_not_present_in_formal_records': not any(
                k in trace_keys or k in row_keys for k in ['started_at','finished_at','created','timestamp'])},
        'scope': 'current client semantics and actual frozen record schema; no reconstruction of missing historical transport retries',
        'inputs_sha256': {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                          for p in [source,collect_path,llm_path]}}


def main():
    r = analyze()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'measurement.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['inputs_sha256','observed_record_keys']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
