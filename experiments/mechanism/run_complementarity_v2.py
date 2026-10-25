"""Explicit pre-task engineering amendment: connectivity fixture uses an ok flag.

The v1 failure/raw response and protocol are preserved. No benchmark output was
collected before this amendment; the same preselected 36 cases are reused.
"""
import json
from pathlib import Path
from experiments.mechanism import study_information_complementarity as study
from experiments.mechanism.run_upgrade_pilot import freeze, read_rows

V1 = study.OUT
OUT = V1.with_name('information_complementarity_v2')


def prepare():
    rows=read_rows(V1/'responses.jsonl')
    if len(rows)!=1 or rows[0]['key']!='preflight/deepseek':
        raise ValueError('unexpected v1 execution history')
    provenance=json.loads((V1/'provenance.json').read_text())
    for name,h in provenance['files'].items():
        assert study.fingerprint(V1/name)==h
        freeze(OUT/name,json.loads((V1/name).read_text()))
    freeze(OUT/'provenance.json',provenance)
    freeze(OUT/'engineering_amendment.json',{
        'reason':'v1 HTTP200 fixture produced malformed JSON; no benchmark questions executed',
        'change':'connectivity checks ok boolean instead of answer/reasoning schema; task prompts and scoring unchanged',
        'old_preflight_response_hash':rows[0]['response_hash'],
        'same_cases_hash':study.p.digest(json.loads((OUT/'public_cases.json').read_text())),
        'wrapper_sha256':study.fingerprint(Path(__file__)),
        'old_protocol_sha256':study.fingerprint(V1/'protocol.json')})


def preflight(journal,models):
    for b in study.BACKENDS:
        req={'backend':b,'model':models[b],'temperature':0.,'max_tokens':100,
             'response_format':{'type':'json_object'},'messages':[{'role':'user','content':'Return exactly one JSON object: {"ok":true}.'}]}
        r=journal.call('preflight/'+b,req)
        body=json.loads(r['response']['body'])
        if r['response']['http_status']!=200 or body['choices'][0]['finish_reason']!='stop':
            raise RuntimeError('backend unavailable: '+b+' HTTP '+str(r['response']['http_status']))
        value=json.loads(body['choices'][0]['message']['content'])
        if value.get('ok') is not True:
            raise RuntimeError('backend connectivity fixture invalid: '+b)
        print('preflight',b,'returned',body.get('model'),flush=True)


if __name__=='__main__':
    study.OUT=OUT
    study.prepare=prepare
    study.preflight=preflight
    study.collect()
