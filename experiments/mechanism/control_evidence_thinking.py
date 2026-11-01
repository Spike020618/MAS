"""Same API model, stronger thinking-mode controls on opened source snapshots."""
import argparse
import copy
import json
from pathlib import Path
from experiments.mechanism import replicate_evidence_acquisition as r
from experiments.mechanism import attribute_acquisition_paths as a
from experiments.mechanism import control_answer_order as order
from experiments.mechanism import repeat_open_evidence_solver as repeat
from experiments.mechanism import probe_acquired_evidence_pool as pool
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows,digest

p=r.p;ROOT=r.ROOT;OUT=ROOT/'experiments/results/redesign_v1/evidence_thinking_control_v1'
DESIGN=ROOT/'docs/research/redesign_v1/推理模式_强控制协议_2026-10-25.md'


def thinking(req):
    changed=copy.deepcopy(req);changed['thinking']={'type':'enabled'};changed['max_tokens']=4096;changed.pop('temperature',None)
    return changed


def prepare():
    r.verify();OUT.mkdir(parents=True,exist_ok=True);requests={}
    for case in p.base.read(r.OUT/'public_cases.json'):
        material=p.base.read(r.OUT/case['id']/'material.json')
        requests[case['id']]={arm:thinking(p.request(case,'native',material[arm])) for arm in ('native','expanded')}
    freeze(OUT/'requests.json',requests)
    freeze(OUT/'protocol.json',{'n':16,'max_calls':32,'not_equal_compute':True,
        'sha256':{str(f.relative_to(ROOT)):p.base.h(f) for f in (Path(__file__),DESIGN,OUT/'requests.json')}})


def verify():
    r.verify()
    for path,value in p.base.read(OUT/'protocol.json')['sha256'].items():
        if p.base.h(ROOT/path)!=value:raise ValueError('thinking protocol changed:'+path)


def collect(transport):
    verify();outputs={}
    with lock(OUT):
        journal=CallJournal(OUT,transport,32)
        if {x['key'] for x in read_rows(OUT/'attempts.jsonl')}-{x['key'] for x in read_rows(OUT/'responses.jsonl')}:raise RuntimeError('unresolved attempt')
        for cid,arms in p.base.read(OUT/'requests.json').items():
            outputs[cid]={}
            for arm,req in sorted(arms.items(),key=lambda v:digest([cid,v[0],'thinking-control'])):
                match=next((x for x in read_rows(OUT/'responses.jsonl') if x['request']==req),None)
                if match is None:
                    paths=list(p.OUT.rglob('responses.jsonl'))+[repeat.OUT/'responses.jsonl',pool.OUT/'responses.jsonl']+list(r.OUT.rglob('responses.jsonl'))+list(a.OUT.rglob('responses.jsonl'))+[order.OUT/'responses.jsonl',OUT/'responses.jsonl']
                    spent=sum((json.loads(x['response']['body'])['usage']['prompt_tokens']*.3+json.loads(x['response']['body'])['usage']['completion_tokens']*1.2)/1e6 for path in paths for x in read_rows(path))
                    reserve=((len(p.base.encoded(req))+4096)*.3+req['max_tokens']*1.2)/1e6
                    if spent+reserve>5:raise RuntimeError('cumulative cap exhausted')
                row=journal.call(match['key'] if match else cid+'/'+arm,req)
                docs=json.loads(req['messages'][1]['content'])['documents'];outputs[cid][arm]=p.parse(row,'native',docs)
            print('thinking source control sealed',len(outputs),'/16',flush=True)
        freeze(OUT/'sealed_outputs.json',outputs)


def analyze():
    verify();outputs=p.base.read(OUT/'sealed_outputs.json');labels=p.base.read(r.OUT/'private_labels.json')
    if set(outputs)!=set(labels):raise ValueError('incomplete thinking seal')
    metrics={}
    for arm in ('native','expanded'):
        scores=[p.answer_score(outputs[c][arm]['answer'],labels[c]) for c in labels]
        slots=[outputs[c][arm] for c in labels]
        metrics[arm]={'em_correct':sum(v['em'] for v in scores),'mean_f1':sum(v['f1'] for v in scores)/16,
            'invalid':sum(not v['valid'] for v in slots),'logical_tokens':sum(v['usage']['total_tokens'] for v in slots)}
    corrected=[c for c in labels if p.answer_score(outputs[c]['expanded']['answer'],labels[c])['em'] and not p.answer_score(outputs[c]['native']['answer'],labels[c])['em']]
    harmed=[c for c in labels if not p.answer_score(outputs[c]['expanded']['answer'],labels[c])['em'] and p.answer_score(outputs[c]['native']['answer'],labels[c])['em']]
    raw=read_rows(OUT/'responses.jsonl')
    result={'scope':'opened same-model thinking control, not equal compute or independent test','n':16,
        'metrics':metrics,'corrected':corrected,'harmed':harmed,'physical_calls':len(raw),
        'physical_tokens':sum(json.loads(x['response']['body'])['usage']['total_tokens'] for x in raw),
        'sum_http_seconds':sum(x['wall_seconds'] for x in raw),
        'usd_proxy':sum((json.loads(x['response']['body'])['usage']['prompt_tokens']*.3+json.loads(x['response']['body'])['usage']['completion_tokens']*1.2)/1e6 for x in raw)}
    freeze(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','analyze','replay'));stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='analyze':analyze()
    elif stage=='replay':
        def blocked(req):raise AssertionError('replay API')
        collect(blocked);analyze()
    else:
        from experiments.mechanism.study_explicit_thinking_control import transport
        collect(transport);analyze()
