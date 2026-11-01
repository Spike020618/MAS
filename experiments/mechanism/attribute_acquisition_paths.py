"""Complete 3-path input interventions on opened development questions."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from experiments.mechanism import replicate_evidence_acquisition as r
from experiments.mechanism import repeat_open_evidence_solver as repeat
from experiments.mechanism import probe_acquired_evidence_pool as pool
from experiments.mechanism.audit_observed_evidence_shapley import shapley
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows,digest

p=r.p;ROOT=r.ROOT;OUT=ROOT/'experiments/results/redesign_v1/acquisition_path_attribution_v1'
DESIGN=ROOT/'docs/research/redesign_v1/取证路径贡献_开发归因协议_2026-10-25.md'


def prepare():
    r.verify();OUT.mkdir(parents=True,exist_ok=True);requests={};materials={}
    for c in p.base.read(r.OUT/'public_cases.json'):
        env=p.base.read(r.OUT/c['id']/'query_environments.json');nxt=p.base.read(r.OUT/c['id']/'next_environment.json')
        packets=[env[0]['documents'],env[1]['documents']+env[2]['documents'],env[0]['documents']+nxt['documents']]
        requests[c['id']]={};materials[c['id']]={}
        for mask in range(8):
            docs,_=r.canonical(c['native_documents']+[d for i,packet in enumerate(packets) if mask&(1<<i) for d in packet])
            requests[c['id']][str(mask)]=p.request(c,'native',docs);materials[c['id']][str(mask)]=[d['docid'] for d in docs]
        old=p.base.read(r.OUT/c['id']/'material.json')
        assert requests[c['id']]['0']==p.request(c,'native',old['native'])
        assert requests[c['id']]['7']==p.request(c,'native',old['expanded'])
        assert requests[c['id']]['5']==requests[c['id']]['4'] # A subset C
    freeze(OUT/'requests.json',requests);freeze(OUT/'materials.json',materials)
    freeze(OUT/'protocol.json',{'n':16,'max_new_calls':96,'opened_development_attribution':True,
        'packet_names':['A:raw','B:two_planned_queries','C:raw_then_followup'],
        'empty_full_reference_repeat':0,'not_training':True,
        'sha256':{str(f.relative_to(ROOT)):p.base.h(f) for f in (Path(__file__),DESIGN,OUT/'requests.json',OUT/'materials.json')}})


def verify():
    r.verify();protocol=p.base.read(OUT/'protocol.json')
    for path,value in protocol['sha256'].items():
        if p.base.h(ROOT/path)!=value:raise ValueError('path attribution changed:'+path)


def collect(transport):
    verify();requests=p.base.read(OUT/'requests.json');outputs={}
    with lock(OUT):
        for cid,arms in requests.items():
            folder=OUT/cid;outputs[cid]={}
            with lock(folder):
                journal=CallJournal(folder,transport,6)
                if {a['key'] for a in read_rows(folder/'attempts.jsonl')}-{x['key'] for x in read_rows(folder/'responses.jsonl')}:raise RuntimeError('unresolved attempt')
                existing=[x for x in read_rows(r.OUT/cid/'responses.jsonl') if x['key'].startswith('0/')]
                for mask,req in arms.items():
                    row=next((x for x in existing+read_rows(folder/'responses.jsonl') if x['request']==req),None)
                    if row is None:
                        paths=list(p.OUT.rglob('responses.jsonl'))+[repeat.OUT/'responses.jsonl',pool.OUT/'responses.jsonl']+list(r.OUT.rglob('responses.jsonl'))+list(OUT.rglob('responses.jsonl'))
                        spent=sum((json.loads(x['response']['body'])['usage']['prompt_tokens']*.3+json.loads(x['response']['body'])['usage']['completion_tokens']*1.2)/1e6 for path in paths for x in read_rows(path))
                        reserve=((len(p.base.encoded(req))+4096)*.3+req['max_tokens']*1.2)/1e6
                        if spent+reserve>5:raise RuntimeError('cumulative cap exhausted')
                        row=journal.call(mask,req)
                    if digest(row['request'])!=row['request_hash'] or digest(row['response'])!=row['response_hash']:raise ValueError('source journal changed')
                    docs=json.loads(req['messages'][1]['content'])['documents']
                    outputs[cid][mask]=p.parse(row,'native',docs)
                freeze(folder/'sealed_outputs.json',outputs[cid])
            print('path table sealed',len(outputs),'/16',flush=True)
        freeze(OUT/'sealed_outputs.json',outputs)


def analyze():
    verify();outputs=p.base.read(OUT/'sealed_outputs.json');labels=p.base.read(r.OUT/'private_labels.json')
    if set(outputs)!=set(labels) or any(set(v)!=set(map(str,range(8))) for v in outputs.values()):raise ValueError('incomplete coalition seal')
    metrics={};credits=[];synergy=[]
    for mask in range(8):
        scores=[p.answer_score(outputs[c][str(mask)]['answer'],labels[c]) for c in labels]
        metrics[str(mask)]={'em_correct':sum(v['em'] for v in scores),'mean_f1':sum(v['f1'] for v in scores)/16,
            'invalid':sum(not outputs[c][str(mask)]['valid'] for c in labels)}
    for cid in labels:
        values={frozenset(i for i in range(3) if mask&(1<<i)):p.answer_score(outputs[cid][str(mask)]['answer'],labels[cid])['em'] for mask in range(8)}
        credits.append({'id':cid,'shapley':shapley(values),'full_minus_native':values[frozenset(range(3))]-values[frozenset()]})
        if values[frozenset(range(3))]>max(values[frozenset([i])] for i in range(3)):synergy.append(cid)
    raw=[x for path in OUT.rglob('responses.jsonl') for x in read_rows(path)]
    result={'scope':'opened development complete source-path table, not game-learning gain','n':16,
        'metrics_by_bitmask_ABC':metrics,'full_correct_all_single_paths_wrong':synergy,
        'best_fixed_single_correct_descriptive':max(metrics[str(1<<i)]['em_correct'] for i in range(3)),
        'mean_observed_shapley':[sum(c['shapley'][i] for c in credits)/16 for i in range(3)],'credits':credits,
        'physical_calls':len(raw),'physical_tokens':sum(json.loads(x['response']['body'])['usage']['total_tokens'] for x in raw),
        'sum_http_seconds':sum(x['wall_seconds'] for x in raw),
        'usd_proxy':sum((json.loads(x['response']['body'])['usage']['prompt_tokens']*.3+json.loads(x['response']['body'])['usage']['completion_tokens']*1.2)/1e6 for x in raw)}
    freeze(OUT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='credits'},indent=2),flush=True)


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
