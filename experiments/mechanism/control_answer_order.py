"""Matched native/expanded answer-last format control on opened 16-case batch."""
import argparse
import copy
import json
from pathlib import Path
from experiments.mechanism import attribute_acquisition_paths as a
from experiments.mechanism import replicate_evidence_acquisition as r
from experiments.mechanism import repeat_open_evidence_solver as repeat
from experiments.mechanism import probe_acquired_evidence_pool as pool
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows,digest

p=r.p;ROOT=r.ROOT;OUT=ROOT/'experiments/results/redesign_v1/answer_order_control_v1'
DESIGN=ROOT/'docs/research/redesign_v1/答案顺序_强控制协议_2026-10-25.md'
OLD=('Return JSON: answer (short answer string, no sentence framing), reasoning (at most150 words), '
    'docids (list of supplied document IDs used; empty when no supplied evidence).')
NEW=('Return JSON: reasoning (at most150 words), docids (list of supplied document IDs used; empty when no supplied evidence), '
    'answer (short answer string, no sentence framing). Write the answer key last.')


def reorder(req):
    changed=copy.deepcopy(req);text=changed['messages'][0]['content']
    if text.count(OLD)!=1:raise ValueError('unexpected primary instruction')
    changed['messages'][0]['content']=text.replace(OLD,NEW)
    return changed


def prepare():
    r.verify();OUT.mkdir(parents=True,exist_ok=True);requests={}
    for c in p.base.read(r.OUT/'public_cases.json'):
        mat=p.base.read(r.OUT/c['id']/'material.json')
        requests[c['id']]={arm:reorder(p.request(c,'native',mat[arm])) for arm in ('native','expanded')}
    freeze(OUT/'requests.json',requests)
    freeze(OUT/'protocol.json',{'n':16,'max_calls':32,'post_hoc_format_control':True,
        'sha256':{str(f.relative_to(ROOT)):p.base.h(f) for f in (Path(__file__),DESIGN,OUT/'requests.json')}})


def verify():
    r.verify()
    for path,expected in p.base.read(OUT/'protocol.json')['sha256'].items():
        if p.base.h(ROOT/path)!=expected:raise ValueError('format control changed:'+path)


def collect(transport):
    verify();requests=p.base.read(OUT/'requests.json');outputs={}
    with lock(OUT):
        journal=CallJournal(OUT,transport,32)
        if {x['key'] for x in read_rows(OUT/'attempts.jsonl')}-{x['key'] for x in read_rows(OUT/'responses.jsonl')}:raise RuntimeError('unresolved attempt')
        for cid,arms in requests.items():
            outputs[cid]={}
            for arm,req in sorted(arms.items(),key=lambda v:digest([cid,v[0],'answer-order'])):
                match=next((x for x in read_rows(OUT/'responses.jsonl') if x['request']==req),None)
                key=match['key'] if match else cid+'/'+arm
                if match is None:
                    paths=list(p.OUT.rglob('responses.jsonl'))+[repeat.OUT/'responses.jsonl',pool.OUT/'responses.jsonl']+list(r.OUT.rglob('responses.jsonl'))+list(a.OUT.rglob('responses.jsonl'))+[OUT/'responses.jsonl']
                    spent=sum((json.loads(x['response']['body'])['usage']['prompt_tokens']*.3+json.loads(x['response']['body'])['usage']['completion_tokens']*1.2)/1e6 for path in paths for x in read_rows(path))
                    reserve=((len(p.base.encoded(req))+4096)*.3+req['max_tokens']*1.2)/1e6
                    if spent+reserve>5:raise RuntimeError('cumulative cap exhausted')
                row=journal.call(key,req);docs=json.loads(req['messages'][1]['content'])['documents']
                outputs[cid][arm]=p.parse(row,'native',docs)
            print('answer order sealed',len(outputs),'/16',flush=True)
        freeze(OUT/'sealed_outputs.json',outputs)


def analyze():
    verify();outputs=p.base.read(OUT/'sealed_outputs.json');labels=p.base.read(r.OUT/'private_labels.json')
    if set(outputs)!=set(labels):raise ValueError('incomplete format control')
    metrics={}
    for arm in ('native','expanded'):
        scores=[p.answer_score(outputs[c][arm]['answer'],labels[c]) for c in labels]
        metrics[arm]={'em_correct':sum(v['em'] for v in scores),'mean_f1':sum(v['f1'] for v in scores)/16,
            'invalid':sum(not outputs[c][arm]['valid'] for c in labels)}
    corrected=[c for c in labels if p.answer_score(outputs[c]['expanded']['answer'],labels[c])['em'] and not p.answer_score(outputs[c]['native']['answer'],labels[c])['em']]
    harmed=[c for c in labels if not p.answer_score(outputs[c]['expanded']['answer'],labels[c])['em'] and p.answer_score(outputs[c]['native']['answer'],labels[c])['em']]
    raw=read_rows(OUT/'responses.jsonl')
    result={'scope':'post-hoc matched format control, not independent test','n':16,'metrics':metrics,'corrected':corrected,'harmed':harmed,
        'physical_calls':len(raw),'physical_tokens':sum(json.loads(x['response']['body'])['usage']['total_tokens'] for x in raw),
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
