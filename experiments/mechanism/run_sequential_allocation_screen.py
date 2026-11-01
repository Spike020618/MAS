"""Frozen 40-case stop/continue screen; no policy or game is trained."""
import argparse
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import time
from urllib.parse import urlparse
from experiments.core.mechanism.evidence_acquisition import Corpus, Ledger, normalized_answer
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism import replicate_evidence_acquisition as canonical_source
from experiments.mechanism.control_answer_order import reorder
from experiments.mechanism.control_evidence_thinking import thinking
from experiments.mechanism.build_upgrade_history import build, normalize
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows, append

ROOT=p.ROOT
OUT=ROOT/'experiments/results/sequential_allocation_v1'
DESIGN=ROOT/'docs/research/sequential_allocation_v1/锁定方案.md'
MODES=('disabled','enabled')


def final_request(case,docs,mode):
    if mode not in MODES:raise ValueError('unknown mode')
    req=reorder(p.request(case,'native',docs))
    return thinking(req) if mode=='enabled' else req


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():verify();return
    p.verify();history=build(ROOT)
    if history['errors']:raise ValueError('unreadable history')
    used={v['text'] for v in history['texts'].values()}
    source=p.data.parquet_module().read_table(p.data.RAW/'dev_fullwiki.parquet',columns=['id','question','context']).to_pylist()
    source.sort(key=lambda c:sha256(('stop-continue-v1/'+c['id']).encode()).hexdigest())
    chosen=[c for c in source if normalize(c['question']) not in used][:40]
    if len(chosen)!=40:raise ValueError('insufficient unused cases')
    cases=[];splits={}
    for i,c in enumerate(chosen):
        docs=[]
        for title,sentences in zip(c['context']['title'],c['context']['sentences']):
            text=''.join(sentences);docs.append({'docid':'native:'+sha256(title.encode()).hexdigest(),
                'title':title,'text':text,'text_sha256':sha256(text.encode()).hexdigest()})
        canonical_source.canonical(docs)
        cases.append({'id':c['id'],'question':c['question'],'native_documents':docs})
        splits[c['id']]='development' if i<20 else 'confirmation'
    freeze(OUT/'public_cases.json',cases);freeze(OUT/'splits.json',splits)
    ids={c['id'] for c in cases}
    labels={c['id']:c['answer'] for c in p.data.parquet_module().read_table(p.data.RAW/'dev_fullwiki.parquet',columns=['id','answer']).to_pylist() if c['id'] in ids}
    freeze(OUT/'private_labels.json',labels)
    freeze(OUT/'selection.json',{'prior_normalized_overlap':0,'not_semantic_or_source_group_isolation':True,
        'history_sha256':{v['path']:v['sha256'] for v in history['files']}})
    files=[Path(__file__),DESIGN,ROOT/'experiments/tests/test_sequential_allocation_screen.py',
        ROOT/'experiments/core/mechanism/evidence_acquisition.py',
        ROOT/'experiments/mechanism/run_upgrade_pilot.py',ROOT/'experiments/mechanism/run_open_evidence_screen_v2.py',
        ROOT/'experiments/mechanism/replicate_evidence_acquisition.py',ROOT/'experiments/mechanism/control_answer_order.py',
        ROOT/'experiments/mechanism/control_evidence_thinking.py',ROOT/'experiments/mechanism/run_full_context_challenge.py',
        ROOT/'experiments/mechanism/redesign_hotpot_data.py',ROOT/'experiments/mechanism/build_upgrade_history.py',
        OUT/'public_cases.json',OUT/'private_labels.json',OUT/'splits.json',OUT/'selection.json']
    freeze(OUT/'protocol.json',{'version':'stop-continue-screen-v1','n':40,'maximum_calls':320,
        'usd_proxy_cap':5,'model':'deepseek-flash','fp':p.base.FP,'no_training':True,
        'index_manifest_sha256':p.base.h(p.data.ENV/'index_manifest.json'),
        'sha256':{str(f.relative_to(ROOT)):p.base.h(f) for f in files}})


def verify():
    p.verify();protocol=p.base.read(OUT/'protocol.json')
    for path,value in protocol['sha256'].items():
        if p.base.h(ROOT/path)!=value:raise ValueError('frozen screen changed:'+path)
    if p.base.h(p.data.ENV/'index_manifest.json')!=protocol['index_manifest_sha256']:raise ValueError('corpus changed')
    return protocol


def usd(row):
    u=json.loads(row['response']['body'])['usage']
    return (u['prompt_tokens']*.3+u['completion_tokens']*1.2)/1e6


def collect(case,transport,budget):
    folder=OUT/case['id'];started=time.perf_counter()
    with lock(folder):
        journal=CallJournal(folder,transport,8)
        prior=read_rows(folder/'attempts.jsonl')
        if {x['key'] for x in prior}-{x['key'] for x in read_rows(folder/'responses.jsonl')}:raise RuntimeError('unresolved attempt; no retry')
        def call(key,req,docs,kind):
            rows=read_rows(folder/'responses.jsonl')
            row=next((x for x in rows if x['request']==req),None)
            if row is None:
                reserve=((len(p.base.encoded(req))+4096)*.3+req['max_tokens']*1.2)/1e6
                if budget['used']+reserve>budget['cap']:raise RuntimeError('budget reserve exhausted')
                row=journal.call(key,req);budget['used']+=usd(row)
            else:row=journal.call(row['key'],req)
            return {'source_key':row['key'],'parsed':p.parse(row,kind,docs)}
        docs,origins=canonical_source.canonical(case['native_documents'])
        states=[];corpus=Corpus(p.data.ENV/'wiki_fts.sqlite');complete=False
        try:
            for stage in range(3):
                stage_start=time.perf_counter();packet=None;query=None
                if stage:
                    plan=call('query'+str(stage),p.request(case,'seq_plan',docs),docs,'seq_plan')
                    if not plan['parsed']['valid']:raise RuntimeError('invalid query; no repair')
                    query=plan['parsed']['query'];envpath=folder/('search'+str(stage)+'.json')
                    if envpath.exists():
                        packet=p.base.read(envpath)
                        if packet['query']!=query:raise ValueError('cached query changed')
                    else:
                        ledger=Ledger(max_searches=1,max_unique_bytes=40000);ledger.acquire(corpus,query)
                        if ledger.searches[0]['skipped_budget_ids']:raise RuntimeError('source packet over budget; no selective omission')
                        packet={'query':query,'documents':ledger.public_documents(),'event':ledger.searches[0]}
                        freeze(envpath,packet)
                    previous={d['docid'] for d in docs}
                    docs,origins=canonical_source.canonical(docs+packet['documents'])
                    novel=len({d['docid'] for d in docs}-previous)
                else:plan=None;novel=0
                answers={}
                for mode in sorted(MODES,key=lambda m:sha256((case['id']+str(stage)+m).encode()).hexdigest()):
                    answers[mode]=call('s'+str(stage)+'/'+mode,final_request(case,docs,mode),docs,'native')
                timer=folder/('s'+str(stage)+'_timing.json')
                if not timer.exists():freeze(timer,{'seconds':time.perf_counter()-stage_start,
                    'resumed_case':bool(prior),'scope':'observed stage execution, may be partial if resumed'})
                state={'stage':stage,'documents':docs,'origins':origins,'new_documents':novel,
                    'query_plan':plan,'retrieval':packet,'answers':answers,'timing':p.base.read(timer)}
                freeze(folder/('s'+str(stage)+'.json'),state);states.append(state)
            freeze(folder/'sealed_outputs.json',states);complete=True
            return states
        finally:
            corpus.close();append(folder/'invocations.jsonl',{'elapsed_seconds':time.perf_counter()-started,'completed':complete})


def run(transport):
    protocol=verify()
    if time.strftime('%Y-%m-%d',time.gmtime())!=p.base.PRICE['checked_date']:raise RuntimeError('recheck provider rates before new calls')
    outputs={};started=time.perf_counter();was_complete=(OUT/'sealed_outputs.json').exists()
    with lock(OUT):
        for path in OUT.rglob('attempts.jsonl'):
            if {v['key'] for v in read_rows(path)}-{v['key'] for v in read_rows(path.with_name('responses.jsonl'))}:raise RuntimeError('unresolved collection attempt')
        budget={'used':sum(usd(row) for path in OUT.rglob('responses.jsonl') for row in read_rows(path)),'cap':protocol['usd_proxy_cap']}
        for case in p.base.read(OUT/'public_cases.json'):
            outputs[case['id']]=collect(case,transport,budget)
            print('sequential allocation sealed',len(outputs),'/40',flush=True)
        freeze(OUT/'sealed_outputs.json',outputs)
        if not was_complete:append(OUT/'invocations.jsonl',{'elapsed_seconds':time.perf_counter()-started,'completed':len(outputs),'known_usd_proxy':budget['used']})


def path_account(states,rows,stage,mode,adaptive):
    keys=[]
    for s in range(1,stage+1):keys.append(states[s]['query_plan']['source_key'])
    for s in (range(stage+1) if adaptive else [stage]):keys.append(states[s]['answers'][mode]['source_key'])
    unique=list(dict.fromkeys(keys));selected=[rows[k] for k in unique]
    return {'tokens':sum(json.loads(v['response']['body'])['usage']['total_tokens'] for v in selected),
        'calls':len(unique),'http_seconds':sum(v['wall_seconds'] for v in selected),
        'search_seconds':sum(states[s]['retrieval']['event']['wall_seconds'] for s in range(1,stage+1))}


def frontier(options):
    """DP: minimum tokens for every exact (correct, calls) value; no labels online."""
    states={(0,0):(0,[])}
    for choices in options:
        nxt={}
        for (correct,calls),(tokens,path) in states.items():
            for stage,quality,cost in choices:
                key=(correct+int(quality),calls+cost['calls']);value=(tokens+cost['tokens'],path+[stage])
                if key not in nxt or value[0]<nxt[key][0]:nxt[key]=value
        states=nxt
    return [{'correct':q,'calls':c,'tokens':v[0],'stages':v[1]} for (q,c),v in sorted(states.items())]


def mcnemar(b,c):
    n=b+c
    return min(1.,2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.


def analyze():
    verify();outputs=p.base.read(OUT/'sealed_outputs.json');cases=p.base.read(OUT/'public_cases.json')
    if set(outputs)!={c['id'] for c in cases} or len(outputs)!=40 or any(len(v)!=3 for v in outputs.values()):raise ValueError('complete 40-case seal required before labels')
    labels=p.base.read(OUT/'private_labels.json');splits=p.base.read(OUT/'splits.json');cells={};per_case={}
    for cid,states in outputs.items():
        rows={v['key']:v for v in read_rows(OUT/cid/'responses.jsonl')};per_case[cid]={}
        for mode in MODES:
            per_case[cid][mode]=[{'score':p.answer_score(state['answers'][mode]['parsed']['answer'],labels[cid]),
                'answer':state['answers'][mode]['parsed']['answer'],'valid':state['answers'][mode]['parsed']['valid'],
                'fixed_cost':path_account(states,rows,s,mode,False),'adaptive_cost':path_account(states,rows,s,mode,True)} for s,state in enumerate(states)]
    for mode in MODES:
        dev=[c for c in outputs if splits[c]=='development']
        chosen=min(range(3),key=lambda s:(-sum(per_case[c][mode][s]['score']['em'] for c in dev),sum(per_case[c][mode][s]['fixed_cost']['tokens'] for c in dev),s))
        cells[mode]={'development_selected_fixed_stage':chosen,'splits':{}}
        for split in ('development','confirmation','all'):
            ids=[c for c in outputs if split=='all' or splits[c]==split];n=len(ids);fixed=[]
            for s in range(3):
                slots=[per_case[c][mode][s] for c in ids]
                fixed.append({'em_correct':sum(x['score']['em'] for x in slots),'mean_f1':sum(x['score']['f1'] for x in slots)/n,
                    'tokens':sum(x['fixed_cost']['tokens'] for x in slots),'calls':sum(x['fixed_cost']['calls'] for x in slots),
                    'invalid':sum(not x['valid'] for x in slots)})
            policies={}
            for policy in ('abstain_only','no_new_documents'):
                stops=[]
                for cid in ids:
                    s=0
                    if policy=='abstain_only':
                        while s<2 and normalized_answer(per_case[cid][mode][s]['answer']) in ('abstain','noanswer'):s+=1
                    else:
                        s=1
                        if outputs[cid][1]['new_documents']>0:s=2
                    stops.append(s)
                policies[policy]={'stages':stops,'em_correct':sum(per_case[c][mode][s]['score']['em'] for c,s in zip(ids,stops)),
                    'tokens':sum(per_case[c][mode][s]['adaptive_cost']['tokens'] for c,s in zip(ids,stops)),
                    'calls':sum(per_case[c][mode][s]['adaptive_cost']['calls'] for c,s in zip(ids,stops))}
            oracle={}
            for kind in ('fixed_cost','adaptive_cost'):
                options=[[(s,per_case[c][mode][s]['score']['em'],per_case[c][mode][s][kind]) for s in range(3)] for c in ids]
                values=frontier(options);feasible=[v for v in values if v['tokens']<=fixed[2]['tokens'] and v['calls']<=fixed[2]['calls']]
                best=max(feasible,key=lambda v:(v['correct'],-v['tokens'],-v['calls'])) if feasible else None
                oracle[kind]={'not_deployable':True,'best_with_fixed_s2_token_and_call_budget':best,'frontier':values}
            pairs=[]
            for s,t in ((0,2),(0,1),(1,2)):
                correct=[c for c in ids if per_case[c][mode][t]['score']['em']>per_case[c][mode][s]['score']['em']]
                harmed=[c for c in ids if per_case[c][mode][t]['score']['em']<per_case[c][mode][s]['score']['em']]
                pairs.append({'from':s,'to':t,'corrected':correct,'harmed':harmed,'exact_p':mcnemar(len(correct),len(harmed))})
            ordered=sorted(range(3),key=lambda i:pairs[i]['exact_p']);previous=0.
            for rank,i in enumerate(ordered):previous=max(previous,min(1.,(3-rank)*pairs[i]['exact_p']));pairs[i]['holm_p']=previous
            best_fixed=max(x['em_correct'] for x in fixed);best_oracle=oracle['adaptive_cost']['best_with_fixed_s2_token_and_call_budget']
            cells[mode]['splits'][split]={'n':n,'fixed_stages':fixed,'selected_fixed_confirmation_score':fixed[chosen]['em_correct'],
                'simple_policies':policies,'oracle':oracle,'paired':pairs,
                'oracle_gap_to_hindsight_best_fixed':None if best_oracle is None else best_oracle['correct']-best_fixed}
    conf=cells['enabled']['splits']['confirmation'];gap=conf['oracle_gap_to_hindsight_best_fixed']
    raw=[v for path in OUT.rglob('responses.jsonl') for v in read_rows(path)]
    result={'scope':'40-case development screen, not policy learning, oracle is not deployable','n':40,'cells':cells,
        'per_case':per_case,'gate':{'quantitative_pass':gap is not None and gap>=2,
            'factual_difference_review_required':True,'automatic_training_allowed':False,
            'status':'requires_factual_review' if gap is not None and gap>=2 else 'do_not_enter_mechanism_training'},
        'physical_calls':len(raw),'physical_tokens':sum(json.loads(v['response']['body'])['usage']['total_tokens'] for v in raw),
        'usd_proxy':sum(usd(v) for v in raw),'sum_http_seconds':sum(v['wall_seconds'] for v in raw),
        'invocations':read_rows(OUT/'invocations.jsonl')}
    freeze(OUT/'result.json',result)
    print(json.dumps({'gate':result['gate'],'physical_calls':len(raw),'physical_tokens':result['physical_tokens'],'usd_proxy':result['usd_proxy'],
        'confirmation':{m:{'fixed':cells[m]['splits']['confirmation']['fixed_stages'],
            'oracle_gap':cells[m]['splits']['confirmation']['oracle_gap_to_hindsight_best_fixed']} for m in MODES}},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','verify','analyze','replay'));stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='verify':verify()
    elif stage=='analyze':analyze()
    elif stage=='replay':
        def blocked(req):raise AssertionError('replay attempted model call')
        run(blocked);analyze()
    else:
        from experiments.core.llm import LLM_BACKENDS
        if urlparse(LLM_BACKENDS['deepseek']['base_url']).hostname!='api.deepseek.com':raise ValueError('unexpected endpoint')
        from experiments.mechanism.study_explicit_thinking_control import transport
        run(transport);analyze()
