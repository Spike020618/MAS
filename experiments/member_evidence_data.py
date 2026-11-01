"""Scarce native evidence, genuinely different initial reports, shared permissions."""
import collections,copy,itertools,json
from pathlib import Path
import numpy as np
from experiments import calibrated_action_value as v
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.bound_opportunity_probe import validate_linked
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.runtime import Runtime

DEFAULT=ROOT/'experiments/results/member_evidence_v2'

def prepare(root,fit_count=12,cal_count=4,test_count=12):
    from sklearn.feature_extraction.text import TfidfVectorizer
    source=ROOT/'experiments/data/public/climate_fever/climate-fever.jsonl'
    raw=[json.loads(x)for x in source.read_text().splitlines()if x.strip()]
    excluded=set()
    for p in [ROOT/'experiments/results/consensus_mechanism_paper_v2/semantic_data/tasks.json',
              ROOT/'experiments/results/calibrated_action_value_v3/candidate_pool/tasks.json']:
        excluded.update(t['metadata']['native_id'].split('/')[-1]for t in read(p))
    corpus={fingerprint(e['evidence']):{'text':e['evidence'],'title':e['article'],'native_evidence_id':e['evidence_id']}
            for r in raw for e in r['evidences']if e['evidence'].strip()}
    keys=sorted(corpus);documents=[corpus[k]for k in keys]
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=35000)
    matrix=vectorizer.fit_transform(d['title']+' '+d['text']for d in documents)
    eligible=[r for r in raw if str(r['claim_id'])not in excluded and r['claim_label']in v.LABELS
              and len({fingerprint(e['evidence'])for e in r['evidences']})>=5]
    eligible.sort(key=lambda r:fingerprint(['member-evidence-v1',str(r['claim_id'])]))
    if len(eligible)<fit_count+cal_count+test_count:raise ValueError('insufficient eligible native claims')
    tasks=[];labels={}
    for i,row in enumerate(eligible[:fit_count+cal_count+test_count]):
        tid='member:'+str(row['claim_id']);partition='fit'if i<fit_count else'calibration'if i<fit_count+cal_count else'test'
        native={fingerprint(e['evidence']):e for e in row['evidences']}
        scores=(matrix@vectorizer.transform([row['claim']]).T).toarray().ravel()
        order=sorted(range(len(keys)),key=lambda j:(-float(scores[j]),keys[j]))
        seeds=[j for j in order if keys[j]not in native][:3]
        if len(seeds)!=3:raise ValueError('three distinct nonnative initial passages required')
        catalog={};seed_ids=[];candidate_ids=[]
        for j,k in enumerate(seeds):
            sid='seed:'+str(j);seed_ids.append(sid);d=documents[k]
            catalog[sid]={'source_id':sid,'title':d['title'],'text':d['text'],'kind':'retrieved_background',
              'scope':{'task_id':tid,'subject':'reference'},'allowed_uses':['prediction_basis'],
              'retrieval_score':float(scores[k]),'native_evidence_id':d['native_evidence_id']}
        query=vectorizer.transform([row['claim']])
        for j,key in enumerate(sorted(native,key=lambda x:fingerprint([tid,x]))):
            e=native[key];sid='candidate:'+str(j);candidate_ids.append(sid)
            score=float((vectorizer.transform([e['article']+' '+e['evidence']])@query.T).toarray()[0,0])
            catalog[sid]={'source_id':sid,'title':e['article'],'text':e['evidence'],'kind':'native_supplied_evidence',
              'scope':{'task_id':tid,'subject':'reference'},'allowed_uses':['prediction_basis'],
              'retrieval_score':score,'native_evidence_id':e['evidence_id']}
        task={'id':tid,'kind':'qa','prompt':'Judge the claim using only supplied evidence: '+row['claim'],
              'options':['supported','refuted','not_enough_info'],'candidates':[],'initial':'not_enough_info','source_catalog':catalog,
              'metadata':{'native_id':'ClimateFEVER/'+str(row['claim_id']),'given_claim':row['claim'],
                'target_claim_id':'target','target_role':'decision_query','evidence_only':True,
                'uncertainty_answers':['not_enough_info'],'acquisition_partition':partition,
                'seed_ids':seed_ids,'acquisition_ids':candidate_ids,'protocol_v3':{'initial_source_ids':seed_ids}}}
        assert len(candidate_ids)>2
        tasks.append(task);labels[tid]={'kind':'qa','answer':v.LABELS[row['claim_label']]}
    create(root/'tasks.json',tasks)
    create(root/'train.labels.private.json',{t['id']:labels[t['id']]for t in tasks if t['metadata']['acquisition_partition']!='test'})
    create(root/'test.labels.private.json',{t['id']:labels[t['id']]for t in tasks if t['metadata']['acquisition_partition']=='test'})
    create(root/'data_manifest.json',{'source_sha256':sha(source),'selection':'claim hash, no label/difficulty selection; prior semantic180 and calibration56 excluded',
      'partitions':dict(collections.Counter(t['metadata']['acquisition_partition']for t in tasks)),
      'initial':'three distinct TFIDF retrieved background passages, one per initial Agent; excludes exact native passage text',
      'additional':'all original supplied evidence, at least five distinct passages; at most two new acquisitions',
      'public_after_initial':'all initial reports shared; members retain their own seed raw passage; new acquired passages public; single-context control sees all three seed raw passages',
      'scope':'original claim labels and supplied-evidence pool with retrieval-background initialization; not original leaderboard; shared background sources permitted'})
    return tasks

def compact_request(task,state,phase,reports,reviewer,ids):
    req=c.request(task,state,phase,100,reports,{'target_id':'target','source_ids':list(ids)},reviewer=reviewer)
    req['max_tokens']=700
    req['messages'][0]['content']+=' Return only decision, claim, span_ids. claim is a concise evidence reason of at most 50 words. Classify as supported, refuted, or not_enough_info. Missing evidence warrants not_enough_info. Do not discuss the output contract.'
    return req

def initialize(task,api,index):
    reports=[];events=[]
    for i,sid in enumerate(task['metadata']['seed_ids']):
        local=c.fresh(task);c.expose(task,local,[sid]);req=compact_request(task,local,'initial',[],None,[sid])
        event={'valid':False,'request_hash':fingerprint(req),'visible':[sid],'reviewer':i}
        try:
            record=api.call(f'{index}_initial_{i}',req);event.update(raw_report=record['report'],usage=record['usage'],returned_model=record['model'])
            c.validate_report(task,local,record['report'],'initial',100)
            event['valid']=True;report={**record['report'],'contract_valid':True}
        except (ValueError,KeyError,TypeError,RuntimeError,OSError)as exc:
            event['error']=str(exc);report={'decision':'not_enough_info','claim':'No valid initial report.','span_ids':[],'contract_valid':False}
        reports.append(report);events.append(event)
    state=c.fresh(task);c.expose(task,state,task['metadata']['seed_ids'])
    state['decision']=c.aggregate(task,[r['decision']for r,e in zip(reports,events)if e['valid']])
    return {'task_id':task['id'],'state':state,'reports':reports,'events':events}

def review(task,state,reports,reviewer,ids,api,slot,independent=False):
    exposed=copy.deepcopy(state);c.expose(task,exposed,ids)
    global_visible=list(exposed['visible'])
    seeds=task['metadata']['seed_ids']
    if reviewer<3:
        exposed['visible']=[sid for sid in exposed['visible']if sid not in seeds or sid==seeds[reviewer]]
    peers=[]if independent else reports
    req=compact_request(task,exposed,'direct',peers,reviewer if not independent else None,ids)
    event={'valid':False,'accepted':False,'reviewer':reviewer,'source_ids':list(ids),
      'before':state['decision'],'state_hash':fingerprint(state),'request_hash':fingerprint(req)}
    try:
        record=api.call(slot,req);direct=record['report'];event.update(raw_report=direct,usage=record['usage'],returned_model=record['model'])
        c.validate_report(task,exposed,direct,'direct',100)
        native=direct['decision'];relation={'supported':'support','refuted':'refute','not_enough_info':'unclear'}[native]
        patch={'operation':'set_answer','value':native}if native!=state['decision']else None
        report={'target_id':'target','verdict':relation,'span_ids':direct.get('span_ids',[]),'rationale':direct['claim'],
           'patch':patch,'decision_effect':{'choice':'revise'if patch else'keep','reason':direct['claim'],'candidate_pair':None}}
        refs,basis=validate_linked(task,exposed,report,{'target_id':'target','source_ids':list(ids)})
        updated,accepted=c.commit(task,exposed,report,'review',100,refs)
        # This target is an immutable decision query, not a refuted assertion to
        # discard. Preserve its judged status but allow subsequent evidence.
        updated['claims']['target']['active']=True
        event.update(valid=True,accepted=accepted,report=report,revision_basis=basis)
    except (ValueError,KeyError,TypeError,RuntimeError,OSError)as exc:
        event['error']=str(exc);updated=exposed
    updated['visible']=global_visible
    event['after']=updated['decision']
    event['actor_visible_sources']=exposed['visible']
    return updated,event

def features(task,origin,reports,reviewer,ids):
    seeds=task['metadata']['seed_ids']
    actor_base=[sid for sid in origin['visible']if reviewer>=3 or sid not in seeds or sid==seeds[reviewer]]
    visible=list(dict.fromkeys(actor_base+list(ids)))
    f=v.features(task,origin,ids)
    f['visible_count']=float(len(actor_base))
    all_words=set().union(*(v.tokens(task['source_catalog'][sid]['text'])for sid in visible))
    q=v.tokens(task['metadata']['given_claim'])
    f['public_plus_action_coverage']=len(q&all_words)/max(1,len(q))
    f['initial_disagreement']=len({r['decision']for r in reports})/max(1,len(reports))
    own=reports[reviewer]if reviewer<3 else None
    f['single_controller']=float(own is None)
    if own:
        own_words=v.tokens(own['claim']);new_words=set().union(*(v.tokens(task['source_catalog'][sid]['text'])for sid in ids))if ids else set()
        f['own_contract_valid']=float(own.get('contract_valid',True))
        f['own_agrees_public']=float(own['decision']==origin['decision'])
        f['own_answer:'+own['decision']]=1.
        f['own_claim_overlap']=len(own_words&new_words)/max(1,len(own_words))
        pool=c.spans(task)
        own_sources={pool[sid]['source_id']for sid in own.get('span_ids',[])if sid in pool}
        seed=task['source_catalog'][seeds[reviewer]]
        f['own_seed_retrieval']=seed['retrieval_score']
        f['own_seed_claim_coverage']=len(v.tokens(seed['text'])&q)/max(1,len(q))
        f['own_source_count']=float(len(own_sources))
    return f

def freeze(root,tasks,stage,max_cny,cap,extra_files=(),model=None):
    files=[Path(__file__),Path(v.__file__),ROOT/'experiments/bound_opportunity_probe.py']+list((ROOT/'experiments/consensus_mechanism').glob('*.py'))+list((ROOT/'experiments/paired_deliberation').glob('*.py'))+[ROOT/'experiments/consensus_recommendation/common.py',ROOT/'experiments/evidence_consensus/schema.py']+list(extra_files)
    plan={'version':'member-evidence-1','stage':stage,'tasks_hash':fingerprint(tasks),'physical_cap':cap,
       'max_cny':max_cny,'reuse_root':None,'model_sha256':sha(model)if model else None,
       'source_sha256':{str(f.relative_to(ROOT)):sha(f)for f in sorted(set(files))}}
    create(root/'plan.json',plan)
    for name,h in plan['source_sha256'].items():
        dest=root/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists()and sha(dest)!=h:raise ValueError('frozen source mismatch')
        dest.write_bytes((ROOT/name).read_bytes())
    return plan
