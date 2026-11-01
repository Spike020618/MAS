"""Natural claim evidence acquisition with valid-only probabilistic value learning.

Human-labelled Climate-FEVER claim task, controlled provisional decision and two
hash-partitioned evidence packets. Not the original leaderboard protocol.
"""
import argparse,collections,copy,itertools,json,math,random,re
from pathlib import Path
import numpy as np
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.data import read
from experiments.consensus_mechanism.runtime import Runtime
from experiments.consensus_mechanism.runner import locked
from experiments.consensus_mechanism.engine import estimate as old_estimate
from experiments.consensus_mechanism.allocation import SetAllocator
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.bound_opportunity_probe import review_request,validate_linked
from experiments.bound_verdict_adapter import canonicalize
from experiments.paired_deliberation.native_optimization_runtime import audit

VERSION='probabilistic-action-value-2-compact-relation'
METHODS=('old_game','new_game','new_greedy','new_central','prior_game','fixed','random','reflection')
BRANCHES={'empty':[],'A':['A'],'B':['B'],'AB':['A','B']}
LABELS={'SUPPORTS':'supported','REFUTES':'refuted','NOT_ENOUGH_INFO':'not_enough_info'}

def tokens(s):return set(re.findall(r'[a-z0-9]{2,}',s.lower()))-c.STOP

def features(task,state,ids):
    q=tokens(task['metadata']['given_claim']); visible=set(state['visible'])
    texts=[task['source_catalog'][s]['text']for s in ids]
    sets=[tokens(t)for t in texts]; union=set().union(*sets)if sets else set()
    overlaps=[len(q&s)/max(1,len(q))for s in sets]
    f={'source_count':float(len(ids)),'chars':sum(map(len,texts))/6000,
       'claim_coverage':len(q&union)/max(1,len(q)),
       'max_coverage':max(overlaps,default=0.),'min_coverage':min(overlaps,default=0.),
       'mean_coverage':sum(overlaps)/max(1,len(overlaps)),
       'new_sources':float(len(set(ids)-visible)),'visible_count':float(len(visible)),
       'claim_chars':len(task['metadata']['given_claim'])/300,
       'version':float(state['version']),
       'initial:'+str(state['decision']):1.,'empty':float(not ids),
       'negation_words':float(len(union&{'not','never','cannot','false','without'})),
       'numbers':float(sum(t.isdigit()for t in union))/10}
    if len(sets)==2:
        f['source_redundancy']=len(sets[0]&sets[1])/max(1,len(sets[0]|sets[1]))
        f['coverage_complement']=f['claim_coverage']-f['max_coverage']
    else:f.update(source_redundancy=0.,coverage_complement=0.)
    return f

def prepare(root,fit_count=16,cal_count=4,test_count=12):
    path=ROOT/'experiments/data/public/climate_fever/climate-fever.jsonl'
    raw=[json.loads(l)for l in path.read_text().splitlines()if l.strip()]
    excluded={str(t['metadata']['native_id']).split('/')[-1]for t in read(ROOT/'experiments/results/consensus_mechanism_paper_v2/semantic_data/tasks.json')}
    available=[r for r in raw if str(r['claim_id'])not in excluded and r['claim_label']in LABELS and len(r['evidences'])>=2]
    available.sort(key=lambda r:fingerprint(['action-value-claim-v1',str(r['claim_id'])]))
    count=fit_count+cal_count+test_count
    if count>len(available):raise ValueError('insufficient native claims')
    tasks=[];vault={}
    for i,row in enumerate(available[:count]):
        partition='fit'if i<fit_count else'calibration'if i<fit_count+cal_count else'test'
        tid='claim-value:'+str(row['claim_id']); es=sorted(row['evidences'],key=lambda e:fingerprint([tid,e['evidence_id']]))
        parts=[es[::2],es[1::2]];catalog={}
        for sid,part in zip(['A','B'],parts):
            text='\n'.join('['+str(j+1)+'] '+e['article']+': '+e['evidence']for j,e in enumerate(part))
            catalog[sid]={'source_id':sid,'title':' / '.join(e['article']for e in part),'text':text,
                          'kind':'public_claim_evidence_packet','scope':{'task_id':tid,'subject':'reference'},
                          'allowed_uses':['prediction_basis'],'native_evidence_ids':[e['evidence_id']for e in part]}
        initial=['supported','refuted','not_enough_info'][int(fingerprint(['provisional',tid])[:8],16)%3]
        task={'id':tid,'kind':'qa','prompt':'Judge this claim using only the supplied evidence. Answer supported, refuted, or not_enough_info. Claim: '+row['claim'],
              'options':['supported','refuted','not_enough_info'],'candidates':[],'initial':initial,'source_catalog':catalog,
              'metadata':{'native_id':'ClimateFEVER/'+str(row['claim_id']),'family':'native_claim_packet',
                          'acquisition_partition':partition,'evidence_only':True,'given_claim':row['claim'],
                          'target_claim_id':'target','uncertainty_answers':['not_enough_info'],
                          'protocol_v3':{'initial_source_ids':[]}}}
        tasks.append(task);vault[tid]={'kind':'qa','answer':LABELS[row['claim_label']]}
    create(root/'tasks.json',tasks);create(root/'evaluation.private.json',vault)
    create(root/'train.labels.private.json',{t['id']:vault[t['id']]for t in tasks if t['metadata']['acquisition_partition']!='test'})
    create(root/'test.labels.private.json',{t['id']:vault[t['id']]for t in tasks if t['metadata']['acquisition_partition']=='test'})
    create(root/'data_manifest.json',{'source_sha256':sha(path),'selection':'claim hash, no label-based sampling; exclude old semantic180',
           'counts':dict(collections.Counter(t['metadata']['acquisition_partition']for t in tasks)),
           'group_key':'native claim ID; shared Wikipedia backgrounds allowed, no unseen-topic claim',
           'scope':'original claim labels with all supplied evidence partitioned into two packets; hash provisional answer. Not original leaderboard or natural initial-model error rate.'})
    return tasks

def actor(api,slot,task,state,ids):
    exposed=copy.deepcopy(state);c.expose(task,exposed,ids)
    action={'target_id':'target','source_ids':list(ids)}
    # Relation tasks have a one-to-one native answer / target relation mapping.
    # Ask once; derive the redundant patch/decision-effect fields mechanically.
    req=c.request(task,exposed,'direct',100,[],action,reviewer=0)
    req['max_tokens']=700
    req['messages'][0]['content'] += (
        ' Classify the supplied claim using only the visible passages. '
        'Return ONLY decision, claim, span_ids. claim is a concise evidence reason '
        'of at most 50 words, not an output-contract explanation. '
        'Use supported, refuted, or not_enough_info as decision. '
        'If neither support nor refutation is established choose not_enough_info.')
    event={'valid':False,'accepted':False,'before':state['decision'],'request_hash':fingerprint(req),
           'state_hash':fingerprint(state),'source_ids':list(ids)}
    try:
        record=api.call(slot,req);direct=record['report']
        event.update(raw_report=direct,usage=record['usage'],returned_model=record['model'])
        c.validate_report(task,exposed,direct,'direct',100,action)
        native=direct['decision']
        relation={'supported':'support','refuted':'refute','not_enough_info':'unclear'}[native]
        patch={'operation':'set_answer','value':native}if native!=state['decision']else None
        report={'target_id':'target','verdict':relation,'span_ids':direct.get('span_ids',[]),
                'rationale':direct['claim'],'patch':patch,
                'decision_effect':{'choice':'revise'if patch else'keep',
                    'reason':direct['claim'],'candidate_pair':None}}
        event['adapter_operations']=['native relation mapped to target verdict and local answer patch']
        refs,basis=validate_linked(task,exposed,report,action)
        after,accepted=c.commit(task,exposed,report,'review',100,refs)
        event.update(valid=True,accepted=accepted,report=report,revision_basis=basis)
    except (ValueError,KeyError,TypeError,RuntimeError,OSError)as exc:
        event['error']=str(exc);after=exposed
    event['after']=after['decision']
    return after,event

def plan_for(root,tasks,stage,max_cny,model=None):
    files=[Path(__file__),ROOT/'experiments/bound_opportunity_probe.py',ROOT/'experiments/bound_verdict_adapter.py']+list((ROOT/'experiments/consensus_mechanism').glob('*.py'))+list((ROOT/'experiments/paired_deliberation').glob('*.py'))+[ROOT/'experiments/consensus_recommendation/common.py',ROOT/'experiments/evidence_consensus/schema.py']
    plan={'version':VERSION,'stage':stage,'task_hash':fingerprint(tasks),'provider':'qwen_flash',
          'physical_cap':max(1,len(tasks)*(4 if stage=='panel'else len(METHODS)*2)),
          'max_cny':max_cny,'reuse_root':None,'model_sha256':sha(model)if model else None,
          'source_sha256':{str(f.relative_to(ROOT)):sha(f)for f in sorted(set(files))}}
    create(root/'plan.json',plan)
    for name,h in plan['source_sha256'].items():
        dest=root/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists()and sha(dest)!=h:raise ValueError('source snapshot mismatch')
        dest.write_bytes((ROOT/name).read_bytes())
    return plan

def collect(root,tasks,max_cny=1.):
    with locked(root):
        plan=plan_for(root,tasks,'panel',max_cny);api=Runtime(root,plan,'qwen_flash')
        for i,t in enumerate(tasks):
            dest=root/'cases'/f'case_{i}.json'
            if dest.exists():continue
            state=c.fresh(t);branches={}
            for name,ids in BRANCHES.items():
                after,event=actor(api,f'{i}_{name}',t,state,ids)
                branches[name]={'decision':after['decision'],'event':event,'features':features(t,state,ids)}
            create(dest,{'task_id':t['id'],'partition':t['metadata']['acquisition_partition'],
                         'initial':state['decision'],'branches':branches})
            print('panel',i+1,len(tasks),t['id'],flush=True)
            if i==3:
                first=[read(root/'cases'/f'case_{j}.json')for j in range(4)]
                if sum(v['event']['valid']for x in first for v in x['branches'].values())<13:
                    raise RuntimeError('engineering gate <13/16 valid; preserve responses and stop')
        cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(tasks))]
        create(root/'generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]})
        create(root/'physical.json',audit(root,'qwen_flash'))
    return cases

def panel_cases(root,tasks):
    cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(tasks))]
    if read(root/'generation_complete.json')!={'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]}:raise ValueError('panel seal mismatch')
    return cases

class OutcomeModel:
    def fit(self,X,y):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vectorizer=DictVectorizer();matrix=self.vectorizer.fit_transform(X)
        self.classes=np.array(sorted(set(y)));self.model=None
        if len(self.classes)>1:self.model=LogisticRegression(C=1.,max_iter=1000,random_state=0).fit(matrix,y)
        self.prior={label:sum(v==label for v in y)/len(y)for label in self.classes};self.temperature=1.
        return self
    def probs(self,X,prior=False):
        if self.model is None or prior:p=np.array([[self.prior[v]for v in self.classes]for _ in X])
        else:p=self.model.predict_proba(self.vectorizer.transform(X))
        p=np.maximum(p,1e-9)**(1/self.temperature);p/=p.sum(axis=1,keepdims=True)
        return [{int(v):float(q)for v,q in zip(self.classes,row)}for row in p]

def fit(root,tasks):
    import joblib
    train=[t for t in tasks if t['metadata']['acquisition_partition']!='test']
    cases=panel_cases(root/'train',train);labels=read(root/'train.labels.private.json');observations=[]
    for t,case in zip(train,cases):
        label=labels[t['id']];base=c.quality(case['initial'],label)
        for name,arm in case['branches'].items():
            observations.append({'task_id':t['id'],'partition':case['partition'],'branch':name,
              'features':arm['features'],'valid':arm['event']['valid'],
              'reward':int(c.quality(arm['decision'],label)-base)if arm['event']['valid']else None,
              'evidence_reward_vs_empty':c.quality(arm['decision'],label)-c.quality(case['branches']['empty']['decision'],label)
                  if arm['event']['valid']and case['branches']['empty']['event']['valid']else None})
    fit_rows=[r for r in observations if r['partition']=='fit']
    good=[r for r in fit_rows if r['valid']]
    if len(good)<8:raise ValueError('insufficient valid training actions')
    outcome=OutcomeModel().fit([r['features']for r in good],[r['reward']for r in good])
    validity=OutcomeModel().fit([r['features']for r in fit_rows],[int(r['valid'])for r in fit_rows])
    cal=[r for r in observations if r['partition']=='calibration'and r['valid']]
    scores={}
    if cal:
        for temperature in [.5,1.,2.,4.]:
            outcome.temperature=temperature;p=outcome.probs([r['features']for r in cal])
            scores[str(temperature)]=float(np.mean([-math.log(max(prob.get(r['reward'],1e-9),1e-9))for prob,r in zip(p,cal)]))
        outcome.temperature=float(min(scores,key=scores.get))
    bundle={'outcome':outcome,'validity':validity,'cost_weight':.1,
            'training_task_ids':[t['id']for t in train],
            'target':'expected decision gain vs current frozen state; evidence-only increment reported separately',
            'scope':'single reviewer calibration shared across allocation members; no heterogeneous-skill claim'}
    dest=root/'model/model.joblib';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise ValueError('model already exists; use frozen artifact instead of refitting')
    joblib.dump(bundle,dest)
    create(root/'model/observations.json',observations)
    create(root/'model/summary.json',{'valid_fit_actions':len(good),'invalid_fit_excluded':len(fit_rows)-len(good),
       'fit_reward_counts':dict(collections.Counter(r['reward']for r in good)),
       'calibration_nll':scores,'temperature':outcome.temperature,
       'fit_claim_ids':[t['id']for t in train if t['metadata']['acquisition_partition']=='fit'],
       'calibration_claim_ids':[t['id']for t in train if t['metadata']['acquisition_partition']=='calibration'],
       'test_labels_used':False})
    return bundle

def action_values(t,state,actions,bundle,prior=False):
    X=[features(t,state,a['source_ids'])for a in actions]
    outcomes=bundle['outcome'].probs(X,prior);validity=bundle['validity'].probs(X,prior)
    values=[]
    for a,p,pv in zip(actions,outcomes,validity):
        valid=pv.get(1,0.);gain=valid*(p.get(1,0.)-p.get(-1,0.))
        cny=.0005+sum(len(t['source_catalog'][s]['text'])for s in a['source_ids'])*.15/3e6
        values.append({**a,'gains':[gain]*3,'cost':bundle['cost_weight']*cny,
                       'predicted_outcome':p,'predicted_valid':valid,'predicted_cny':cny})
    return values

def select(t,state,bundle,old,mode,calls,sources):
    actions=c.actions(t,state,allow_joint=True)
    if not actions:return [],{'reason':'no_active_action'}
    values=old_estimate(t,state,actions,[],old)if mode=='old_game'else action_values(t,state,actions,bundle,mode=='prior_game')
    if mode=='old_game':
        for v in values:v['cost']*=bundle['cost_weight']
    solver=SetAllocator(values,calls,sources,.02)
    solver_mode=mode.split('_')[-1]if mode in ('old_game','new_game','new_greedy','new_central','prior_game')else mode
    if mode=='random':
        feasible=[s for s in itertools.product([None]+list(range(len(values))),repeat=3)if solver.feasible(s)and any(q is not None for q in s)]
        assignment=random.Random(int(fingerprint([t['id'],state,calls])[:16],16)).choice(feasible)if feasible else(None,None,None)
        allocation={'assignment':assignment,'potential':solver.potential(assignment),'mode':'random'}
    else:allocation=solver.solve(solver_mode)
    allocation.pop('solver_seconds',None)
    chosen=[{**values[q],'reviewer':i,'marginal':solver.utility(allocation['assignment'],i)}for i,q in enumerate(allocation['assignment'])if q is not None]
    chosen.sort(key=lambda a:(-a['marginal'],a['id'],a['reviewer']))
    return chosen,{'allocation':allocation,'all_values':values}

def execute(t,api,bundle,old,method,index):
    state=c.fresh(t);events=[];traces=[];calls=2;sources=2
    while calls>0:
        if method=='reflection':ids=[]
        else:
            if sources<=0:break
            chosen,trace=select(t,state,bundle,old,method,calls,sources);traces.append(trace)
            if not chosen:break
            ids=chosen[0]['source_ids']
        before=copy.deepcopy(state);state,event=actor(api,f'{index}_{method}_{2-calls}',t,state,ids)
        event['state_before']=before;events.append(event);calls-=1;sources-=len(ids)
        if not state['claims'].get('target',{}).get('active',True):break
    return {'decision':state['decision'],'events':events,'traces':traces,'calls':len(events),'source_reads':2-sources}

def evaluate(root,tasks,max_cny=1.):
    import joblib
    test=[t for t in tasks if t['metadata']['acquisition_partition']=='test']
    model=root/'model/model.joblib';bundle=joblib.load(model)
    if set(t['id']for t in test)&set(bundle['training_task_ids']):raise ValueError('train/test overlap')
    old=joblib.load(ROOT/'experiments/results/consensus_mechanism_paper_v2/model/model.joblib')
    dest=root/'evaluation'
    with locked(dest):
        plan=plan_for(dest,test,'evaluation',max_cny,model)
        create(dest/'old_model_sha256.json',{'sha256':sha(ROOT/'experiments/results/consensus_mechanism_paper_v2/model/model.joblib')})
        first={t['id']:{m:select(t,c.fresh(t),bundle,old,m,2,2)for m in METHODS if m!='reflection'}for t in test}
        create(dest/'selection.before_generation.json',first)
        api=Runtime(dest,plan,'qwen_flash')
        for i,t in enumerate(test):
            arms={}
            for method in METHODS:
                path=dest/'arms'/f'{i}_{method}.json'
                if not path.exists():create(path,execute(t,api,bundle,old,method,i))
                arms[method]=read(path)
            create(dest/'cases'/f'case_{i}.json',{'task_id':t['id'],'methods':arms})
            print('evaluate',i+1,len(test),flush=True)
        cases=[read(dest/'cases'/f'case_{i}.json')for i in range(len(test))]
        create(dest/'generation_complete.json',{'task_ids':[t['id']for t in test],'hashes':[fingerprint(x)for x in cases]})
    labels=read(root/'test.labels.private.json');summary={'tasks':len(test),'methods':{},'physical':audit(dest,'qwen_flash')}
    for method in METHODS:
        before=[c.quality(t['initial'],labels[t['id']])for t in test]
        after=[c.quality(case['methods'][method]['decision'],labels[t['id']])for t,case in zip(test,cases)]
        events=[e for case in cases for e in case['methods'][method]['events']]
        summary['methods'][method]={'correct':int(sum(after)),'initial_correct':int(sum(before)),
             'improved':sum(b>a for a,b in zip(before,after)),'harmed':sum(b<a for a,b in zip(before,after)),
             'calls':len(events),'invalid':sum(not e['valid']for e in events),
             'quality':float(np.mean(after))}
    create(dest/'summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','fit','evaluate','all'])
    p.add_argument('--root',type=Path,default=ROOT/'experiments/results/calibrated_action_value_v2')
    p.add_argument('--run',action='store_true');p.add_argument('--max-cny',type=float,default=1.)
    a=p.parse_args()
    if not 0<a.max_cny<=2:p.error('hard cap 0..2 CNY per stage')
    if a.stage=='prepare':tasks=prepare(a.root)
    else:tasks=read(a.root/'tasks.json')
    if not a.run:print(json.dumps({'stage':a.stage,'tasks':len(tasks),'network':False,'max_cny_per_stage':a.max_cny}));return
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    if a.stage in ('train','all'):collect(a.root/'train',[t for t in tasks if t['metadata']['acquisition_partition']!='test'],a.max_cny)
    if a.stage in ('fit','all')and not(a.root/'model/model.joblib').exists():fit(a.root,tasks)
    if a.stage in ('evaluate','all'):print(json.dumps(evaluate(a.root,tasks,a.max_cny),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
