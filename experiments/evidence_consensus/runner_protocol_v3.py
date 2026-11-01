"""V3 real complementary inputs, explicit quoted review and bounded six-arm comparison."""
import argparse,collections,copy,fcntl,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from . import protocol_v3 as p
from .allocation_v3 import Allocation
from . import runner_protocol_v2 as old
from .state import register_report,initialize_decision
from .actions import BudgetLedger,fetch_source
from .task_adapter import load_evaluation_labels
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.native_optimization_runtime import Runtime,audit
METHODS=('single','independent','fixed','score_greedy','central','game')

def ledger():return BudgetLedger(5,500000,1.)
def initialize(task,api,index,single=False):
 events=[];reports=[];states=[];budget=ledger();state=p.common_state(task)
 for i in range(3):
  if single:
   sid=task['metadata']['protocol_v3']['initial_source_ids'][i];fetch_source(state,sid)
   req=p.request(task,state,'initial',i,reports)
  else:
   state=p.initial_state(task,i);req=p.request(task,state,'initial',i)
  event=p.call(api,f'case{index}_{"single"if single else "shared"}_initial{i}',req,budget,task,state);events.append(event)
  report=event['report']if event['valid']else {'A':[],'decision':copy.deepcopy(task['initial'])}
  reports.append(report);states.append(copy.deepcopy(state))
  if single and event['valid']:
   state['decision']=old.validate_decision(task,report['decision'])
   if task['kind']=='rec':state['decision']+= [m for m in old.candidate_ids(task)if m not in state['decision']]
   state=register_report(state,report)
 if single:
  state=initialize_decision(state,copy.deepcopy(state['decision']));return state,reports,events,states,budget
 state=p.common_state(task)
 for sid in task['metadata']['protocol_v3']['initial_source_ids']:fetch_source(state,sid)
 state=initialize_decision(state,old.aggregate(task,[r for e,r in zip(events,reports)if e['valid']]))
 for e,r in zip(events,reports):
  if e['valid']:state=register_report(state,r)
 return state,reports,events,states,budget

def select(task,state,rows,reports,method,bundle=None):
 values=[];feature_rows=[]
 for row in rows:
  fs=[p.features(task,state,row,reports,i)for i in range(3)];feature_rows.append(fs)
  if bundle:
   gains=[float(bundle['gain'].predict([f])[0]+bundle['gain_bias'])for f in fs]
  else:
   gains=[.05+.15*f['retrieval_score']+.1*f['reviewer_claim_title_overlap']for f in fs]
  values.append({'id':fingerprint([row['source_id'],row['target_claim_ids']]),'unit':fingerprint(state['catalog'][row['source_id']]['text']),'targets':row['target_claim_ids'],'gains':gains,'cost':.003+4*row['source_characters']*.15/1e6})
 allocation=Allocation(values);solved=allocation.solve()
 if method in ('game','central'):s=solved[method]
 elif method=='score_greedy':s=allocation.greedy()
 else:s=allocation.fixed()
 return {'actions':[(i,q)for i,q in enumerate(s)if q is not None],'solver':solved,'candidate_count':len(rows),'action_cap':2,'estimated_actions':values,'features':feature_rows}

def task_case(index,task,api,root,bundle=None):
 path=root/'cases'/f'case_{index}.json'
 if path.exists():return json.loads(path.read_text())
 base,reports,initial,private,budget=initialize(task,api,index);methods={}
 for method in METHODS:
  state=copy.deepcopy(base);events=copy.deepcopy(initial);own=reports;led=ledger()
  if method=='single':state,own,events,_,led=initialize(task,api,index,True)
  else:
   for i,e in enumerate(initial):
    usage=e.get('usage',{});led.reserve(f'shared_{i}',calls=1,tokens=60000,cny=.1);led.settle(f'shared_{i}',calls=1,tokens=sum(usage.get(k,0)for k in ('prompt_tokens','completion_tokens')),cny=old.cost(usage,api.profile),known=bool(usage))
  initial_decision=copy.deepcopy(state['decision'])
  if method=='independent':
   finals=[];allocation={'actions':[],'scope':'at most two own-report reviews; no peer information'}
   # Precommitted rotating reviewers. The third retains its initial decision.
   start=int(fingerprint(task['id'])[:8],16)%3;chosen={(start+j)%3 for j in range(2)}
   for i in range(3):
    local=copy.deepcopy(private[i]);local=initialize_decision(local,old.aggregate(task,[reports[i]]if initial[i]['valid']else []))
    if initial[i]['valid']:local=register_report(local,reports[i])
    if i in chosen:
     rows=p.proposals(task,local,[reports[i]])
     # All arms use the same public pool of three initially allocated sources,
     # plus at most two distinct additional cards; do not fetch a peer's initial card.
     rows=[r for r in rows if r['source_id']not in task['metadata']['protocol_v3']['initial_source_ids']]
     if rows:
      row=max(rows,key=lambda r:(r['score'],r['source_id']));allocation['actions'].append([i,row['source_id']]);local,event=p.review(api,f'case{index}_independent_review{i}',task,local,row,[reports[i]],i,led);events.append(event)
    finals.append({'decision':local['decision']})
   state['decision']=old.aggregate(task,finals)
  else:
   rows=p.proposals(task,state,own);allocation=select(task,state,rows,own,'fixed'if method=='single'else method,bundle)
   for step,(reviewer,q)in enumerate(allocation['actions']):
    state,event=p.review(api,f'case{index}_{method}_review{step}',task,state,rows[q],own,reviewer,led);events.append(event)
  methods[method]={'decision':state['decision'],'initial_decision':initial_decision,'state':state,'allocation':allocation,'events':events,'logical_calls':len(events),'budget':led.snapshot(),'semantic_truth_certified':False}
 result=json.loads(json.dumps({'task_id':task['id'],'kind':task['kind'],'methods':methods,'labels_available':False}));create(path,result);print('completed',index+1,task['id'],{m:a['logical_calls']for m,a in methods.items()},flush=True);return result

def prepare(root,inputs,estimator='heuristic',model=None):
 tasks=json.loads(inputs.read_text());commit=json.loads((inputs.parent/'split_commitment.json').read_text())
 if commit.get('task_split')!={t['id']:t['metadata']['acquisition_partition']for t in tasks}:raise ValueError('evaluation commitment mismatch')
 if not tasks or any(t['metadata']['acquisition_partition']!='diagnostic'for t in tasks):raise ValueError('evaluation accepts only precommitted diagnostic tasks')
 bundle=None
 if estimator=='learned':
  import joblib
  if model is None:raise ValueError('model required')
  bundle=joblib.load(model)
  if bundle.get('feature_schema')!='v3-query-action-reviewer-v1':raise ValueError('v3 model schema required')
  lineage=json.loads((model.parent/'split_commitment.json').read_text());used=set(lineage['task_split'])
  if used&{t['id']for t in tasks}:raise ValueError('training/evaluation task overlap')
  from experiments.prepare_disjoint_acquisition import group
  training_tasks=json.loads((model.parent/'training_tasks.json').read_text())
  if {group(t)for t in training_tasks}&{group(t)for t in tasks}:raise ValueError('native group train/evaluation overlap')
 files=sorted(set(list(Path(__file__).parent.glob('*.py'))+[ROOT/'experiments/prepare_protocol_v3.py',ROOT/'experiments/fit_protocol_v3.py',ROOT/'experiments/acquire_protocol_v3.py',ROOT/'experiments/report_protocol_v3.py',ROOT/'experiments/prepare_protocol_v3_stress.py',ROOT/'experiments/consensus_recommendation/common.py',ROOT/'experiments/core/mechanism/title_anchor_retrieval.py',ROOT/'experiments/core/mechanism/evidence_acquisition.py',ROOT/'scripts/run_protocol_v3_pipeline.sh',ROOT/'scripts/run_protocol_v3_stress.sh']+list((ROOT/'experiments/paired_deliberation').glob('*.py'))))
 plan={'version':'protocol-v3','count':len(tasks),'task_ids':[t['id']for t in tasks],'input_sha256':sha(inputs),'split_sha256':sha(inputs.parent/'split_commitment.json'),'estimator':estimator,'model_sha256':sha(model)if model else None,'reuse_root':str(root.parent/'heuristic')if estimator=='learned'else None,'methods':list(METHODS),'physical_cap':1000,'max_cny':5.,'source_sha256':{str(f.relative_to(ROOT)):sha(f)for f in files},'scope':'development task heldout; public background shared; no semantic certification','additional_review_cap':2,'initial_source_allocations':3}
 create(root/'inputs.json',tasks);create(root/'plan.json',plan)
 if model:
  dst=root/'estimator_model.joblib';blob=model.read_bytes()
  if dst.exists()and dst.read_bytes()!=blob:raise ValueError('model snapshot changed')
  dst.write_bytes(blob)
 for name,h in plan['source_sha256'].items():
  f=root/'source_snapshot'/name
  if f.exists()and sha(f)!=h:raise ValueError('source snapshot changed; use new version')
  f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((ROOT/name).read_bytes())
 return tasks,plan,bundle

def score(root):
 plan=json.loads((root/'plan.json').read_text());tasks=json.loads((root/'inputs.json').read_text());seal=json.loads((root/'generation_complete.json').read_text());cases=[json.loads((root/'cases'/f'case_{i}.json').read_text())for i in range(len(tasks))]
 if seal['hashes']!=[fingerprint(c)for c in cases]:raise ValueError('seal mismatch')
 labels=load_evaluation_labels(tasks,generation_complete=root/'generation_complete.json');create(root/'labels.after_generation.json',labels);summary={'scope':plan['scope'],'methods':{}}
 def quality(decision,label):
  if label['kind']=='qa':return int(decision==label['answer'])
  return sum(str(x)in decision[:10]for x in label['targets'])/len(label['targets'])
 for method in METHODS:
  st=collections.Counter();paired={}
  for c in cases:
   arm=c['methods'][method];label=labels[c['task_id']];d=arm['decision'];prior=arm['initial_decision'];st['tasks']+=1;st['logical_calls']+=arm['logical_calls'];st['initial_correct_or_recall_sum']+=quality(prior,label)
   if c['kind']=='qa':st['qa_tasks']+=1;st['qa_correct']+=d==label['answer']
   else:st['rec_tasks']+=1;st['rec_targets']+=len(label['targets']);st['rec_hits_at_10']+=sum(str(x)in d[:10]for x in label['targets'])
   st['tasks_improved_from_own_initial']+=quality(d,label)>quality(prior,label);st['tasks_harmed_from_own_initial']+=quality(d,label)<quality(prior,label)
   for e in arm['events']:
    st['valid_events']+=bool(e.get('valid'));st['accepted_patches']+=bool(e.get('patch_accepted'));st['invalid_events']+=not e.get('valid',False)
    if 'source_action'in e:
     st['review_calls']+=1;st['review_abstentions']+=e.get('valid',False)and e['report']['C']['proposed_patch']['operation']=='abstain'
     if e.get('valid'):
      st['quoted_findings']+=sum(bool(f['quote'])for f in e['report']['findings'])
   other=c['methods']['independent']['decision'];paired[c['task_id']]=quality(d,label)-quality(other,label)
  summary['methods'][method]={**dict(st),'qa_accuracy':st['qa_correct']/st['qa_tasks']if st['qa_tasks']else None,'rec_recall_at_10':st['rec_hits_at_10']/st['rec_targets']if st['rec_targets']else None,'paired_vs_independent':{'improved_tasks':sum(v>0 for v in paired.values()),'harmed_tasks':sum(v<0 for v in paired.values()),'differences':paired}}
 create(root/'summary.json',summary);create(root/'physical_summary.json',audit(root));return summary

def run(root,inputs,estimator,model):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False);root.mkdir(parents=True,exist_ok=True)
 with (root/'run.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tasks,plan,bundle=prepare(root,inputs,estimator,model)
  api=Runtime(root,plan,cap=plan['physical_cap'],max_cny=plan['max_cny']);results={}
  with ThreadPoolExecutor(max_workers=2)as pool:
   jobs={pool.submit(task_case,i,t,api,root,bundle):i for i,t in enumerate(tasks)}
   for future in as_completed(jobs):results[jobs[future]]=future.result()
  create(root/'generation_complete.json',{'task_ids':plan['task_ids'],'hashes':[fingerprint(results[i])for i in range(len(tasks))]})
  print(json.dumps(score(root),ensure_ascii=False,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--inputs',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--estimator',choices=['heuristic','learned'],default='heuristic');a.add_argument('--model',type=Path);a.add_argument('--run',action='store_true');args=a.parse_args()
 if args.run:run(args.output,args.inputs,args.estimator,args.model)
 else:_,plan,_=prepare(args.output,args.inputs,args.estimator,args.model);print(json.dumps(plan,indent=2))
