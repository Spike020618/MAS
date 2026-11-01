"""Precommitted train-side random source-action acquisition; one real review/query."""
import argparse,json,hashlib,copy,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.evidence_consensus import runner_v6 as engine
from experiments.evidence_consensus.grounded_update import grounded_patch
from experiments.evidence_consensus.task_adapter import load_evaluation_labels
from experiments.evidence_consensus.action_calibration import quality
from experiments.paired_deliberation.native_optimization_runtime import Runtime,audit

OUT=ROOT/'experiments/results/e2_action_acquisition_160_live_v1'
INPUTS=ROOT/'experiments/results/e2_action_acquisition_rec80_v1/tasks.json'
QA_INPUTS=ROOT/'experiments/results/e2_qa80_corpus_v2/tasks.json'

def run(root=OUT):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False);root=Path(root);root.mkdir(parents=True,exist_ok=True)
 with (root/'run.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  tasks=[engine.bounded_task(t)for path in (QA_INPUTS,INPUTS)for t in json.load(open(path))];actions=[]
  for t in tasks:
   state=engine.common_state(t);available=[sid for sid in state['catalog']if sid not in state['visible_sources']]
   sid=min(available,key=lambda s:hashlib.sha256(('e2-explore-source:'+t['id']+s).encode()).hexdigest())
   actions.append({'task_id':t['id'],'source_id':sid,'selection':'uniform hash exploration across unseen permitted sources, no labels'})
  files=list((ROOT/'experiments/evidence_consensus').glob('*.py'))+[Path(__file__)]
  plan={'version':'e2-native160-action-exploration-v1','task_ids':[t['id']for t in tasks],'inputs_sha256':sha(INPUTS),'qa_inputs_sha256':sha(QA_INPUTS),'initial_QA_call':'one actual independent generation before source review; no abstain baseline inflation','actions_hash':fingerprint(actions),'query_labels_available':False,'physical_cap':300,'max_cny':3.,'scope':'train-side random action observations, not method confirmation','source_sha256':{str(p.relative_to(ROOT)):sha(p)for p in files}}
  create(root/'plan.json',plan);create(root/'inputs.json',tasks);create(root/'action_commitment.json',actions)
  for name,h in plan['source_sha256'].items():
   p=root/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists()and sha(p)!=h:raise ValueError('source snapshot changed')
   p.write_bytes((ROOT/name).read_bytes())
  api=Runtime(root,plan,cap=300,max_cny=3.);results={}
  def collect(i):
   path=root/'cases'/f'case_{i}.json'
   if path.exists():return json.load(open(path))
   task=tasks[i];state=engine.common_state(task);ledger=engine.BudgetLedger(2,150000,.1);baseline_event=None
   if task['kind']=='qa':
    baseline_event=engine.call(api,f'e2_initial_{i}',engine.request(task,state,'initial',0,()),ledger,task,state)
    initial=baseline_event['report']['decision']if baseline_event['valid']else task['initial']
   else:initial=task['initial']
   state=engine.initialize_decision(state,initial);before=copy.deepcopy(state['decision']);sid=actions[i]['source_id'];fetch=engine.fetch_source(state,sid)
   req=engine.request(task,state,'review',0,())
   event=engine.call(api,f'e2_{i}',req,ledger,task,state);event['source_action']=fetch;event['proposal']={'source_id':sid,'score':0.,'source_characters':len(state['catalog'][sid]['text'])}
   result={'accepted':False,'reason':'invalid_response','state':state}
   if event['valid']:
    try:result=grounded_patch(engine.register_report(state,event['report']),event['report'],required_source=sid)
    except (ValueError,KeyError,TypeError)as exc:result={'accepted':False,'reason':str(exc),'state':state}
   row={'task_id':task['id'],'kind':task['kind'],'before_decision':before,'after_decision':result['state']['decision'],'event':event,'baseline_event':baseline_event,'accepted':result['accepted'],'reason':result['reason'],'budget':ledger.snapshot(),'labels_available':False}
   create(path,row);print('sealed',i+1,'/160',event['valid'],result['reason'],flush=True);return row
  first=list(range(8))+list(range(80,88))
  for i in first:results[i]=collect(i)
  create(root/'probe_complete.json',{'indices':first,'scope':'interface probe only, labels unavailable; no effect-based filtering','valid':sum(results[i]['event']['valid']for i in first)})
  with ThreadPoolExecutor(max_workers=2)as pool:
   jobs={pool.submit(collect,i):i for i in range(len(tasks))if i not in results}
   for f in as_completed(jobs):results[jobs[f]]=f.result()
  create(root/'generation_complete.json',{'task_ids':plan['task_ids'],'hashes':[fingerprint(results[i])for i in range(len(tasks))],'labels_available':False})
  labels=load_evaluation_labels(tasks,generation_complete=root/'generation_complete.json');create(root/'labels.after_generation.json',labels)
  rewards=[{'task_id':r['task_id'],'reward':quality(r['after_decision'],labels[r['task_id']])-quality(r['before_decision'],labels[r['task_id']]),'valid':r['event']['valid'],'accepted':r['accepted']}for r in [results[i]for i in range(len(tasks))]]
  create(root/'rewards.after_generation.json',rewards);summary={'tasks':len(tasks),'actual_action_calls':len(tasks),'valid':sum(r['valid']for r in rewards),'accepted':sum(r['accepted']for r in rewards),'positive':sum(r['reward']>0 for r in rewards),'negative':sum(r['reward']<0 for r in rewards),'physical':audit(root),'scope':plan['scope']};create(root/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
 if a.run:run(a.output)
 else:print('Explicit --run required; no API calls.')
