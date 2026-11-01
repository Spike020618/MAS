"""Random-action training acquisition with a contract-only probe, no evaluation tasks."""
import argparse,collections,copy,fcntl,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.evidence_consensus import protocol_v3 as p
from experiments.evidence_consensus.runner_protocol_v3 import initialize,ledger
from experiments.evidence_consensus.task_adapter import load_evaluation_labels
from experiments.evidence_consensus.protocol_v3 import quality
from experiments.paired_deliberation.native_optimization_runtime import Runtime,audit

def prepare(root,inputs):
 tasks=json.loads(inputs.read_text());commit=json.loads((inputs.parent/'split_commitment.json').read_text())
 if not tasks or commit['task_split']!={t['id']:t['metadata']['acquisition_partition']for t in tasks}:raise ValueError('split mismatch')
 if any(v not in ('fit','calibration')for v in commit['task_split'].values()):raise ValueError('never acquire diagnostic training actions')
 groups={}
 from experiments.prepare_disjoint_acquisition import group
 for t in tasks:
  g=group(t);v=commit['task_split'][t['id']]
  if groups.setdefault(g,v)!=v:raise ValueError('native group crosses training splits')
 files=list((ROOT/'experiments/evidence_consensus').glob('*.py'))+[Path(__file__).resolve(),ROOT/'experiments/prepare_protocol_v3.py',ROOT/'experiments/fit_protocol_v3.py',ROOT/'experiments/report_protocol_v3.py',ROOT/'experiments/prepare_protocol_v3_stress.py',ROOT/'experiments/consensus_recommendation/common.py',ROOT/'experiments/core/mechanism/title_anchor_retrieval.py',ROOT/'experiments/core/mechanism/evidence_acquisition.py',ROOT/'scripts/run_protocol_v3_pipeline.sh',ROOT/'scripts/run_protocol_v3_stress.sh']+list((ROOT/'experiments/paired_deliberation').glob('*.py'))
 plan={'version':'protocol-v3-training','task_ids':[t['id']for t in tasks],'input_sha256':sha(inputs),'split_sha256':sha(inputs.parent/'split_commitment.json'),'physical_cap':800,'max_cny':5.,'selection':'uniform stable hash over unseen rows and reviewer; never outcome-selected','source_sha256':{str(f.relative_to(ROOT)):sha(f)for f in sorted(set(files))}}
 create(root/'inputs.json',tasks);create(root/'plan.json',plan);create(root/'acquisition_split_commitment.json',commit)
 for name,h in plan['source_sha256'].items():
  dst=root/'source_snapshot'/name
  if dst.exists()and sha(dst)!=h:raise ValueError('source changed; use a new version')
  dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/name).read_bytes())
 return tasks,plan

def collect(index,task,root,api):
 path=root/'cases'/f'case_{index}.json'
 if path.exists():return json.loads(path.read_text())
 state,reports,initial,_,led=initialize(task,api,index);rows=p.proposals(task,state,reports);before=copy.deepcopy(state['decision']);event={'valid':False,'error':'no_registered_claim_or_source'};fs=None
 if rows:
  row=min(rows,key=lambda r:fingerprint([task['id'],r['source_id']]))
  reviewer=int(fingerprint(['reviewer',task['id']])[:8],16)%3;fs=p.features(task,state,row,reports,reviewer)
  state,event=p.review(api,f'train_{index}_review',task,state,row,reports,reviewer,led)
 case={'task_id':task['id'],'kind':task['kind'],'initial_events':initial,'before_decision':before,'after_decision':state['decision'],'event':event,'pre_action_features':fs,'labels_available':False};create(path,case);print('sealed training',index+1,task['id'],event['valid'],flush=True);return case

def gate(cases):
 events=[e for c in cases for e in c['initial_events']];n=len(cases);iv=sum(e['valid']for e in events);rv=sum(c['event']['valid']for c in cases)
 return {'probe_tasks':n,'initial_valid':iv,'review_valid':rv,'passed':iv>=.8*len(events)and rv>=.8*n,'criterion':'>=80% structurally valid real initial claims and assigned-source findings; no correctness/positive-reward gate','labels_available':False}

def run(root,inputs,probe_only=False):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False);root.mkdir(parents=True,exist_ok=True)
 with (root/'run.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tasks,plan=prepare(root,inputs);api=Runtime(root,plan,cap=800,max_cny=5.)
  first=[]
  for kind in ('qa','rec'):first.extend([i for i,t in enumerate(tasks)if t['kind']==kind][:3])
  if len(first)<6:raise ValueError('probe requires three tasks of each kind, no fallback')
  results={i:collect(i,tasks[i],root,api)for i in first};g=gate(list(results.values()));create(root/'probe_gate.json',g);print(json.dumps(g,indent=2))
  if not g['passed']:raise RuntimeError('contract gate failed; saved all responses, no full acquisition')
  if probe_only:return
  with ThreadPoolExecutor(max_workers=2)as pool:
   jobs={pool.submit(collect,i,t,root,api):i for i,t in enumerate(tasks)if i not in results}
   for future in as_completed(jobs):results[jobs[future]]=future.result()
  create(root/'generation_complete.json',{'task_ids':plan['task_ids'],'hashes':[fingerprint(results[i])for i in range(len(tasks))]})
  labels=load_evaluation_labels(tasks,generation_complete=root/'generation_complete.json');create(root/'labels.after_generation.json',labels)
  rewards=[{'task_id':c['task_id'],'reward':quality(c['after_decision'],labels[c['task_id']])-quality(c['before_decision'],labels[c['task_id']]),'valid':c['event']['valid']}for c in [results[i]for i in range(len(tasks))]]
  create(root/'rewards.after_generation.json',rewards);create(root/'summary.json',{'tasks':len(tasks),'valid':sum(r['valid']for r in rewards),'positive':sum(r['reward']>0 for r in rewards),'negative':sum(r['reward']<0 for r in rewards),'physical':audit(root)})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--inputs',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--run',action='store_true');a.add_argument('--probe-only',action='store_true');args=a.parse_args()
 if args.run:run(args.output,args.inputs,args.probe_only)
 else:ts,plan=prepare(args.output,args.inputs);print(json.dumps({'tasks':len(ts),'model_api_calls':0,'physical_cap':plan['physical_cap']},indent=2))
