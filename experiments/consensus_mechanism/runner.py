"""Complete native train/evaluate lifecycle with global seals and resumed receipts."""
import argparse,collections,copy,fcntl,json,math,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.prepare_disjoint_acquisition import group
from experiments.paired_deliberation.native_optimization_runtime import audit
from . import core as c
from .engine import initialize,execute,PRIMARY,EXTERNAL,ABLATIONS
from .runtime import Runtime
from .data import read,labels_for


def frozen_plan(root,tasks,stage,provider,budget=2,methods=(),model=None,reuse=None,max_cny=5.):
 if not tasks:raise ValueError('empty tasks')
 if budget not in (0,1,2,3):raise ValueError('budget must be 0..3')
 files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/run_consensus_mechanism.sh',ROOT/'experiments/consensus_recommendation/common.py']+list((ROOT/'experiments/paired_deliberation').glob('*.py'))
 plan={'version':'evidence-set-consensus-release-3','stage':stage,'task_ids':[t['id']for t in tasks],'tasks_hash':fingerprint(tasks),'provider':provider,'additional_calls':budget,'additional_sources':budget,'methods':list(methods),'model_sha256':sha(model)if model else None,'physical_cap':max(100,len(tasks)*(7 if stage=='train'else 12+(budget+3)*max(1,len(methods)))),'max_cny':max_cny,'reuse_root':str(reuse)if reuse else None,'source_sha256':{str(f.relative_to(ROOT)):sha(f)for f in sorted(set(files))},'scope':'native task/group-heldout relative to contribution training; data history disclosed separately','labels_available_during_generation':False}
 from experiments.evidence_consensus.task_adapter import QA_LABELS,REC_LABELS,ENTITIES
 label_paths={Path(t['metadata']['evaluation_vault'])for t in tasks if t['metadata'].get('evaluation_vault')}
 if any(t['kind']=='qa'and not t['metadata'].get('evaluation_vault')for t in tasks):label_paths.add(QA_LABELS)
 if any(t['kind']=='rec'for t in tasks):label_paths.update((REC_LABELS,ENTITIES))
 plan['label_sources_sha256']={str(p):sha(p)for p in sorted(label_paths)}
 create(root/'inputs.json',tasks);create(root/'plan.json',plan)
 for name,h in plan['source_sha256'].items():
  dst=root/'source_snapshot'/name
  if dst.exists()and sha(dst)!=h:raise ValueError('frozen source mismatch')
  dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/name).read_bytes())
 if model:
  blob=model.read_bytes();dst=root/'estimator.joblib'
  if dst.exists()and dst.read_bytes()!=blob:raise ValueError('model snapshot mismatch')
  dst.write_bytes(blob)
 return plan

def locked(root):
 root.mkdir(parents=True,exist_ok=True);lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return lock

def training_case(index,task,root,api):
 path=root/'cases'/f'case_{index}.json'
 if path.exists():return read(path)
 state,reports,initial,_=initialize(task,index,api);rows=c.actions(task,state)
 if not rows:
  result={'task_id':task['id'],'kind':task['kind'],'initial_events':initial,'baseline':state,'features':None,'control':None,'review':None,'before':state['decision'],'after':state['decision'],'control_decision':state['decision'],'reason':'no_registered_claim','labels_available':False}
 else:
  pairs=[r for r in rows if len(r['source_ids'])==2]
  if not pairs:raise ValueError('matched training requires two unseen sources')
  row=min(pairs,key=lambda a:fingerprint(['explore',task['id'],a['id']]))
  reviewer=int(fingerprint(['owner',task['id']])[:8],16)%3;target=row['target_id']
  control=copy.deepcopy(state);control,event0=c.invoke(api,f'{index}_control',task,control,'review',80,reports,{'target_id':target,'source_ids':[]},reviewer=reviewer)
  branches=[]
  for j,ids in enumerate(([row['source_ids'][0]],[row['source_ids'][1]],row['source_ids'])):
   action={**row,'source_ids':ids,'id':fingerprint([ids,target])};features=c.features(task,state,action,reports,reviewer)
   review=copy.deepcopy(state);c.expose(task,review,ids)
   review,event=c.invoke(api,f'{index}_matched{j}',task,review,'review',81,reports,{'target_id':target,'source_ids':ids},reviewer=reviewer)
   branches.append({'features':features,'action':action,'review':event,'after':review['decision']})
  result={'task_id':task['id'],'kind':task['kind'],'initial_events':initial,'baseline':state,'features':branches[-1]['features'],'action':row,'control':event0,'review':branches[-1]['review'],'before':state['decision'],'after':branches[-1]['after'],'control_decision':control['decision'],'matched_branches':branches,'labels_available':False}

 result=json.loads(json.dumps(result));create(path,result);print('training',index+1,task['id'],bool(result.get('review',{}).get('valid'))if result.get('review')else False,flush=True);return result

def collect_training(inputs,root,provider='qwen_flash',probe=False,reuse=None,max_cny=5.,diagnostic=False):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False);tasks=read(inputs)
 if not diagnostic and any(t['metadata'].get('acquisition_partition')not in ('fit','calibration')for t in tasks):raise ValueError('training accepts only fit/calibration')
 if probe:tasks=sum(([t for t in tasks if t['kind']==kind][:2]for kind in ('qa','rec')),[])
 with locked(root):
  plan=frozen_plan(root,tasks,'train',provider,reuse=reuse,max_cny=max_cny);api=Runtime(root,plan,provider);results={}
  with ThreadPoolExecutor(max_workers=int(os.environ.get('MAS_EVAL_WORKERS','2')))as pool:
   jobs={pool.submit(training_case,i,t,root,api):i for i,t in enumerate(tasks)}
   for future in as_completed(jobs):results[jobs[future]]=future.result()
  cases=[results[i]for i in range(len(tasks))];create(root/'generation_complete.json',{'task_ids':plan['task_ids'],'hashes':[fingerprint(case)for case in cases]});labels=labels_for(tasks,root/'generation_complete.json');create(root/'labels.after_generation.json',labels)
  stats={'tasks':len(tasks),'initial_valid':sum(e['valid']for case in cases for e in case['initial_events']),'initial_events':len(tasks)*3,'review_valid':sum(bool(case['review']and case['review']['valid'])for case in cases),'matched_branches_valid':sum(bool(b['review']['valid'])for case in cases for b in case.get('matched_branches',[])),'matched_branches':sum(len(case.get('matched_branches',[]))for case in cases),'control_valid':sum(bool(case['control']and case['control']['valid'])for case in cases),'physical':audit(root,provider)}
  stats['contract_gate_passed']=stats['initial_valid']>=.8*stats['initial_events']and stats['review_valid']>=.8*len(tasks)and stats['control_valid']>=.8*len(tasks)and stats['matched_branches_valid']>=.8*max(1,stats['matched_branches'])
  create(root/'summary.json',stats);print(json.dumps(stats,ensure_ascii=False,indent=2));return stats

def fit(root,output):
 import numpy as np,joblib
 from sklearn.feature_extraction import DictVectorizer
 from sklearn.pipeline import make_pipeline
 from sklearn.linear_model import Ridge
 tasks=read(root/'inputs.json');seal=read(root/'generation_complete.json')
 if (root/'plan.json').exists():
  for name,expected in read(root/'plan.json')['source_sha256'].items():
   if sha(ROOT/name)!=expected:raise ValueError('training implementation drift; preserve old release and use new output')
 cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(tasks))]
 if seal['hashes']!=[fingerprint(case)for case in cases]or seal['task_ids']!=[t['id']for t in tasks]:raise ValueError('training seal mismatch')
 labels=labels_for(tasks,root/'generation_complete.json');rows=[];groups={}
 for t,case in zip(tasks,cases):
  partition=t['metadata']['acquisition_partition'];g=group(t)
  if groups.setdefault(g,partition)!=partition:raise ValueError('training group split overlap')
  if case['features']is None:continue
  if len(case.get('matched_branches',[]))!=3:raise ValueError('matched single/single/joint training required; use a new release directory')
  control_quality=c.quality(case['control_decision'],labels[t['id']]);branches=case['matched_branches']
  rewards=[c.quality(b['after'],labels[t['id']])-control_quality for b in branches]
  for j,b in enumerate(branches):
   rows.append({'task_id':t['id'],'partition':partition,'features':b['features'],'reward':rewards[j]if j<2 else rewards[2]-rewards[0]-rewards[1],'component':'singleton'if j<2 else'interaction','action_valid':b['review']['valid'],'control_valid':case['control']['valid']})
 models={};biases={}
 for component in ('singleton','interaction'):
  fit_rows=[r for r in rows if r['partition']=='fit'and r['component']==component];cal=[r for r in rows if r['partition']=='calibration'and r['component']==component]
  if not fit_rows or not cal:raise ValueError('nonempty fit/calibration required for both value components')
  models[component]=make_pipeline(DictVectorizer(),Ridge(alpha=10,solver='lsqr')).fit([r['features']for r in fit_rows],[r['reward']for r in fit_rows])
  biases[component]=float(np.mean([r['reward']for r in cal])-models[component].predict([r['features']for r in cal]).mean())
 create(output/'training_tasks.json',tasks);create(output/'observations.json',rows);create(output/'lineage.json',{'training_seal_sha256':sha(root/'generation_complete.json'),'group_split':groups,'features':'evidence-set-v2','query_ids_not_features':True,'interaction_target':'Q(ab)-Q(a)-Q(b)+Q(empty)'})
 path=output/'model.joblib'
 if path.exists():
  if joblib.load(path).get('training_seal_sha256')!=sha(root/'generation_complete.json'):raise ValueError('existing model lineage differs')
 else:joblib.dump({'model':models['singleton'],'bias':biases['singleton'],'interaction_model':models['interaction'],'interaction_bias':biases['interaction'],'feature_schema':'evidence-set-v2','training_seal_sha256':sha(root/'generation_complete.json')},path)
 create(output/'summary.json',{'fit_tasks':sum(t['metadata']['acquisition_partition']=='fit'for t in tasks),'calibration_tasks':sum(t['metadata']['acquisition_partition']=='calibration'for t in tasks),'positive_fit_rewards':sum(r['reward']>0 for r in fit_rows),'negative_fit_rewards':sum(r['reward']<0 for r in fit_rows),'model_sha256':sha(path),'target':'executed action quality minus matched no-extra-evidence control quality','backbone_weights_trained':False})

def value_diagnostic(inputs,root,model,provider='qwen_flash',max_cny=5.,reuse=None,limit=None):
 import joblib
 tasks=read(inputs)
 if limit is not None:tasks=tasks[:limit]
 trained=read(model.parent/'training_tasks.json')
 if {group(t)for t in trained}&{group(t)for t in tasks}:raise ValueError('value evaluation/training overlap')
 create(root/'selected_inputs.json',tasks);create(root/'value_model.json',{'model_sha256':sha(model),'feature_schema':'evidence-set-v2'})
 collect_training(root/'selected_inputs.json',root,provider,reuse=reuse,max_cny=max_cny,diagnostic=True)
 bundle=joblib.load(model);labels=labels_for(tasks,root/'generation_complete.json');records=[]
 for i,t in enumerate(tasks):
  case=read(root/'cases'/f'case_{i}.json');bs=case.get('matched_branches',[])
  if len(bs)!=3:continue
  observed=[c.quality(b['after'],labels[t['id']])-c.quality(case['control_decision'],labels[t['id']])for b in bs]
  singles=[float(bundle['model'].predict([b['features']])[0]+bundle['bias'])for b in bs[:2]]
  predictions=singles+[sum(singles)+float(bundle['interaction_model'].predict([bs[2]['features']])[0]+bundle['interaction_bias'])]
  choice=max(range(4),key=lambda j:([0.]+predictions)[j]);values=[0.]+observed
  records.append({'task_id':t['id'],'predicted':[0.]+predictions,'observed':values,'choice':choice,'selected_reward':values[choice],'uniform_reward':sum(values)/4,'observed_interaction':observed[2]-sum(observed[:2]),'predicted_interaction':predictions[2]-sum(singles),'all_branches_valid':case['control']['valid']and all(b['review']['valid']for b in bs)})
 n=len(records);summary={'scope':'heldout same-state value diagnostic, one response per branch; not repeatability proof','tasks':len(tasks),'actionable_tasks':n,'unactionable_tasks':len(tasks)-n,'selected_mean_gain':sum(r['selected_reward']for r in records)/max(1,len(tasks)),'uniform_mean_gain':sum(r['uniform_reward']for r in records)/max(1,len(tasks)),'interaction_mae':sum(abs(r['observed_interaction']-r['predicted_interaction'])for r in records)/max(1,n),'records':records,'model_sha256':sha(model),'invalid_retained':True}
 create(root/'value_summary.json',summary);return summary

def eval_case(index,task,root,api,methods,budget,bundle):
 path=root/'cases'/f'case_{index}.json'
 if path.exists():return read(path)
 baseline,reports,initial,private=initialize(task,index,api);arms={}
 for method in methods:arms[method]=execute(task,index,method,api,baseline,reports,initial,private,budget,bundle)
 case=json.loads(json.dumps({'task_id':task['id'],'kind':task['kind'],'methods':arms,'initial_events':initial,'labels_available':False}));create(path,case);print('completed',index+1,task['id'],{m:a['logical_calls']for m,a in arms.items()},flush=True);return case

def evaluate(inputs,root,methods=PRIMARY+EXTERNAL,budget=2,provider='qwen_flash',model=None,max_cny=5.,reuse=None,limit=None):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False);tasks=read(inputs)
 if limit is not None:tasks=tasks[:limit]
 bundle=None
 if model:
  import joblib
  bundle=joblib.load(model)
  if bundle.get('feature_schema')!='evidence-set-v2':raise ValueError('new estimator schema required')
  trained=read(model.parent/'training_tasks.json')
  if {group(t)for t in trained}&{group(t)for t in tasks}:raise ValueError('contribution training/test group overlap')
 valid=set(PRIMARY+EXTERNAL+ABLATIONS)
 if not methods or not set(methods)<=valid:raise ValueError('unregistered method')
 with locked(root):
  plan=frozen_plan(root,tasks,'evaluate',provider,budget,methods,model,reuse,max_cny);api=Runtime(root,plan,provider);results={}
  with ThreadPoolExecutor(max_workers=int(os.environ.get('MAS_EVAL_WORKERS','2')))as pool:
   jobs={pool.submit(eval_case,i,t,root,api,methods,budget,bundle):i for i,t in enumerate(tasks)}
   for future in as_completed(jobs):results[jobs[future]]=future.result()
  create(root/'generation_complete.json',{'task_ids':plan['task_ids'],'hashes':[fingerprint(results[i])for i in range(len(tasks))]})
  summary=score(root)
  if summary.get('budget_truncated'):raise RuntimeError('stage budget truncated; results preserved; subsequent stages stopped')
  return summary

def score(root):
 from experiments.evidence_consensus.paired_statistics_v3 import paired
 plan=read(root/'plan.json');tasks=read(root/'inputs.json');cases=[read(root/'cases'/f'case_{i}.json')for i in range(len(tasks))];seal=read(root/'generation_complete.json')
 if seal['task_ids']!=[t['id']for t in tasks]or seal['hashes']!=[fingerprint(case)for case in cases]:raise ValueError('global generation seal mismatch')
 labels=labels_for(tasks,root/'generation_complete.json');create(root/'labels.after_generation.json',labels);summary={'tasks':len(tasks),'methods':{},'scope':plan['scope'],'provider':plan['provider'],'budget':plan['additional_calls']}
 for method in plan['methods']:
  st=collections.Counter();usage=collections.Counter();transitions=[]
  for case in cases:
   arm=case['methods'][method];gold=labels[case['task_id']];after,before=c.quality(arm['decision'],gold),c.quality(arm['initial_decision'],gold);st['tasks']+=1;st['logical_calls']+=arm['logical_calls'];st['improved_tasks']+=after>before;st['harmed_tasks']+=after<before
   if case['kind']=='qa':st['qa_tasks']+=1;st['qa_correct']+=after
   else:
    st['rec_tasks']+=1;st['rec_targets']+=len(gold['targets']);st['rec_hits_at_10']+=after*len(gold['targets'])
    ranking=arm['decision'];st['rec_ndcg_sum']+=sum(1/math.log2(ranking.index(str(mid))+2)for mid in gold['targets']if str(mid)in ranking[:10]);st['candidate_targets_covered']+=sum(str(mid)in ranking for mid in gold['targets'])
   for event in arm['events']:
    st['invalid_events']+=not event['valid'];st['accepted_updates']+=event.get('accepted',False)
    for key in ('prompt_tokens','completion_tokens'):usage[key]+=event.get('usage',{}).get(key,0)
   transitions.append({'task_id':case['task_id'],'before':before,'after':after,'decision':arm['decision']})
  from experiments.paired_deliberation.review_model_profiles import PROFILES
  profile=PROFILES[plan['provider']];price=(usage['prompt_tokens']*profile['input_cny_per_million']+usage['completion_tokens']*profile['output_cny_per_million'])/1e6
  st['budget_rejected_events']=sum('budget'in e.get('error','').lower()or 'cap reached'in e.get('error','').lower()for case in cases for e in case['methods'][method]['events'])
  summary['methods'][method]={**dict(st),'qa_accuracy':st['qa_correct']/st['qa_tasks']if st['qa_tasks']else None,'rec_recall_at_10':st['rec_hits_at_10']/st['rec_targets']if st['rec_targets']else None,'solver_seconds':sum(a.get('solver_seconds',0)for case in cases for a in case['methods'][method]['allocations']),'max_recorded_deviation':max([0.]+[a.get('max_feasible_deviation',0)for case in cases for a in case['methods'][method]['allocations']]),'rec_target_averaged_ndcg_at_10':st['rec_ndcg_sum']/st['rec_targets']if st['rec_targets']else None,'candidate_coverage':st['candidate_targets_covered']/st['rec_targets']if st['rec_targets']else None,'tokens':dict(usage),'logical_known_cny':price,'transitions':transitions}
 if 'game'in plan['methods']:
  summary['game_comparisons']={ref:paired(tasks,cases,labels,reference=ref)for ref in plan['methods']if ref!='game'}
 summary['budget_truncated']=any(v.get('budget_rejected_events',0)for v in summary['methods'].values())
 create(root/'summary.json',summary);create(root/'physical_summary.json',audit(root,plan['provider']));print(json.dumps({k:v for k,v in summary.items()if k!='methods'},ensure_ascii=False,indent=2));return summary
