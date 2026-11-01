"""Explicit experiment entry points; defaults never dispatch model requests."""
import argparse,json,math
from pathlib import Path
from . import data
from .engine import PRIMARY,EXTERNAL,ABLATIONS
from . import runner
from experiments.consensus_recommendation.common import ROOT,create

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','paper_prepare','benchmark_prepare','benchmark','rank_baselines','check','probe','train','value','main','ablations','budget','stress','transfer','semantic_prepare','semantic','report','all']);p.add_argument('--root',type=Path,default=ROOT/'experiments/results/consensus_mechanism_set_v2r1');p.add_argument('--inputs',type=Path,default=data.DEFAULT);p.add_argument('--provider',default='qwen_flash',choices=['qwen_flash','glm_flash','qwen_plus','deepseek']);p.add_argument('--transfer-provider',default='glm_flash',choices=['qwen_flash','glm_flash','qwen_plus','deepseek']);p.add_argument('--run',action='store_true');p.add_argument('--fetch-data',action='store_true');p.add_argument('--test-per-kind',type=int);p.add_argument('--expand-test',type=int,default=0);p.add_argument('--semantic-count',type=int,default=180);p.add_argument('--semantic-source',type=Path);p.add_argument('--budget',type=int,default=2,choices=[0,1,2,3]);p.add_argument('--max-cny',type=float,default=5.);p.add_argument('--limit',type=int);p.add_argument('--reuse',type=Path,help='replay exact 200 receipts from another run root; denied and unknown requests are retried or recorded per discipline');a=p.parse_args();root=a.root
 if a.semantic_count<=0 or a.expand_test<0 or(a.test_per_kind is not None and a.test_per_kind<=0)or(a.limit is not None and a.limit<=0):p.error('counts and limit must be positive; expand-test may be zero')
 if not math.isfinite(a.max_cny)or a.max_cny<=0:p.error('positive finite max-cny required')
 def prepare():
  train,test=data.prepare(root/'data',a.inputs,a.test_per_kind,a.expand_test);print(json.dumps({'training_tasks':len(train),'test_tasks':len(test),'model_api_calls':0},indent=2));return train,test
 def execution(stage):
  if stage=='paper_prepare':
   train,test=data.prepare(root/'data',a.inputs,expand_count=200);execution('semantic_prepare');execution('benchmark_prepare');stress_inputs(root,a.inputs,data.auxiliary(root/'data'))
   from .quality import audit_inputs
   from .rank_references import prepare as rank_prepare
   rank_prepare(root)
   return audit_inputs(root)
  if stage=='benchmark_prepare':
   from .benchmark import prepare as benchmark_prepare
   tasks=benchmark_prepare(root/'benchmark_data',fetch=a.fetch_data);print(json.dumps({'benchmark_tasks':len(tasks),'model_api_calls':0}));return
  if stage=='prepare':return prepare()
  if stage=='semantic_prepare':
   from .semantic import prepare as semantic_prepare
   tasks=semantic_prepare(root/'semantic_data',a.semantic_count,a.semantic_source,a.fetch_data);print(json.dumps({'semantic_tasks':len(tasks),'model_api_calls':0}));return
  if stage=='check':
   if (root/'data/train.json').exists()and(root/'data/test.json').exists():train,test=data.read(root/'data/train.json'),data.read(root/'data/test.json')
   else:train,test=prepare()
   execution('semantic_prepare');paths=stress_inputs(root,a.inputs,data.auxiliary(root/'data'))
   from .quality import audit_inputs
   audit_inputs(root)
   print(json.dumps({'training_tasks':len(train),'test_tasks':len(test),'stress_variants':list(paths),'model_api_calls':0,'status':'inputs_ready'},ensure_ascii=False));return
  if stage=='report':
   from .report import report
   return report(root)
  train_path=root/'data/train.json';test_path=root/'data/test.json';model=root/'model/model.joblib'
  if not train_path.exists():prepare()
  aux_path=data.auxiliary(root/'data')
  if not a.run:
   print(json.dumps({'stage':stage,'model_api_calls':0,'next':'add --run to execute this stage','root':str(root)},indent=2));return
  if stage=='probe':return runner.collect_training(train_path,root/'probe',a.provider,probe=True,max_cny=a.max_cny)
  if stage=='train':
   probe=execution('probe')
   if not probe['contract_gate_passed']:raise RuntimeError('interface probe did not pass; responses saved, full collection stopped')
   stats=runner.collect_training(train_path,root/'train',a.provider,reuse=root/'probe',max_cny=a.max_cny)
   if not stats['contract_gate_passed']:raise RuntimeError('full training acquisition did not pass contract gate; responses retained')
   runner.fit(root/'train',root/'model');return stats
  if stage=='value':return runner.value_diagnostic(aux_path,root/'value',model,a.provider,a.max_cny,reuse=root/'main',limit=a.limit)
  if stage=='rank_baselines':
   from .rank_references import score as rank_score
   return rank_score(root)
  if stage=='benchmark':return runner.evaluate(root/'benchmark_data/tasks.json',root/'benchmark',PRIMARY+EXTERNAL,a.budget,a.provider,model,a.max_cny,reuse=a.reuse,limit=a.limit)
  if stage=='main':return runner.evaluate(test_path,root/'main',PRIMARY+EXTERNAL,a.budget,a.provider,model,a.max_cny,reuse=a.reuse,limit=a.limit)
  if stage=='ablations':return runner.evaluate(aux_path,root/'ablations',('game',)+ABLATIONS,a.budget,a.provider,model,a.max_cny,reuse=root/'main',limit=a.limit)
  if stage=='budget':
   for k in (0,1,2,3):runner.evaluate(aux_path,root/'budget'/str(k),('fixed','greedy','game'),k,a.provider,model,a.max_cny,reuse=root/'main',limit=a.limit)
   return
  if stage=='stress':
   for environment,dest in stress_inputs(root,a.inputs,aux_path).items():
    runner.evaluate(dest,root/'stress'/environment,('fixed','greedy','game'),a.budget,a.provider,model,a.max_cny,reuse=root/'main',limit=a.limit)
   return
  if stage=='transfer':return runner.evaluate(aux_path,root/'transfer',('single_strong','independent','greedy','central','game'),a.budget,a.transfer_provider,model,a.max_cny,limit=a.limit)
  if stage=='semantic':
   from .semantic import evaluate as semantic_evaluate
   path=root/'semantic_data/tasks.json'
   if not path.exists():execution('semantic_prepare')
   return semantic_evaluate(path,root/'semantic',a.provider,a.limit,a.max_cny)
  if stage=='all':
   # Resolve public-data prerequisites before the first paid training request.
   execution('semantic_prepare')
   from dotenv import load_dotenv
   from experiments.paired_deliberation.transport_d2 import ENDPOINTS
   import os
   load_dotenv(ROOT/'.env.local',override=False)
   for name in {a.provider,a.transfer_provider}:
    key_name=ENDPOINTS[name][1]
    if key_name and not os.environ.get(key_name):raise RuntimeError('missing provider credentials: '+name)
   stress_inputs(root,a.inputs,aux_path)
   from .quality import audit_inputs
   audit_inputs(root)
   from .rank_references import prepare as rank_prepare
   rank_prepare(root)
   for part in ('train','main','rank_baselines','value','ablations','budget','stress','transfer','semantic'):execution(part)
   if (root/'benchmark_data/tasks.json').exists():execution('benchmark')
   execution('report')
 execution(a.stage)
def stress_inputs(root,original_path,test_path):
 import copy
 from experiments.consensus_recommendation.common import fingerprint
 from experiments.prepare_protocol_v3_stress import variants
 tasks=data.read(test_path);originals=data.read(original_path);lookup={t['id']:t for t in originals}
 # Extra static catalog cards may come from other tasks, but conditioned scores may not.
 static={kind:[card for t in originals+tasks if t['kind']==kind for card in t['source_catalog'].values()if card.get('kind')in ('knowledge_graph','cached_web_reference')]for kind in ('qa','rec')}
 extended=[]
 for task in tasks:
  original=copy.deepcopy(lookup.get(task['id'],task));present={fingerprint(s['text'][:2500])for s in task['source_catalog'].values()}
  for card in static[task['kind']]:
   if fingerprint(card['text'][:2500])not in present:
    original['source_catalog'].setdefault('stress-pool:'+fingerprint(card['text']),card);break
  extended.append(original)
 paths={}
 for name,rows in variants(tasks,extended).items():
  path=root/'stress_auxiliary_inputs'/name/'tasks.json';create(path,rows);paths[name]=path
 return paths
if __name__=='__main__':main()

