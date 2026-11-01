"""Pre-acquisition structural/provenance checks; never labels as selection features."""
import collections,json,math
from pathlib import Path
from experiments.consensus_recommendation.common import create,fingerprint,sha
from experiments.prepare_disjoint_acquisition import group
from experiments.evidence_consensus.schema import scope_allows,candidate_ids
from .data import read

def audit_inputs(root):
 root=Path(root);train=read(root/'data/train.json');test=read(root/'data/test.json');sets={'training':train,'primary':test}
 if(root/'benchmark_data/tasks.json').exists():sets['hotpot']=read(root/'benchmark_data/tasks.json')
 training_groups={group(t)for t in train};questions={fingerprint(t['prompt'].strip().lower())for t in train};report={}
 for name,tasks in sets.items():
  ids=[t['id']for t in tasks]
  if not tasks or len(ids)!=len(set(ids)):raise ValueError('empty or repeated task identity: '+name)
  if name!='training':
   if training_groups&{group(t)for t in tasks}:raise ValueError('training/evaluation group overlap')
   if questions&{fingerprint(t['prompt'].strip().lower())for t in tasks}:raise ValueError('exact training/evaluation question overlap')
  sizes=[];source_types=collections.Counter()
  for t in tasks:
   if any(k in t for k in ('answer','gold','label','targets','supporting_facts')):raise ValueError('evaluation gold in runtime task')
   catalog=t['source_catalog'];initial=t['metadata']['protocol_v3']['initial_source_ids']
   expected_distinct=min(3,len(catalog))if name=='hotpot'else 3
   if len(initial)!=3 or len(set(initial))!=expected_distinct or not set(initial)<=set(catalog):raise ValueError('three distinct initial source IDs required')
   unique={fingerprint(s['text'])for s in catalog.values()};sizes.append(len(unique))
   if len(unique)<(2 if name=='hotpot'else 5):raise ValueError('at least five distinct source texts required')
   for sid,s in catalog.items():
    if not s.get('text','').strip()or not scope_allows(s,t['id']):raise ValueError('empty or out-of-scope source')
    if s.get('current_user_fact_authority')and s.get('kind')in ('historical_prediction_basis','training_aggregate','model_forecast'):raise ValueError('history/forecast falsely authorized as current-user fact')
    source_types[s.get('kind','unknown')]+=1
   if t['kind']=='rec'and len(candidate_ids(t))<10:raise ValueError('insufficient native candidate pool')
  report[name]={'tasks':len(tasks),'groups':len({group(t)for t in tasks}),'kinds':dict(collections.Counter(t['kind']for t in tasks)),'unique_sources_min':min(sizes),'unique_sources_max':max(sizes),'source_types':dict(source_types),'runtime_task_hash':fingerprint(tasks)}
 manifests={}
 for rel in ('data/data_manifest.json','benchmark_data/manifest.json','semantic_data/manifest.json'):
  path=root/rel
  if path.exists():manifests[rel]={'sha256':sha(path),'metadata':read(path)}
 result={'sets':report,'manifests':manifests,'group_and_exact_question_checks_passed':True,'selection_uses_model_outcomes':False,'scope':'structural/provenance quality; lexical retrieval does not certify semantic coverage','sample_planning_assumption':{'paired_discordance':.2,'alpha_two_sided':.05,'power':.8,'approx_detectable_absolute_delta_by_n':{str(n):math.sqrt((1.96+.84)**2*.2/n)for n in (200,458,1000)},'interpretation':'planning approximation, not observed variance or a guarantee; recommendation needs conversation-cluster analysis'},'model_api_calls':0}
 create(root/'input_quality.json',result);print(json.dumps({'input_quality_passed':True,'tasks':{k:v['tasks']for k,v in report.items()},'model_api_calls':0},ensure_ascii=False));return result
