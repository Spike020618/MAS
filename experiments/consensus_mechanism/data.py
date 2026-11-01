"""Shared public evidence; query-conditioned cards never cross native users."""
import copy,json,re
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.prepare_disjoint_acquisition import group

DEFAULT=ROOT/'experiments/results/protocol_v3_frozen/inputs/tasks.json'
QA_RAW=ROOT/'experiments/results/mechanism/stackelberg/locked_source_game_test_v1/official_train.json'

def read(path):return json.loads(Path(path).read_text())
def labels_for(tasks,seal_path):
 from experiments.evidence_consensus.task_adapter import validate_generation_seal,QA_LABELS,REC_LABELS,ENTITIES
 validate_generation_seal(tasks,seal_path)
 plan_path=Path(seal_path).parent/'plan.json'
 if plan_path.exists():
  for name,expected in read(plan_path).get('label_sources_sha256',{}).items():
   if sha(Path(name))!=expected:raise ValueError('evaluation label source changed after acquisition freeze')
 result={};native=[t for t in tasks if not t['metadata'].get('evaluation_vault')]
 wanted={t['metadata']['native_id']:t['id']for t in native if t['kind']=='qa'}
 if wanted:
  for row in read(QA_LABELS):
   key='StrategyQA-official/'+row['qid']
   if key in wanted:result[wanted[key]]={'kind':'qa','answer':'yes'if row['answer']else'no'}
 wanted={t['metadata']['native_id']:t['id']for t in native if t['kind']=='rec'}
 if wanted:
  from experiments.paired_deliberation.native_review_calibration_data import read_fit_targets
  found=read_fit_targets(read(REC_LABELS),set(wanted),read(ENTITIES));result.update({wanted[k]:{'kind':'rec','targets':list(map(str,v))}for k,v in found.items()})
 vaults={t['metadata']['evaluation_vault']for t in tasks if t['metadata'].get('evaluation_vault')}
 for path in vaults:
  vault=read(path)
  for t in tasks:
   if t['metadata'].get('evaluation_vault')==path:result[t['id']]=vault[t['id']]
 if set(result)!={t['id']for t in tasks}:raise ValueError('missing native evaluation labels')
 return result

def historic_ids():
 qa=set();rec=set();paths=[]
 for p in (ROOT/'experiments/results').rglob('*.json'):
  if any(s in p.parts for s in ('source_snapshot','slots','logical_calls','branches')):continue
  if not(p.name in ('inputs.json','tasks.json')or p.name.endswith('.inputs.json')):continue
  if p.stat().st_size>30_000_000:continue
  try:rows=read(p)
  except (ValueError,OSError):continue
  if isinstance(rows,dict):rows=rows.get('tasks',[])
  if not isinstance(rows,list):continue
  for t in rows:
   if not isinstance(t,dict):continue
   meta=t.get('metadata',{});native=meta.get('native_id');kind=t.get('kind')
   if native and kind=='qa':qa.add(native)
   if native and kind=='rec':rec.add(str(native).rsplit(':',1)[0])
  paths.append(str(p))
 return qa,rec,paths

def expand_test(count,output,exclude_tasks):
 import numpy as np
 cached=output/'expanded_test_tasks.json'
 if cached.exists():
  saved=read(cached)
  if len(saved)!=2*count:raise ValueError('expanded selection count changed')
  return saved
 from experiments.prepare_protocol_v3 import build
 from experiments.paired_deliberation.native_review_diagnostic_packets import PrefixOnlyEvidence,prepare_safe_directory
 from experiments.evidence_consensus.task_adapter import adapt_source
 qa_seen,rec_seen,paths=historic_ids();qa_seen|={t['metadata']['native_id']for t in exclude_tasks if t['kind']=='qa'};rec_seen|={t['metadata']['native_id'].rsplit(':',1)[0]for t in exclude_tasks if t['kind']=='rec'}
 questions=[r for r in read(QA_RAW)if 'StrategyQA-official/'+r['qid']not in qa_seen];questions.sort(key=lambda r:fingerprint(['release-test-qa',r['qid']]))
 if len(questions)<count:raise ValueError(f'only {len(questions)} native-ID-unused QA available; requested {count}')
 tasks=[]
 for r in questions[:count]:
  native='StrategyQA-official/'+r['qid'];tasks.append({'id':'release:qa:'+r['qid'],'kind':'qa','prompt':r['question'],'options':['yes','no'],'candidates':[],'initial':'abstain','source_catalog':{},'metadata':{'native_id':native,'native_dataset':'StrategyQA','query_labels_used_for_selection':False,'history_scope':'known native IDs excluded; no claim of semantic or pretraining novelty'}})
 gain=ROOT/'experiments/results/native_gain_calibration_v1';rows=read(gain/'hold20.inputs.json');by={}
 for row in rows:
  conv=row['id'].rsplit(':',1)[0]
  if conv not in rec_seen:by.setdefault(conv,[]).append(row)
 convs=sorted(by,key=lambda g:fingerprint(['release-test-rec',g]))
 if len(convs)<count:raise ValueError(f'only {len(convs)} native-ID-unused recommendation groups available')
 chosen=[min(by[g],key=lambda r:fingerprint(r['id']))for g in convs[:count]];builder=PrefixOnlyEvidence(prepare_safe_directory(output/'prefix-only-serving-data'),chosen)
 ids=read(gain/'hold20.ids.json');lookup={cid:i for i,cid in enumerate(ids)};items=read(gain/'native_item_ids.json');sr=np.load(gain/'sparse/hold20.native_ranks.npy',mmap_mode='r');kr=np.load(gain/'kbrd/hold20.native_ranks.npy',mmap_mode='r')
 train_convs={str(c['conv_id'])for c in builder.train_raw}
 if train_convs&set(convs[:count]):raise ValueError('retrieval fitting group overlaps evaluation')
 for row in chosen:
  q=lookup[row['id']];union=np.flatnonzero((sr[q]<=100)|(kr[q]<=100));score=1/(60+sr[q,union].astype(float))+1/(60+kr[q,union].astype(float));order=sorted(range(len(union)),key=lambda i:(-score[i],int(items[union[i]])));ranking=[str(items[union[i]])for i in order];tid='release:rec:'+row['id'];serving={'id':row['id'],'text':row['text'],'candidates':[{'movie_id':mid,'title':builder.names[int(mid)]}for mid in ranking]};evidence=builder.augment(serving,history_limit=3,kg_limit=3)
  catalog={sid:adapt_source(sid,s,tid)for sid,s in evidence['sources'].items()if s.get('kind')in ('knowledge_graph','training_aggregate','historical_prediction_basis')}
  tasks.append({'id':tid,'kind':'rec','prompt':evidence['text'],'options':[],'candidates':serving['candidates'],'initial':ranking,'source_catalog':catalog,'metadata':{'native_id':row['id'],'native_dataset':'CRSLab-ReDial','query_labels_used_for_selection':False,'history_scope':'fold80-fitted Sparse/KBRD, fixed RRF60; visible historical groups excluded; not globally pristine'}})
 prepared,_,diagnostics=build(tasks,corpus_path=ROOT/'experiments/results/redesign_v1/retrieval_env/wiki_fts.sqlite')
 for t in prepared:t['metadata']['acquisition_partition']='diagnostic'
 create(output/'expanded_test_tasks.json',prepared);create(output/'expanded_test_diagnostics.json',diagnostics);create(output/'history_exclusion.json',{'qa_native_ids_excluded':sorted(qa_seen),'rec_groups_excluded':sorted(rec_seen),'metadata_files':paths,'scope':'observable native identity history, not semantic/pretraining novelty','source_qa_sha256':sha(QA_RAW),'source_rec_sha256':sha(gain/'hold20.inputs.json')})
 return prepared

def prepare(output,inputs=DEFAULT,test_per_kind=None,expand_count=0):
 tasks=read(inputs)
 train=[t for t in tasks if t['metadata']['acquisition_partition']in ('fit','calibration')];test=[t for t in tasks if t['metadata']['acquisition_partition']=='diagnostic']
 if expand_count:test=expand_test(expand_count,output,tasks)
 if test_per_kind is not None:
  selected=[]
  for kind in ('qa','rec'):
   bucket=sorted([t for t in test if t['kind']==kind],key=lambda t:fingerprint(['release-evaluation',t['id']]))
   if len(bucket)<test_per_kind:raise ValueError('insufficient tasks; do not silently reduce requested count')
   selected+=bucket[:test_per_kind]
  test=selected
 if not train or not test:raise ValueError('both training and test tasks required')
 if {group(t)for t in train}&{group(t)for t in test}:raise ValueError('original task group overlap')
 for name,rows in [('train',train),('test',test)]:create(output/f'{name}.json',rows)
 create(output/'data_manifest.json',{'input_sha256':sha(inputs),'train_tasks':len(train),'test_tasks':len(test),'test_kind_counts':{kind:sum(t['kind']==kind for t in test)for kind in ('qa','rec')},'training_scope':'only original fit/calibration groups','test_scope':'new native-ID-excluded tasks'if expand_count else'reused development tasks, not confirmation','training_groups':[group(t)for t in train],'test_groups':[group(t)for t in test],'query_labels_used_for_selection':False,'shared_public_background':True,'model_api_calls':0})
 return train,test

def auxiliary(output,per_kind=32):
 """Lock secondary analyses before outcomes; do not expand a factorial matrix."""
 tasks=read(output/'test.json');selected=[]
 for kind in ('qa','rec'):
  bucket=sorted([t for t in tasks if t['kind']==kind],key=lambda t:fingerprint(['secondary-analysis-v2',t['id']]))
  selected+=bucket[:per_kind]
 create(output/'auxiliary.json',selected);create(output/'auxiliary_manifest.json',{'per_kind_cap':per_kind,'tasks':len(selected),'kind_counts':{kind:sum(t['kind']==kind for t in selected)for kind in ('qa','rec')},'full_test_hash':fingerprint(tasks),'selection':'pre-outcome task hash, no quality or label filter','scope':'secondary diagnostic/ablation/budget/stress/transfer; main uses full test'})
 return output/'auxiliary.json'
