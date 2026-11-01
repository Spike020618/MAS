"""Public human-labelled evidence relations from the original Climate-FEVER release.

This is a supplied-evidence relation subtask, not the original claim-level FEVER
leaderboard. One labelled passage per native claim; groups never cross splits.
"""
import collections,copy,json
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from . import core as c
from .data import read,labels_for
from .runtime import Runtime
from .runner import locked,frozen_plan
from experiments.paired_deliberation.native_optimization_runtime import audit
REVISION='03de61617b10a5c1935f8e08bb0e8ac1ee775356'
URL=f'https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/{REVISION}/dataset/climate-fever.jsonl'


def prepare(output,count=180,source=None,fetch=False):
 source=Path(source)if source else ROOT/'experiments/data/public/climate_fever/climate-fever.jsonl'
 if not source.exists():
  if not fetch:raise ValueError('public semantic data missing; use --fetch-data to download pinned original release')
  import requests
  response=requests.get(URL,timeout=(15,60));response.raise_for_status();source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(response.content)
 rows=[json.loads(line)for line in source.read_text().splitlines()if line.strip()];rows.sort(key=lambda r:fingerprint(['semantic-eval',str(r['claim_id'])]))
 if count>len(rows):raise ValueError('requested more native groups than available')
 tasks=[];vault={};label_map={'SUPPORTS':'supported','REFUTES':'refuted','NOT_ENOUGH_INFO':'not_enough_info'}
 for r in rows[:count]:
  evidence=min(r['evidences'],key=lambda e:fingerprint([str(r['claim_id']),e['evidence_id']]));tid='semantic:'+str(r['claim_id']);sid='evidence:'+fingerprint([tid,evidence['evidence_id']])[:18];rawlabel=evidence['evidence_label'];label=label_map[rawlabel]if isinstance(rawlabel,str)else ['supported','refuted','not_enough_info'][rawlabel]
  source_card={'source_id':sid,'kind':'public_annotated_evidence','title':evidence.get('article',''),'text':evidence['evidence'],'scope':{'task_id':tid,'subject':'reference'},'allowed_uses':['prediction_basis']}
  tasks.append({'id':tid,'kind':'qa','prompt':'Determine whether the following claim is supported, refuted, or not established by the supplied passage. Claim: '+r['claim'],'options':['supported','refuted','not_enough_info'],'candidates':[],'initial':['supported','refuted','not_enough_info'][int(fingerprint(tid)[:8],16)%3],'source_catalog':{sid:source_card},'metadata':{'native_id':'ClimateFEVER/'+str(r['claim_id']),'family':'evidence_relation','evidence_only':True,'uncertainty_answers':['not_enough_info'],'given_claim':r['claim'],'target_claim_id':'given_claim','evaluation_vault':str((output/'evaluation_vault.json').resolve()),'protocol_v3':{'initial_source_ids':[sid,sid,sid]},'subtask_scope':'one human-labelled claim-evidence relation, not claim-level FEVER score'}})
  vault[tid]={'kind':'qa','answer':label}
 create(output/'tasks.json',tasks);create(output/'evaluation_vault.json',vault);create(output/'manifest.json',{'source_url':URL,'revision':REVISION,'source_sha256':sha(source),'available_claims':len(rows),'evaluation_claims':len(tasks),'selection':'native claim hash then evidence hash, not label-selected','runtime_gold_hidden':True,'labels_read_by_preparation_for_vault_only':True,'scope':'human-labelled passage relation subtask; not user-satisfaction or original FEVER leaderboard','label_counts_for_reporting_only':dict(collections.Counter(v['answer']for v in vault.values()))})
 return tasks

def evaluate(inputs,root,provider='qwen_flash',limit=None,max_cny=5.):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False);tasks=read(inputs)
 if limit:tasks=tasks[:limit]
 with locked(root):
  plan=frozen_plan(root,tasks,'semantic',provider,methods=('plain_judge','ordinary_revision','grounded_update'),max_cny=max_cny);api=Runtime(root,plan,provider);cases=[]
  for index,task in enumerate(tasks):
   path=root/'cases'/f'case_{index}.json'
   if path.exists():cases.append(read(path));continue
   initial=c.expose(task,c.fresh(task),list(task['source_catalog']));initial['claims']['given_claim']={'id':'given_claim','text':task['metadata']['given_claim'],'owner':0,'status':'unverified','active':True};methods={}
   for method,phase in [('plain_judge','initial'),('ordinary_revision','direct'),('grounded_update','review')]:
    state,event=c.invoke(api,f'{index}_{method}',task,copy.deepcopy(initial),phase,index,action={'target_id':'given_claim','source_ids':[]})
    if phase=='initial'and event['valid']:state['decision']=c.full_decision(task,event['report']['decision'])
    methods[method]={'decision':state['decision'],'initial_decision':initial['decision'],'events':[event],'logical_calls':1}
   case={'task_id':task['id'],'kind':'qa','methods':methods,'labels_available':False};create(path,case);cases.append(case);print('semantic',index+1,flush=True)
  create(root/'generation_complete.json',{'task_ids':plan['task_ids'],'hashes':[fingerprint(c)for c in cases]});gold=labels_for(tasks,root/'generation_complete.json');create(root/'labels.after_generation.json',gold);summary={'scope':'supplied-evidence 3-way relation; random provisional labels are a controlled starting state','tasks':len(tasks),'methods':{}}
  for method in plan['methods']:
   truth=[gold[t['id']]['answer']for t in tasks];pred=[case['methods'][method]['decision']for case in cases];classes=['supported','refuted','not_enough_info'];conf={a:{b:sum(y==a and p==b for y,p in zip(truth,pred))for b in classes+['abstain']}for a in classes};f1=[]
   for label in classes:
    tp=sum(y==p==label for y,p in zip(truth,pred));fp=sum(p==label and y!=label for y,p in zip(truth,pred));fn=sum(y==label and p!=label for y,p in zip(truth,pred));f1.append(2*tp/max(1,2*tp+fp+fn))
   summary['methods'][method]={'correct':sum(y==p for y,p in zip(truth,pred)),'macro_f1':sum(f1)/3,'confusion':conf,'invalid':sum(not case['methods'][method]['events'][0]['valid']for case in cases),'corrected_provisional':sum(gold[t['id']]['answer']!=t['initial']and pred[i]==gold[t['id']]['answer']for i,t in enumerate(tasks)),'damaged_provisional':sum(gold[t['id']]['answer']==t['initial']and pred[i]!=gold[t['id']]['answer']for i,t in enumerate(tasks))}
  create(root/'summary.json',summary);create(root/'physical_summary.json',audit(root,provider));return summary
