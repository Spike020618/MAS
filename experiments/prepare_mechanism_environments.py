"""Label-free structural information environments. No network or gold labels.

Missing is deterministic removal, not guaranteed loss of decisive evidence.
Distractors are real other-task materials, not certified irrelevant/false facts.
Redundancy is repeated exposure of one source, never independent corroboration.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from experiments.consensus_recommendation.common import create,sha,fingerprint

ENVIRONMENTS=('baseline','missing','redundant','distractor')
def original_group(t):
 n=str(t.get('metadata',{}).get('native_id',t['id']))
 return t['kind']+':'+(n.rsplit(':',1)[0]if t['kind']=='rec'else n)
def build(tasks,environment):
 if environment not in ENVIRONMENTS:raise ValueError('unknown environment')
 bounded=copy.deepcopy(tasks)
 for t in bounded:
  for source in t['source_catalog'].values():
   full=source['text'];source['canonical_original_content_hash']=fingerprint(full);source['full_text_sha256']=hashlib.sha256(full.encode()).hexdigest();source['text']=full[:1000];source['environment_clipped']=len(full)>1000;source['environment_visible_text_sha256']=hashlib.sha256(source['text'].encode()).hexdigest();source['independent_evidence_units']=1
 rows=[]
 for i,t in enumerate(bounded):
  x=copy.deepcopy(t);catalog=x['source_catalog'];ids=sorted(catalog,key=lambda s:hashlib.sha256((t['id']+s).encode()).hexdigest())
  if environment=='missing':
   for sid in ids[len(ids)//2:]:del catalog[sid]
  elif environment=='redundant':
   # Exposure duplication in one card preserves original source identity.
   for sid in ids:
    s=catalog[sid];s['original_content_hash']=fingerprint(s['text']);s['text']=s['text']+'\n\n'+s['text'];s['intervention_text_sha256']=hashlib.sha256(s['text'].encode()).hexdigest();s['repeated_exposure']=2;s['independent_evidence_units']=1
  elif environment=='distractor'and len(tasks)>1:
   donor=bounded[(i+1)%len(bounded)]
   for j,(sid,s)in enumerate(sorted(donor['source_catalog'].items())[:2]):
    k='distractor:'+str(j)+':'+sid;card=copy.deepcopy(s);card['source_id']=k;card['origin_task_id']=donor['id'];card['scope']={'task_id':t['id'],'subject':'other_context','role':'reference'};card['current_user_fact_authority']=False;card['allowed_uses']=['prediction_basis'];card['synthetic_environment_role']='cross_task_material_not_known_false';catalog[k]=card
  x.setdefault('metadata',{}).update(environment=environment,original_group=original_group(t),environment_intervention='structural only; no decisive-evidence or semantic-truth guarantee')
  rows.append(x)
 return rows

def prepare(inputs,output):
 tasks=json.loads(Path(inputs).read_text());output=Path(output)
 if isinstance(tasks,dict):tasks=tasks['tasks']
 groups=sorted({original_group(t)for t in tasks},key=lambda g:hashlib.sha256(('mechanism-environment-split-v1:'+g).encode()).hexdigest())
 split={g:'fit'if i<int(.6*len(groups))else 'calibration'if i<int(.8*len(groups))else 'diagnostic'for i,g in enumerate(groups)}
 hashes={}
 for env in ENVIRONMENTS:
  rows=build(tasks,env)
  for x in rows:x['metadata']['group_split']=split[original_group(x)]
  p=output/env/'tasks.json';create(p,rows);hashes[env]=sha(p)
 manifest={'version':'structural-information-environments-v1','input_sha256':sha(inputs),'environments':list(ENVIRONMENTS),'task_counts':len(tasks),'base_source_char_limit':1000,'redundant_exposure_char_limit':2002,'runner_char_limit_expected':2500,'group_split':split,'inputs_sha256':hashes,'permissions':'identical environment input supplied to all five methods','gold_labels_read':False,'model_api_calls':0,'scope':'structural stress tests; baseline not guaranteed sufficient, distractors not guaranteed irrelevant, no fabricated contradiction truth'}
 create(output/'manifest.json',manifest);return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.inputs,a.output),indent=2))
