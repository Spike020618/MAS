"""Explicit manual, source-connected transfer protocol. Never calls model APIs."""
import argparse,json,hashlib
from pathlib import Path
from experiments.consensus_recommendation.common import create,fingerprint,sha
from experiments.fit_e2_action_calibration import quality

def read(p):return json.loads(Path(p).read_text())
def components(tasks):
 parent=list(range(len(tasks)));owners={}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i,t in enumerate(tasks):
  native=t['metadata']['native_id'];group=t['kind']+':'+(native.rsplit(':',1)[0]if t['kind']=='rec'else native)
  keys=['group:'+group]+['content:'+hashlib.sha256(' '.join(s['text'].split()).encode()).hexdigest()for s in t['source_catalog'].values()]
  for key in keys:
   if key in owners:parent[find(i)]=find(owners[key])
   owners[key]=i
 out={}
 for i,t in enumerate(tasks):out.setdefault(find(i),[]).append(t['id'])
 return sorted(out.values(),key=lambda ids:fingerprint(sorted(ids)))

def prepare(run,out,held_kind=None):
 tasks=read(run/'inputs.json');cc=components(tasks);lookup={t['id']:t for t in tasks};split={}
 if held_kind:
  training=[]
  for ids in cc:
   kinds={lookup[i]['kind']for i in ids}
   if held_kind in kinds:
    if len(kinds)>1:raise ValueError('shared source bridges held-out task kind; cannot claim source isolation')
    for i in ids:split[i]='diagnostic'
   else:training.append(ids)
  if len(training)<2:raise ValueError('insufficient source-connected training components')
  cut=max(1,min(len(training)-1,int(.8*len(training))))
  for j,ids in enumerate(training):
   for i in ids:split[i]='fit'if j<cut else 'calibration'
 else:
  if len(cc)<3:raise ValueError('fewer than three source-connected components; acquire disjoint sources, do not fall back to random task split')
  a=max(1,min(len(cc)-2,int(.6*len(cc))));b=max(a+1,min(len(cc)-1,int(.8*len(cc))))
  for j,ids in enumerate(cc):
   for i in ids:split[i]='fit'if j<a else 'calibration'if j<b else 'diagnostic'
 if set(split.values())!={'fit','calibration','diagnostic'}:raise ValueError('empty partition')
 commit={'inputs_sha256':sha(run/'inputs.json'),'source_components':cc,'task_split':split,'held_kind':held_kind,'features':'pre-action structural counts only; no task/source identities, URLs, title text or content','scope':'source-connected development transfer, not untouched confirmation'}
 create(out/'split_commitment.json',commit);return tasks,commit

def features(t,c):
 e=c['event'];s=t['source_catalog'][e['source_action']['source_id']]
 return {'source_chars':float(len(s['text'])),'visible_count':float(len(e['source_action']['before_visible_ids'])),'version':float(e['expected_state_version']),'candidate_count':float(len(t.get('candidates',[]))),'baseline_abstain':float(c['before_decision']=='abstain')}

def fit(run,out,held_kind=None):
 import joblib,numpy as np
 from sklearn.feature_extraction import DictVectorizer
 from sklearn.pipeline import make_pipeline
 from sklearn.linear_model import Ridge
 tasks,commit=prepare(run,out,held_kind);seal=read(run/'generation_complete.json');cases=[read(run/'cases'/f'case_{i}.json')for i in range(len(tasks))]
 if seal['task_ids']!=[t['id']for t in tasks]or seal['hashes']!=[fingerprint(c)for c in cases]:raise ValueError('generation seal mismatch')
 plan=read(run/'plan.json')
 for name,h in plan['source_sha256'].items():
  if sha(run/'source_snapshot'/name)!=h:raise ValueError('source snapshot mismatch')
 labels=read(run/'labels.after_generation.json');rows=[];seen=set()
 for t,c in zip(tasks,cases):
  if c['task_id']!=t['id']:raise ValueError('case identity mismatch')
  key=c['event']['request_hash']
  if key in seen:continue
  seen.add(key);base=c.get('baseline_event');eligible=t['kind']!='qa'or bool(base and base['valid']and c['before_decision']!='abstain')
  reward=quality(c['after_decision'],labels[t['id']])-quality(c['before_decision'],labels[t['id']])
  rows.append({'task_id':t['id'],'split':commit['task_split'][t['id']],'eligible':eligible,'valid':c['event']['valid'],'features':features(t,c),'reward':reward,'damage':int(reward<0)})
 create(out/'observations.json',rows);parts={k:[r for r in rows if r['split']==k and r['eligible']]for k in ['fit','calibration','diagnostic']}
 if any(not p for p in parts.values()):raise ValueError('empty eligible partition; no fallback')
 def train(target):return make_pipeline(DictVectorizer(),Ridge(alpha=10,solver='lsqr')).fit([r['features']for r in parts['fit']],[r[target]for r in parts['fit']])
 gain=train('reward');damage=train('damage');cal=parts['calibration'];bias=float(np.mean([r['reward']for r in cal])-gain.predict([r['features']for r in cal]).mean())
 path=out/'model.joblib'
 if path.exists():raise ValueError('model exists; use new output')
 joblib.dump({'gain':gain,'damage':damage,'gain_bias':bias,'feature_schema':'identity-free-structural-v1','split_sha256':sha(out/'split_commitment.json')},path)
 pred=[{'task_id':r['task_id'],'observed':r['reward'],'predicted':float(gain.predict([r['features']])[0]+bias)}for r in parts['diagnostic']]
 create(out/'diagnostic_predictions.json',pred);create(out/'summary.json',{'model_sha256':sha(path),'counts':{k:len(v)for k,v in parts.items()},'gain_mse':float(np.mean([(p['observed']-p['predicted'])**2 for p in pred])),'scope':commit['scope']})

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--held-kind',choices=['qa','rec']);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-only',action='store_true');g.add_argument('--fit',action='store_true');a=p.parse_args()
 if a.fit:fit(a.run_root,a.output,a.held_kind)
 else:prepare(a.run_root,a.output,a.held_kind)
