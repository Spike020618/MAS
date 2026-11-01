"""Development-only grouped crossfit of observed action gains; no API calls.
Features are available BEFORE executing an action. Labels are read only from
sealed runs. Observational rewards do not identify causal evidence value.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha

FEATURES=('is_rec','source_relevance','source_characters','state_version','visible_before')

def features(kind,proposal,version,visible):
 return [float(kind=='rec'),float(proposal.get('score',0)),float(proposal.get('source_characters',0)),float(version),float(visible)]

def quality(decision,label):
 if label['kind']=='qa':return float(decision==label['answer'])
 return sum(str(t)in list(map(str,decision))[:10]for t in label['targets'])

def extract(root):
 root=Path(root);plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'generation_complete.json').read_text())
 cases=[json.loads((root/'cases'/('case_'+str(i)+'.json')).read_text())for i in range(plan['count'])]
 if seal['hashes']!=[fingerprint(c)for c in cases]:raise ValueError('generation identity mismatch')
 labels=json.loads((root/'labels.after_generation.json').read_text());rows=[];seen=set()
 for c in cases:
  for method,arm in c['methods'].items():
   for e in arm['events']:
    if 'source_action'not in e:continue
    # Same receipt at same decision is one observation, regardless of method.
    key=(e['request_hash'],fingerprint(e.get('before_decision')))
    if key in seen:continue
    seen.add(key);before=e.get('before_decision');after=e.get('after_decision')
    observed=before is not None and after is not None
    gain=quality(after,labels[c['task_id']])-quality(before,labels[c['task_id']])if observed else None
    rows.append({'task_id':c['task_id'],'group':c['task_id'].rsplit(':',1)[0]if c['kind']=='rec'else c['task_id'],'request_hash':e['request_hash'],'features':features(c['kind'],e['proposal'],e['expected_state_version'],len(e['source_action']['before_visible_ids'])),'reward':gain,'damage':float(gain<0)if observed else None,'valid':e['valid'],'accepted':bool(e.get('patch_accepted')),'outcome_observed':observed,'method_first_seen':method})
 return rows

class ActionCalibrator:
 def __init__(self,path):self.bundle=joblib.load(path)
 def predict(self,kind,proposal,version=0,visible=0):
  x=np.asarray([features(kind,proposal,version,visible)])
  return {'estimated_gain':float(self.bundle['gain'].predict(x)[0]),'estimated_damage_rate':float(np.clip(self.bundle['damage'].predict(x)[0],0,1)),'scope':'pilot observational grouped calibration, not certified VOI'}

def run(root,output):
 root=Path(root);output=Path(output);rows=extract(root);create(output/'actions.all.json',rows)
 usable=[r for r in rows if r['outcome_observed']]
 groups=sorted({r['group']for r in rows});fold={g:int(hashlib.sha256(('action-pilot-v1:'+g).encode()).hexdigest(),16)%3 for g in groups}
 create(output/'split_commitment.json',{'group_fold':fold,'unit':'original QA question or recommendation conversation','features':FEATURES,'scope':'all E1 subjects are reused development; no pristine test claim'})
 x=np.array([r['features']for r in usable]);y=np.array([r['reward']for r in usable]);damage=np.array([r['damage']for r in usable]);oof=[]
 def model():return make_pipeline(StandardScaler(),Ridge(alpha=10.))
 for f in range(3):
  train=np.array([fold[r['group']]!=f for r in usable]);test=~train
  if not train.any()or not test.any():continue
  m=model().fit(x[train],y[train]);pred=m.predict(x[test])
  oof += [{'task_id':r['task_id'],'fold':f,'observed_reward':r['reward'],'predicted_reward':float(p)}for r,p in zip([r for r,t in zip(usable,test)if t],pred)]
 bundle={'gain':model().fit(x,y),'damage':model().fit(x,damage),'features':FEATURES}
 output.mkdir(parents=True,exist_ok=True);path=output/'pilot_model.joblib'
 if path.exists():raise ValueError('model exists: use a new output version')
 joblib.dump(bundle,path);create(output/'oof.json',oof)
 summary={'rows':len(rows),'observed_rows':len(usable),'failed_or_no_transition_rows_retained':len(rows)-len(usable),'groups':len(groups),'source_seal_sha256':sha(root/'generation_complete.json'),'model_sha256':sha(path),'scope':'small observational development pilot; failed transitions retained but not falsely labeled zero reward; no causal or deployed efficacy claim','oof_reward_mse':float(np.mean([(r['observed_reward']-r['predicted_reward'])**2 for r in oof]))}
 create(output/'summary.json',summary);print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run_root,a.output)
