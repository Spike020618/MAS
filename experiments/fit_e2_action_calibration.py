"""Sealed E2 contribution learning: precommitted question/conversation splits.
No API. New observations remain training development, never pristine tests.
"""
import argparse,json,hashlib,re,math
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.evidence_consensus.action_calibration import quality
RUN=ROOT/'experiments/results/e2_action_acquisition_160_live_v1'
OUT=ROOT/'experiments/results/e2_action_contribution_model_v1'
SEED='e2-action-contribution-group-split-v1:'


def read(p):return json.loads(Path(p).read_text())
def group(task):
 native=task['metadata']['native_id']
 return 'qa:'+native if task['kind']=='qa'else 'rec:'+native.rsplit(':',1)[0]
def split_commit(tasks):
 groups=sorted({group(t)for t in tasks});splits={}
 for kind in ('qa','rec'):
  ordered=sorted([g for g in groups if g.startswith(kind+':')],key=lambda g:hashlib.sha256((SEED+g).encode()).hexdigest())
  a=int(.6*len(ordered));b=int(.8*len(ordered))
  for i,g in enumerate(ordered):splits[g]='fit'if i<a else 'calibration'if i<b else 'diagnostic'
 return {'seed':SEED,'group_split':splits,'task_groups':{t['id']:group(t)for t in tasks},'unit':'original question/native conversation; task kind stratified 60/20/20 fixed SHA order','label_use':'none in selection','scope':'E2 reused train-side random-action development, not independent final test'}

def action_features(task,source_id,before_decision=None,version=0,visible=0,doubt=None):
 s=task['source_catalog'][source_id];words=lambda x:set(re.findall(r'[a-z]{3,}',str(x).lower()))
 q=words(task['prompt']);title=words(s.get('title',''));d=doubt or {}
 # No source body content, final report, actual usage, accepted flag or label.
 return {'task_kind':task['kind'],'source_kind':str(s.get('kind','unknown')),'source_chars':float(len(s.get('text',''))),'query_chars':float(len(task['prompt'])),'source_title_overlap':len(q&title)/max(1,len(q)),'retrieval_cosine':float(s.get('retrieval_cosine',0.)),'state_version':float(version),'visible_sources':float(visible),'candidate_count':float(len(task.get('candidates',[]))),'baseline_abstain':float(before_decision=='abstain'),'doubt_type':str(d.get('kind',d.get('type','source_lookup')))}

class E2ActionCalibrator:
 def __init__(self,path):
  self.path=Path(path);self.bundle=joblib.load(path);self.sha256=sha(self.path)
 def predict(self,task,source_id,before_decision=None,version=0,visible=0,doubt=None):
  x=[action_features(task,source_id,before_decision,version,visible,doubt)];b=self.bundle
  gain=float(b['gain'].predict(x)[0]+b['gain_bias']);damage=float(np.clip(b['damage'].predict(x)[0],0,1))
  if b['damage_calibrator'] is not None:damage=float(b['damage_calibrator'].predict([damage])[0])
  else:damage=b['calibration_damage_prior']
  return {'estimated_gain':gain,'estimated_damage_rate':damage,'model_sha256':self.sha256,'scope':'random-source action developmental predictor; not semantic truth or certified causal VOI'}


def prepare(run,output):
 tasks=read(run/'inputs.json');commit=split_commit(tasks);create(output/'split_commitment.json',commit)
 create(output/'feature_protocol.json',{'source_inputs_sha256':sha(run/'inputs.json'),'features':'task/source metadata, lengths, title overlap, retrieval cosine, pre-action decision/visibility; no outcome features','reward':'native accuracy/count delta, failures preserve actual before/after zero where transition observed','model':'DictVectorizer + Ridge alpha10 solverlsqr; fixed before labels','calibration':'gain residual bias and held calibration damage isotonic if both classes; otherwise Laplace prior','held_diagnostic':'never fit model or calibrator on diagnostic groups'})
 return tasks,commit


def fit(run=RUN,output=OUT):
 run=Path(run);output=Path(output);tasks,commit=prepare(run,output)
 seal=read(run/'generation_complete.json');cases=[read(run/'cases'/f'case_{i}.json')for i in range(len(tasks))]
 if seal['hashes']!=[fingerprint(c)for c in cases]or seal['task_ids']!=[t['id']for t in tasks]:raise ValueError('complete generation seal mismatch')
 labels=read(run/'labels.after_generation.json');rows=[];seen=set()
 for task,c in zip(tasks,cases):
  if c['task_id']!=task['id']:raise ValueError('case identity changed')
  e=c['event'];sid=e['source_action']['source_id'];key=e['request_hash']
  if key in seen:continue
  seen.add(key);reward=quality(c['after_decision'],labels[c['task_id']])-quality(c['before_decision'],labels[c['task_id']])
  x=action_features(task,sid,c['before_decision'],e['expected_state_version'],len(e['source_action']['before_visible_ids']))
  usage=e.get('usage',{});cost=(usage.get('prompt_tokens',0)*.15+usage.get('completion_tokens',0)*1.5)/1e6
  baseline=c.get('baseline_event');eligible=task['kind']!='qa' or (baseline is not None and baseline['valid'] and c['before_decision']!='abstain')
  available=max(1,len(task['source_catalog'])-len(e['source_action']['before_visible_ids']))
  rows.append({'eligible_gain_fit':eligible,'baseline_failure_or_abstention':not eligible,'design_source_probability':1/available,'task_id':task['id'],'group':group(task),'split':commit['group_split'][group(task)],'kind':task['kind'],'request_hash':key,'features':x,'reward':reward,'damage':float(reward<0),'valid':e['valid'],'accepted':c['accepted'],'configured_review_cny':cost,'usage_known':bool(usage)})
 create(output/'observations.json',rows)
 subsets={name:[r for r in rows if r['split']==name and r['eligible_gain_fit']]for name in ('fit','calibration','diagnostic')}
 if any(not x for x in subsets.values()):raise ValueError('empty group split')
 def model():return make_pipeline(DictVectorizer(sparse=True),Ridge(alpha=10.,solver='lsqr'))
 train=subsets['fit'];cal=subsets['calibration'];gain=model().fit([r['features']for r in train],[r['reward']for r in train]);damage=model().fit([r['features']for r in train],[r['damage']for r in train]);cx=[r['features']for r in cal]
 bias=float(np.mean(np.asarray([r['reward']for r in cal])-gain.predict(cx)));dy=[r['damage']for r in cal];iso=None
 if len(set(dy))>1:iso=IsotonicRegression(out_of_bounds='clip').fit(np.clip(damage.predict(cx),0,1),dy)
 bundle={'gain':gain,'damage':damage,'gain_bias':bias,'damage_calibrator':iso,'calibration_damage_prior':(sum(dy)+1)/(len(dy)+2),'schema':'e2-action-contribution-v1'}
 path=output/'action_model.joblib'
 if path.exists():raise ValueError('frozen model exists; use new version')
 output.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,path);predictor=E2ActionCalibrator(path);predictions=[]
 for row in subsets['diagnostic']:
  # Use stored PRE-action feature records directly; no training on held outcomes.
  x=[row['features']];g=float(gain.predict(x)[0]+bias);p=float(np.clip(damage.predict(x)[0],0,1));p=float(iso.predict([p])[0])if iso is not None else bundle['calibration_damage_prior']
  predictions.append({'task_id':row['task_id'],'kind':row['kind'],'observed_reward':row['reward'],'predicted_gain':g,'predicted_damage':p,'observed_damage':row['damage']})
 create(output/'diagnostic_predictions.json',predictions)
 summary={'observations':len(rows),'excluded_baseline_failure_or_abstention':sum(not r['eligible_gain_fit']for r in rows),'splits':{k:len(v)for k,v in subsets.items()},'model_sha256':sha(path),'source_generation_seal_sha256':sha(run/'generation_complete.json'),'diagnostic':{kind:{'n':len(rr),'gain_mse':float(np.mean([(r['predicted_gain']-r['observed_reward'])**2 for r in rr]))if rr else None,'observed_positive':sum(r['observed_reward']>0 for r in rr),'observed_negative':sum(r['observed_reward']<0 for r in rr)}for kind in ('qa','rec')for rr in [[r for r in predictions if r['kind']==kind]]},'scope':'group-held developmental diagnostic, no final benchmark claim; random selection compares sampled actions, not all possible actions; QA abstention-baseline outcomes separately require reporting'}
 create(output/'summary.json',summary);print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--prepare-only',action='store_true');p.add_argument('--predict-task',type=Path);p.add_argument('--model',type=Path);p.add_argument('--source-id');a=p.parse_args()
 if a.predict_task:
  t=read(a.predict_task);print(json.dumps(E2ActionCalibrator(a.model).predict(t,a.source_id,t['initial']),indent=2))
 elif a.prepare_only:prepare(a.run_root,a.output);print('Group split and feature protocol frozen before labels.')
 else:fit(a.run_root,a.output)
