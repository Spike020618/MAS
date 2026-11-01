"""Fit observed net action reward from fit groups; calibrate without test labels."""
import argparse,json
from pathlib import Path
from experiments.consensus_recommendation.common import create,fingerprint,sha
from experiments.prepare_disjoint_acquisition import group
from experiments.evidence_consensus.protocol_v3 import quality

def fit(root,output):
 import joblib,numpy as np
 from sklearn.pipeline import make_pipeline
 from sklearn.feature_extraction import DictVectorizer
 from sklearn.linear_model import Ridge
 tasks=json.loads((root/'inputs.json').read_text());commit=json.loads((root/'acquisition_split_commitment.json').read_text());plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'generation_complete.json').read_text());cases=[json.loads((root/'cases'/f'case_{i}.json').read_text())for i in range(len(tasks))]
 if seal['task_ids']!=[t['id']for t in tasks]or seal['hashes']!=[fingerprint(c)for c in cases]:raise ValueError('seal mismatch')
 if commit['task_split']!={t['id']:t['metadata']['acquisition_partition']for t in tasks}:raise ValueError('commitment mismatch')
 if any(v not in ('fit','calibration')for v in commit['task_split'].values()):raise ValueError('diagnostic data prohibited')
 for name,h in plan['source_sha256'].items():
  if sha(root/'source_snapshot'/name)!=h:raise ValueError('source snapshot mismatch')
 owners={}
 for t in tasks:
  g=group(t);part=commit['task_split'][t['id']]
  if owners.setdefault(g,part)!=part:raise ValueError('native group overlap')
 labels=json.loads((root/'labels.after_generation.json').read_text());rows=[]
 for t,c in zip(tasks,cases):
  if c['task_id']!=t['id']:raise ValueError('case mismatch')
  # Operational failures keep zero realized reward; no optimistic valid-only selection.
  if c['pre_action_features']is None:continue
  eligible=t['kind']!='qa'or c['before_decision']!='abstain'
  if not eligible:continue
  reward=quality(c['after_decision'],labels[t['id']])-quality(c['before_decision'],labels[t['id']])
  rows.append({'task_id':t['id'],'partition':commit['task_split'][t['id']],'features':c['pre_action_features'],'reward':reward,'valid':c['event']['valid']})
 parts={part:[r for r in rows if r['partition']==part]for part in ('fit','calibration')}
 if any(not v for v in parts.values()):raise ValueError('empty eligible partition; no fallback')
 gain=make_pipeline(DictVectorizer(),Ridge(alpha=10,solver='lsqr')).fit([r['features']for r in parts['fit']],[r['reward']for r in parts['fit']]);cal=parts['calibration'];bias=float(np.mean([r['reward']for r in cal])-gain.predict([r['features']for r in cal]).mean())
 create(output/'training_tasks.json',tasks);create(output/'split_commitment.json',commit);create(output/'observations.json',rows)
 model=output/'model.joblib';bundle={'gain':gain,'gain_bias':bias,'feature_schema':'v3-query-action-reviewer-v1','split_sha256':sha(output/'split_commitment.json')}
 # Re-running a finished pipeline verifies lineage and preserves its model.
 if model.exists():
  existing=joblib.load(model)
  if existing.get('training_seal_sha256')!=sha(root/'generation_complete.json'):raise ValueError('existing model has different training seal')
 else:
  bundle['training_seal_sha256']=sha(root/'generation_complete.json');joblib.dump(bundle,model)
 create(output/'summary.json',{'counts':{k:len(v)for k,v in parts.items()},'positive_fit_rewards':sum(r['reward']>0 for r in parts['fit']),'negative_fit_rewards':sum(r['reward']<0 for r in parts['fit']),'model_sha256':sha(model),'feature_schema':bundle['feature_schema'],'scope':'observational net-reward estimator; not semantic verification; no diagnostic labels'})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();fit(args.run_root,args.output)
