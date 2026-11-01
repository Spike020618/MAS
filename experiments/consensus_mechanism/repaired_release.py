"""Fail-closed release entry: interface and learning signal precede main comparison."""
import argparse,json,copy
from pathlib import Path
from .data import read
from .runner import collect_training,fit,evaluate
from .engine import PRIMARY
from experiments.consensus_recommendation.common import ROOT,create

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','main']);p.add_argument('--root',type=Path,default=ROOT/'experiments/results/decision_linked_release_v1');p.add_argument('--probe',type=Path,default=ROOT/'experiments/results/decision_linked_probe_v2');p.add_argument('--run',action='store_true');p.add_argument('--max-cny',type=float,default=1.);a=p.parse_args()
 if not 0<a.max_cny<=5:raise ValueError('per-stage hard cap is 5 CNY; default 1 CNY')
 source=ROOT/'experiments/results/consensus_mechanism_paper_v2/data'
 if a.root.resolve()==source.parent.resolve():raise ValueError('old paper release is read-only')
 if a.stage=='prepare':
  for name in ['train','test']:
   tasks=copy.deepcopy(read(source/(name+'.json')))
   for t in tasks:t['metadata']['decision_linked_v1']=True
   create(a.root/'data'/(name+'.json'),tasks)
  print('Prepared new linked-interface inputs. No API requests.');return
 if not a.run:print('No API requests; explicit --run required.');return
 gate=read(a.probe/'summary.json')
 if not gate.get('promotion_gate'):raise RuntimeError('Blocked: paired interface/early-signal probe did not pass. Do not expand the matrix.')
 if a.stage=='train':
  stats=collect_training(a.root/'data/train.json',a.root/'train',max_cny=a.max_cny)
  if not stats['contract_gate_passed']:raise RuntimeError('training contract gate failed')
  fit(a.root/'train',a.root/'model');rows=read(a.root/'model/observations.json');signals={c:sum(r['partition']=='fit'and r['component']==c and r['reward']>0 for r in rows)for c in ['singleton','interaction']}
  create(a.root/'learning_gate.json',{'positive_fit':signals,'passed':signals['singleton']>=3 and signals['interaction']>=2,'scope':'minimum exploration signal; not statistical evidence'})
 else:
  if not read(a.root/'learning_gate.json')['passed']:raise RuntimeError('Blocked: insufficient nonzero learning signal')
  evaluate(a.root/'data/test.json',a.root/'main',PRIMARY,2,'qwen_flash',a.root/'model/model.joblib',a.max_cny)
if __name__=='__main__':main()
