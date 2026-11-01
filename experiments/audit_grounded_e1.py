"""Sealed v4/v5 development comparison; no network or new labels."""
import json,collections,argparse
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint

def audit(root):
 root=Path(root);old=ROOT/'experiments/results/unified_e1_v4';plan=json.load(open(root/'plan.json'));seal=json.load(open(root/'generation_complete.json'))
 cases=[json.load(open(root/'cases'/f'case_{i}.json'))for i in range(plan['count'])];assert seal['hashes']==[fingerprint(c)for c in cases]
 prior={c['task_id']:c for c in [json.load(open(p))for p in (old/'cases').glob('*.json')]};labels=json.load(open(root/'labels.after_generation.json'));out={'scope':'same24 reused development comparison, not confirmation','methods':{}}
 for method in plan['methods']:
  counts=collections.Counter();reasons=collections.Counter()
  for c in cases:
   a=c['methods'][method];b=prior[c['task_id']]['methods'][method];g=labels[c['task_id']]
   def hits(dec):
    return int(str(dec).lower()==str(g['answer']).lower())if c['kind']=='qa'else sum(str(t)in dec[:10]for t in g['targets'])
   counts[c['kind']+'_new_hits_vs_v4']+=max(0,hits(a['decision'])-hits(b['decision']));counts[c['kind']+'_lost_hits_vs_v4']+=max(0,hits(b['decision'])-hits(a['decision']))
   for e in a['events']:
    counts[c['kind']+'_accepted']+=bool(e.get('patch_accepted'));reasons[e.get('patch_reason','no_patch')]+=1
  out['methods'][method]={'counts':dict(counts),'reasons':dict(reasons)}
 out['v5_summary']=json.load(open(root/'summary.json'));out['physical']=json.load(open(root/'physical_summary.json'));create(root/'independent_v4_comparison.json',out);print(json.dumps(out,ensure_ascii=False,indent=2));return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/results/unified_e1_v5_grounded');a=p.parse_args();audit(a.output)
