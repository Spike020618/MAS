"""Native Sparse/KBRD/RRF controls from fold80 fits, no LLM or test labels."""
import math
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from .data import read,labels_for

def prepare(root):
 import numpy as np
 tasks=read(root/'data/test.json');rec=[t for t in tasks if t['kind']=='rec'];base=ROOT/'experiments/results/native_gain_calibration_v1';ids=read(base/'hold20.ids.json');lookup={k:i for i,k in enumerate(ids)}
 missing=[t['id']for t in rec if t['metadata']['native_id']not in lookup]
 if missing:
  create(root/'rank_references_unavailable.json',{'reason':'not all recommendation tasks belong to fold80 hold20 prediction inputs','tasks':missing});return
 items=list(map(str,read(base/'native_item_ids.json')));sr=np.load(base/'sparse/hold20.native_ranks.npy',mmap_mode='r');kr=np.load(base/'kbrd/hold20.native_ranks.npy',mmap_mode='r');pred={}
 for t in rec:
  row=lookup[t['metadata']['native_id']];rankings={}
  for name,ranks in (('Sparse-fold80',sr[row]),('KBRD-rec-fold80',kr[row])):
   rankings[name]=[items[j]for j in sorted(range(len(items)),key=lambda j:(int(ranks[j]),int(items[j])))[:10]]
  rankings['RRF60-fold80']=[items[j]for j in sorted(range(len(items)),key=lambda j:(-(1/(60+float(sr[row,j]))+1/(60+float(kr[row,j]))),int(items[j])))[:10]]
  pred[t['id']]=rankings
 create(root/'rank_references.json',pred);create(root/'rank_reference_manifest.json',{'prediction_sha256':sha(root/'rank_references.json'),'task_hash':fingerprint(tasks),'fit_scope':'fold80 predictors; hold20 task-group exclusion checked in dataset expansion','source_files_sha256':{str(p.relative_to(ROOT)):sha(p)for p in (base/'hold20.ids.json',base/'native_item_ids.json',base/'sparse/hold20.native_ranks.npy',base/'kbrd/hold20.native_ranks.npy')},'native_full_ranks':True,'labels_used':False,'scope':'recommendation component reproductions; not full dialogue-system paper reproduction'})

def score(root):
 path=root/'rank_references.json'
 if not path.exists():return
 manifest=read(root/'rank_reference_manifest.json');tasks=read(root/'main/inputs.json')
 if sha(path)!=manifest['prediction_sha256']or fingerprint(tasks)!=manifest['task_hash']:raise ValueError('ranking references or main task set changed')
 gold=labels_for(tasks,root/'main/generation_complete.json');pred=read(path);result={}
 for name in ('Sparse-fold80','KBRD-rec-fold80','RRF60-fold80'):
  hits=0;targets=0;ndcg=0.
  for tid,rankings in pred.items():
   ranking=rankings[name];ys=list(map(str,gold[tid]['targets']));targets+=len(ys);hits+=sum(y in ranking for y in ys);ndcg+=sum(1/math.log2(ranking.index(y)+2)for y in ys if y in ranking)
  result[name]={'contexts':len(pred),'item_targets':targets,'hits_at_10':hits,'recall_at_10':hits/max(1,targets),'target_averaged_ndcg_at_10':ndcg/max(1,targets),'model_api_calls':0}
 create(root/'rank_baselines/summary.json',{'methods':result,'scope':manifest['scope'],'source_manifest_sha256':sha(root/'rank_reference_manifest.json')});return result
