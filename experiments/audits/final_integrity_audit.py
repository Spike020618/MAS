"""Read-only receipt integrity and independent Hotpot normalization audit; zero network."""
import json,re,string,hashlib
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def score(a,g):
 def norm(x):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',str(x).lower().translate(str.maketrans('','',string.punctuation))).split())
 a=norm(a);g=norm(g)
 if a!=g and(a in('yes','no','noanswer')or g in('yes','no','noanswer')):return 0.
 aa=a.split();gg=g.split();common=sum((Counter(aa)&Counter(gg)).values())
 return 2*common/(len(aa)+len(gg))if aa or gg else 1.
def audit(root):
 p=ROOT/'experiments/results'/root;seal=json.loads((p/'generation_complete.json').read_text());plan=json.loads((p/'plan.json').read_text());slots={};counts=Counter();errors=[]
 for f in (p/'slots').glob('*.json'):
  r=json.loads(f.read_text());slots[r['slot']]=r
  if r['request_hash']!=digest(r['request']):errors.append('request:'+r['slot'])
  if r['status']=='rejected':
   reason=r.get('reconciliation','')+' '+str(r.get('response',{}))
   cat='network_completion_unknown_abandoned'if 'unknown'in reason or 'network'in reason else'other_rejection'
   counts[cat]+=1
 if seal.get('plan_hash')!=digest(plan):errors.append('plan_hash')
 reconstructed=digest({**plan,'methods':['random','fixed','central','single_concentrated']})==seal.get('plan_hash')
 for item in seal['receipts']:
  if item['slot']not in slots or digest(slots[item['slot']])!=item['hash']:errors.append('receipt:'+item['slot'])
 panel=json.loads((p/'panel.json').read_text());records=panel['records']
 if panel['seal'].get('records_hash')!=digest(records):errors.append('panel_hash')
 return {'path':str(p),'errors':errors,'records':len(records),'sealed_receipts':len(seal['receipts']),'slots':len(slots),'rejections':dict(counts),'claimed_budget_violations':panel['seal'].get('budget_violations'),'leakage_zero_is_measured':False,'plan_hash_matches_after_restoring_methods':reconstructed},records

def main():
 out={}
 for name,data in [('coordination_game_s3_v1/collect','coordination_game_s3_prepare_v1'),('coordination_game_s3_power_v1/collect','coordination_game_s3_power_prepare_v1'),('coordination_game_s3_anchor_v1','coordination_game_s3_prepare_v1')]:
  result,rows=audit(name);vault=json.loads((ROOT/'experiments/results'/data/'private/scoring.json').read_text());means=defaultdict(list);changed=0
  for r in rows:
   gold=vault[r['task_id']]['answer'];v=score(r['answer'],gold);means[r.get('method','full_context')].append(v)
   a=set(r['answer'].lower().split());g=set(gold.lower().split());old=2*len(a&g)/(len(a)+len(g))if a or g else 1.;changed+=abs(v-old)>1e-12
  result['native_f1']={m:sum(x)/len(x)for m,x in means.items()};result['scores_changed_by_native_normalization']=changed;out[name]=result
 for sub in ('member-0','member-1','member-2','reader'):
  p=ROOT/'experiments/results/coordination_game_s4_hetero_v1'/sub
  seal=json.loads((p/'generation_complete.json').read_text());slots={json.loads(f.read_text())['slot']:json.loads(f.read_text())for f in(p/'slots').glob('*.json')}
  errors=[r['slot']for r in seal['receipts']if r['slot']not in slots or digest(slots[r['slot']])!=r['hash']]
  out['hetero/'+sub]={'receipt_errors':errors,'receipts':len(seal['receipts'])}
 p=ROOT/'experiments/results/consensus_mechanism_paper_v2r2/benchmark';seal=json.loads((p/'generation_complete.json').read_text());errors=[]
 for i,(tid,h)in enumerate(zip(seal['task_ids'],seal['hashes'])):
  case=json.loads((p/'cases'/f'case_{i}.json').read_text())
  if digest(case)!=h or case['task_id']!=tid:errors.append(i)
 out['benchmark']={'sealed_cases':len(seal['task_ids']),'case_hash_identity_errors':errors}
 target=ROOT/'experiments/results/consolidation_2026_10_31/integrity/summary.json';target.write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
