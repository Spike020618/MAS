"""Commit fresh training-side recommendation groups before any outcome reads."""
import json,hashlib
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha

def run():
 gain=ROOT/'experiments/results/native_gain_calibration_v1';pilot=ROOT/'experiments/results/evidence_consensus_e1_tasks_v1/tasks.json'
 tasks=json.load(open(pilot));exclude={t['metadata'].get('native_id',t['id']).rsplit(':',1)[0]for t in tasks if t['kind']=='rec'}
 # Also use actual task-id suffix: e1:rec:<conversation>:<turn>.
 exclude|={t['id'].split(':')[-2]for t in tasks if t['kind']=='rec'}
 ids=json.load(open(gain/'hold20.ids.json'));by={}
 for cid in ids:
  conv=cid.rsplit(':',1)[0]
  if conv not in exclude:by.setdefault(conv,[]).append(cid)
 ordered=sorted(by,key=lambda c:hashlib.sha256(('e2-rec80-blind-20261028:'+c).encode()).hexdigest())
 chosen=[min(by[c],key=lambda q:hashlib.sha256(('e2-query:'+q).encode()).hexdigest())for c in ordered[:80]]
 assert len(chosen)==80 and len({q.rsplit(':',1)[0]for q in chosen})==80
 out=ROOT/'experiments/results/e2_action_acquisition_rec80_v1'
 plan={'version':'e2-rec80-blind-commitment-v1','ids':chosen,'selection':'conversation hash then query hash, no outcome access','excluded_e1_conversations':sorted(exclude),'source_ids_sha256':sha(gain/'hold20.ids.json'),'pilot_tasks_sha256':sha(pilot),'target_labels_read':False,'status':'group commitment only, material packets and model calls pending','requested_total160_status':'only Rec80 groups committed; QA80 not claimed'}
 create(out/'blind_group_commitment.json',plan);print(json.dumps(plan,indent=2))
if __name__=='__main__':run()
