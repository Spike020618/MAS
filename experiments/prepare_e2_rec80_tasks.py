"""Prefix-only material packets for precommitted E2 training queries; no labels."""
import json,numpy as np
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha,fingerprint
from experiments.paired_deliberation.native_review_diagnostic_packets import PrefixOnlyEvidence,prepare_safe_directory
from experiments.evidence_consensus.task_adapter import adapt_source

def run():
 out=ROOT/'experiments/results/e2_action_acquisition_rec80_v1';commit=json.load(open(out/'blind_group_commitment.json'));gain=ROOT/'experiments/results/native_gain_calibration_v1';ranker=ROOT/'experiments/results/native_calibrated_challenge_v3'
 allrows={r['id']:r for r in json.load(open(gain/'hold20.inputs.json'))};rows=[allrows[c]for c in commit['ids']];builder=PrefixOnlyEvidence(prepare_safe_directory(out/'prefix-only-serving-data'),rows)
 ids=json.load(open(gain/'hold20.ids.json'));lookup={c:i for i,c in enumerate(ids)};cand=np.load(ranker/'candidate_ids.npy',mmap_mode='r');offset=np.load(ranker/'candidate_offsets.npy',mmap_mode='r');mu=np.load(ranker/'hold20.expected_count.npy',mmap_mode='r');tasks=[]
 for row in rows:
  cid=row['id'];q=lookup[cid];sl=slice(offset[q],offset[q+1]);pool=np.asarray(cand[sl]);order=np.argsort(-mu[sl],kind='stable');ranking=[str(pool[j])for j in order];taskid='e2:rec:'+cid
  serving={'id':cid,'text':row.get('original_text',row['text']),'candidates':[{'movie_id':m,'title':builder.names[int(m)]}for m in ranking]};evidence=builder.augment(serving,history_limit=3,kg_limit=3)
  allowed=set(ranking[:24]);catalog={sid:adapt_source(sid,c,taskid)for sid,c in evidence['sources'].items()if c.get('kind')in ('dialogue','historical_prediction_basis')or str(c.get('movie_id'))in allowed}
  sid='forecast:shared_prior:'+cid;catalog[sid]=adapt_source(sid,{'source_id':sid,'kind':'model_forecast','allowed_use':'prediction_basis','text':'Training-derived prior ranking, NOT observed target: '+json.dumps(ranking)},taskid)
  tasks.append({'id':taskid,'kind':'rec','prompt':evidence['text'],'options':[],'candidates':serving['candidates'],'initial':ranking,'source_catalog':catalog,'metadata':{'native_dataset':'CRSLab-ReDial','native_id':cid,'query_labels_available':False,'history_scope':'E2 train-side action fitting, count ranker may have fit these training queries; not generalization evaluation','prefix_only_inputs':True,'source_selection':'all visible dialogue/history plus KG for prior top24, no label-based selection'}})
 create(out/'tasks.json',tasks);create(out/'materials_complete.json',{'tasks':len(tasks),'task_ids':[t['id']for t in tasks],'inputs_sha256':sha(out/'tasks.json'),'commitment_sha256':sha(out/'blind_group_commitment.json'),'query_future_labels_read':False,'model_api_calls':0})
 print('prepared',len(tasks))
if __name__=='__main__':run()
