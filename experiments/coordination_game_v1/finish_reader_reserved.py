"""Continue only unsent reader slots. Preserve unknowns; publish partial provenance."""
import json
from pathlib import Path
from .schema import digest
from .live_runtime import DurableRuntime,atomic
from .collect import PacedTransport
from .transport import Transport
from experiments.audits.final_integrity_audit import score
ROOT=Path(__file__).resolve().parents[2]
def main():
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False)
 root=ROOT/'experiments/results/evidence_use_repair_v2';plan=json.loads((root/'collect/plan.json').read_text());requests=json.loads((root/'requests.json').read_text())
 runtime=DurableRuntime(root/'collect',plan,PacedTransport(Transport('qwen_flash')),True,.3)
 protocol={'scope':'continue reserved slots only; never resend intent; missing transport completion is unknown, not service rejection','source_hash':digest(Path(__file__).read_text())}
 atomic(root/'continuation_protocol.json',protocol)
 for slot,req in requests:
  receipt=json.loads(runtime.path(slot).read_text())
  if receipt['status']!='reserved':continue
  try:runtime.call(slot,req);print('sent',slot,flush=True)
  except RuntimeError:print('unknown retained',slot,flush=True)
 receipts=[json.loads(runtime.path(s).read_text())for s,r in requests]
 unknown=[r['slot']for r in receipts if r['status']=='intent']
 status={'complete':not unknown and all(r['status']in('complete','rejected')for r in receipts),'unknown_slots':unknown,'receipt_hashes':[digest(r)for r in receipts]}
 atomic(root/'generation_status.json',status)
 if status['complete']:runtime.seal([s for s,r in requests])
 vault=json.loads((ROOT/'experiments/results/coordination_game_s3_prepare_v1/private/scoring.json').read_text())
 old=json.loads((ROOT/'experiments/results/evidence_use_repair_v1/summary.json').read_text())['records'];rows=[r for r in old if r['method']=='old_observations']
 missing=[];citation_errors=[]
 for receipt in receipts:
  tid,arm=receipt['slot'].split('/');parsed=(receipt.get('response')or{}).get('parsed')
  if receipt['status']!='complete'or not isinstance(parsed,dict)or not isinstance(parsed.get('answer'),str):missing.append(receipt['slot']);continue
  row={'task_id':tid,'method':arm,'answer':parsed['answer'],'f1':score(parsed['answer'],vault[tid]['answer'])};rows.append(row)
  payload=json.loads(receipt['request']['messages'][-1]['content'])
  if arm.startswith('chain'):
   for step in parsed.get('steps',[]):
    if not isinstance(step,dict)or step.get('source_id')not in payload['sources']or not isinstance(step.get('exact_quote'),str)or not step['exact_quote']or step['exact_quote']not in payload['sources'][step['source_id']]:citation_errors.append(receipt['slot'])
 arms=['old_observations','raw_only','chain_same5','chain_full10'];by={(r['task_id'],r['method']):r for r in rows};complete=[tid for tid in plan['task_ids']if all((tid,arm)in by for arm in arms)]
 means={arm:sum(by[(t,arm)]['f1']for t in complete)/len(complete)for arm in arms}if complete else{}
 usage=[(r.get('response')or{}).get('usage',{})for r in receipts];cost=sum((u.get('prompt_tokens',0)*.15+u.get('completion_tokens',0)*1.5)/1e6 for u in usage)
 result={'generation_status':status,'paired_complete_cases':len(complete),'paired_complete_case_ids':complete,'paired_native_f1':means,'missing_slots':missing,'citation_error_slots':citation_errors,'records':rows,'reported_usage_cny_including_reused_raw':cost,
  'scope':'exposed development diagnostic; common complete-case comparison excludes transport/schema missingness, which is disclosed; no independent confirmation; no successful-batch claim while unknowns remain'}
 atomic(root/'result_audit.json',result);print(json.dumps({k:v for k,v in result.items()if k not in('records','generation_status')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
