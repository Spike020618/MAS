"""Independent partial-development result accounting; never labels unknown as success."""
import json
from pathlib import Path
from experiments.audits.consolidate_experiments import paired
from experiments.audits.final_integrity_audit import score
from experiments.coordination_game_v1.schema import digest
ROOT=Path(__file__).resolve().parents[2]
def main():
 root=ROOT/'experiments/results/evidence_use_repair_v2';d=json.loads((root/'result_audit.json').read_text());rows=d['records'];by={(r['task_id'],r['method']):r for r in rows};ids=d['paired_complete_case_ids'];base='old_observations';comparisons={}
 for arm in ('raw_only','chain_same5','chain_full10'):
  delta=[by[t,arm]['f1']-by[t,base]['f1']for t in ids]
  comparisons[arm]={**paired(delta),'improved':sum(x>0 for x in delta),'harmed':sum(x<0 for x in delta),'scope':'complete-case development estimate, not final sealed efficacy result'}
 roots=[ROOT/'experiments/results/evidence_use_repair_v1/collect',root/'collect'];receipts={};unknown=set()
 for r in roots:
  for p in(r/'slots').glob('*.json'):
   receipt=json.loads(p.read_text())
   if receipt['status']=='intent':unknown.add(receipt['slot']);continue
   if receipt['status']=='complete':receipts[digest(receipt)]=receipt
 usage=[(r.get('response')or{}).get('usage',{})for r in receipts.values()];cost=sum((u.get('prompt_tokens',0)*.15+u.get('completion_tokens',0)*1.5)/1e6 for u in usage)
 old=[r for r in rows if r['method']==base];chain=[r for r in rows if r['method']=='chain_same5'];total=12
 lower=sum(r['f1']for r in chain)/total;upper=(sum(r['f1']for r in chain)+(total-len(chain)))/total
 anchor_rows=json.loads((ROOT/'experiments/results/coordination_game_s3_anchor_v1/panel.json').read_text())['records']
 vault=json.loads((ROOT/'experiments/results/coordination_game_s3_prepare_v1/private/scoring.json').read_text())
 af={r['task_id']:score(r['answer'],vault[r['task_id']]['answer'])for r in anchor_rows}
 full_comparison={'old_full10_native_f1':sum(af[t]for t in ids)/len(ids),'new_chain_full10_native_f1':d['paired_native_f1']['chain_full10'],'new_minus_old':paired([by[t,'chain_full10']['f1']-af[t]for t in ids])}
 result={'full_source_matched_comparison':full_comparison,'scope':d['scope'],'batch_complete':d['generation_status']['complete'],'paired_cases':len(ids),'paired_native_f1':d['paired_native_f1'],'comparisons':comparisons,
 'chain_same5_all12_missingness_bounds':[lower,upper],'baseline_all12':sum(r['f1']for r in old)/12,
 'citation_error_responses':len(set(d['citation_error_slots'])),'unknown_completion_slots':sorted(unknown),'unique_known_complete_calls_two_revisions':len(receipts),'known_usage_estimated_cny_two_revisions':cost,'unknown_usage_cost_included':False,
 'decisions':{'promote_reader_to_formal_method':False,'reason':'small exposed diagnostic, unresolved completion, source-quote failures and no fresh confirmation','real_game_adapter':'true_policy_adapter plus true_live_round uses actual P0 solver; online learned coordination gain not tested'}}
 (root/'independent_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
