"""Frozen new-question reader-package validation; all arms share model/output cap."""
import argparse,json
from pathlib import Path
from .stage3_prepare import prepare
from .schema import digest
from .live_runtime import DurableRuntime,atomic
from .transport import Transport
from .collect import PacedTransport
from .collection import message
from .source_bound_reader import request as bound_request,bind
from experiments.audits.final_integrity_audit import score
from experiments.audits.consolidate_experiments import paired
ROOT=Path(__file__).resolve().parents[2]
def run(execute=False):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False)
 root=ROOT/'experiments/results/reader_validation_new24_v1';data=root/'data'
 if not data.exists():prepare(data,count=24)
 tasks=json.loads((data/'tasks.json').read_text());arms=[];source_sets={}
 for t in tasks:
  ids=list(dict.fromkeys([t['member_initial_sources'][str(m)][0]for m in range(3)]+t['candidates'][:2]));limited=[s for s in t['sources']if s['id']in ids];raw=lambda ss:{s['id']:s['title']+': '+' '.join(s['sentences'])for s in ss}
  for name,req,sources in [('plain5',message(t['question'],raw(limited),'readout'),limited),('bound_chain5',bound_request(t['question'],limited),limited),('plain_full',message(t['question'],raw(t['sources']),'readout'),t['sources'])]:
   slot=t['id']+'/'+name;arms.append((slot,req));source_sets[slot]=sources
 plan={'schema':'reader-validation-new24.v1','task_ids':[t['id']for t in tasks],'request_hash':digest(arms),'source_hash':digest(Path(__file__).read_text()),'max_requests':72,'max_cny':.5,'input_byte_cap':32768,'output_token_cap':768,'scope':'24 new locally recorded questions excluding P1/60/200/458; public validation development replication, not untouched test certification; fixed hash selection; same provider and output cap; reader-package comparison, not pure prompt-only ablation'}
 rt=DurableRuntime(root/'collect',plan,PacedTransport(Transport()),run=execute,approved_cny=.5);rt.reserve_batch(arms)
 if not execute:return {'prepared':72,'plan':plan}
 for slot,req in arms:
  try:rt.call(slot,req);print('completed',slot,flush=True)
  except RuntimeError:print('unknown or unavailable retained',slot,flush=True)
 receipts=[json.loads(rt.path(slot).read_text())for slot,req in arms];unresolved=[r['slot']for r in receipts if r['status']not in('complete','rejected')]
 atomic(root/'generation_status.json',{'complete':not unresolved,'unresolved':unresolved})
 if unresolved:return {'complete':False,'unresolved':unresolved,'effects_available':False}
 rt.seal([s for s,r in arms]);vault=json.loads((data/'private/scoring.json').read_text());rows=[]
 for r in receipts:
  tid,arm=r['slot'].split('/');parsed=(r.get('response')or{}).get('parsed');valid=isinstance(parsed,dict)and isinstance(parsed.get('answer'),str);answer=parsed['answer']if valid else'unknown';binding=bind(parsed,source_sets[r['slot']])if arm=='bound_chain5'else None
  rows.append({'task_id':tid,'arm':arm,'answer':answer,'valid_answer':valid,'binding_valid':binding['valid']if binding else None,'binding_errors':binding['errors']if binding else [],'native_f1':score(answer,vault[tid]['answer'])if valid else 0.})
 by={(r['task_id'],r['arm']):r for r in rows};ids=plan['task_ids'];means={a:sum(by[t,a]['native_f1']for t in ids)/24 for a in('plain5','bound_chain5','plain_full')};comparison=paired([by[t,'bound_chain5']['native_f1']-by[t,'plain5']['native_f1']for t in ids]);u=[(r.get('response')or{}).get('usage',{})for r in receipts]
 result={'scope':plan['scope'],'means':means,'chain_minus_plain5':comparison,'chain_improved':sum(by[t,'bound_chain5']['native_f1']>by[t,'plain5']['native_f1']for t in ids),'chain_harmed':sum(by[t,'bound_chain5']['native_f1']<by[t,'plain5']['native_f1']for t in ids),'binding_failures':sum(r['binding_valid']is False for r in rows),'invalid_answers':sum(not r['valid_answer']for r in rows),'estimated_cny':sum((x.get('prompt_tokens',0)*.15+x.get('completion_tokens',0)*1.5)/1e6 for x in u),'records':rows}
 atomic(root/'summary.json',result);return {k:v for k,v in result.items()if k!='records'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();print(json.dumps(run(a.run),ensure_ascii=False,indent=2))
