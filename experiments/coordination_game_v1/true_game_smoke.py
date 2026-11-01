"""One real round wiring test. Lexical proxy, not learned contributions or efficacy."""
import json,argparse,re
from pathlib import Path
from .schema import Action,MemberState,digest
from .true_live_round import run_live_round
from .live_runtime import DurableRuntime,atomic
from .transport import Transport
from .collect import PacedTransport
ROOT=Path(__file__).resolve().parents[2]
def run(execute=False):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False)
 tasks=json.loads((ROOT/'experiments/results/coordination_game_s3_prepare_v1/tasks.json').read_text())
 task=sorted(tasks,key=lambda t:digest(['reader-repair-v1',t['id']]))[0]
 members=[MemberState(str(m),tuple(task['member_initial_sources'][str(m)]))for m in range(3)]
 actions=[Action(str(m)+'-'+c,str(m),'question-relation',(c,))for m in range(3)for c in task['candidates']]
 tokenize=lambda text:set(re.findall('[a-z0-9]{3,}',text.lower()))-{'the','and','what','which','was','who','with','that'}
 q=tokenize(task['question']);catalog={s['id']:tokenize(s['title']+' '+' '.join(s['sentences']))for s in task['sources']}
 def value(chosen):
  covered=set()
  for a in chosen:covered|=catalog[a.source_ids[0]]&q
  return len(covered)/max(1,len(q))-.025*len(chosen)
 plan={'schema':'true-game-live-smoke.v1','task_id':task['id'],'prediction_id':'untrained-query-coverage-proxy.v1','max_requests':3,'max_cny':.03,'input_byte_cap':32768,'output_token_cap':768,'source_hash':digest(Path(__file__).read_text()),'scope':'one real API integration check, no learned contribution or coordination quality claim'}
 root=ROOT/'experiments/results/true_game_live_smoke_v1';runtime=DurableRuntime(root,plan,PacedTransport(Transport()),run=execute,approved_cny=.03)
 if not execute:return {'prepared':plan}
 result=run_live_round(runtime,task,members,[],actions,value,plan['prediction_id'],mode='pair_game',call_budget=3)
 runtime.seal(result['slots']);atomic(root/'result.json',result)
 return {'selected':result['executed_action_ids'],'calls':len(result['slots']),'unilateral_residual':result['certificate']['unilateral_residual'],'permitted_pair_residual':result['certificate']['permitted_pair_residual'],'answer':(result.get('readout')or{}).get('parsed'),'scope':plan['scope']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();print(json.dumps(run(a.run),ensure_ascii=False,indent=2))
