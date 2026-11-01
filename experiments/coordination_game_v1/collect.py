"""P1 real collection entry: paced transport, records adapter, seal, then sealed analysis.
Loads no scoring labels before the generation seal; fail-closed on unknown intents."""
import json,os,fcntl,time
from pathlib import Path
from .collection import collect_task,message
from .schema import digest
from .live_runtime import DurableRuntime
from .analysis import analyze
from .preregistration import PROTOCOL

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/'experiments/results/.qwen-native-dispatch.lock'

def pace(interval=None):
 interval=float(os.environ.get('MAS_DISPATCH_INTERVAL','3')if interval is None else interval)
 LEDGER.parent.mkdir(parents=True,exist_ok=True)
 with LEDGER.open('a+')as stream:
  fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);value=stream.read().strip();last=float(value)if value else 0
  remaining=interval-(time.time()-last)
  while remaining>0:time.sleep(min(10,remaining));remaining=interval-(time.time()-last)
  stream.seek(0);stream.truncate();stream.write(str(time.time()));stream.flush();fcntl.flock(stream,fcntl.LOCK_UN)

class PacedTransport:
 def __init__(self,transport,interval=None):self.transport=transport;self.interval=interval
 def __call__(self,request):
  pace(self.interval);return self.transport(request)

def source_text(source):return f"{source['title']}: {' '.join(source['sentences'])}"

def task_sources(task):
 order=task['source_order'];catalog={s['id']:s for s in task['sources']}
 initial=[{sid:source_text(catalog[sid])for sid in task['member_initial_sources'][str(m)]}for m in range(3)]
 action=[{sid:source_text(catalog[sid])}for sid in order[3:6]]
 return initial,action

def branch_kind(name):
 if name=='empty':return 'empty'
 if name.startswith('single_'):return 'single'
 if name.startswith('joint_'):return 'same'
 if name.startswith('cross_'):return 'cross'
 raise ValueError('unknown branch '+name)

def matched(name):
 if not name.startswith(('joint_','cross_')):return {}
 m=int(name.rsplit('_',1)[-1])if name.startswith('joint_')else int(name.split('_')[1])
 if name.startswith('joint_'):return {'matched_single_branches':[f'single_{m}_0',f'single_{m}_1']}
 return {'matched_single_branches':[f'single_{m}_0',f'single_{(m+1)%3}_1'],'matched_same_branch':f'joint_{m}'}

def readout_answer(response):
 parsed=response.get('parsed')if isinstance(response,dict)else None
 if isinstance(parsed,dict)and isinstance(parsed.get('answer'),str)and parsed['answer'].strip():
  return parsed['answer'].strip(),True
 return 'noanswer',False

def build_plan(manifest,budget):
 # input_byte_cap 8192: measured real P1 requests peak at ~5.1KB (readouts accumulating source texts);
 # the P0 draft's 4096 draft cap rejected 21/444 readouts. Cost accounting stays actual-size based,
 # so the approved max_cny cap remains the binding spend limit.
 return {'schema':'coordination-p1-collect.v1','tasks':manifest['tasks'],'repeats':manifest['repeats'],
  'task_ids':manifest['task_ids'],'tasks_sha256':manifest['tasks_sha256'],'source_split':manifest['source_split'],
  'max_requests':budget['requests'],'max_cny':budget['estimated_upper_cny_configured_rates'],
  'input_byte_cap':8192,'output_token_cap':768,'dispatch_interval_seconds':float(os.environ.get('MAS_DISPATCH_INTERVAL','3'))}

def collect(data_root,out_root,provider='qwen_flash',approved_cny=None):
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env.local',override=False)
 from .transport import Transport
 data_root=Path(data_root);out_root=Path(out_root)
 tasks=json.loads((data_root/'tasks.json').read_text());manifest=json.loads((data_root/'manifest.json').read_text())
 from .panel import budget as make_budget
 b=make_budget(manifest['tasks'],manifest['repeats'])
 plan=build_plan(manifest,b)
 transport=PacedTransport(Transport(provider))
 runtime=DurableRuntime(out_root,plan,transport,run=True,approved_cny=approved_cny if approved_cny is not None else plan['max_cny'])
 records=[];budget_violations=0
 # Structural preflight before any dispatch: matched design needs three action sources per task.
 for task in tasks:
  initial,action=task_sources(task)
  if len(action)<3:raise ValueError(f"task {task['id']} lacks three action sources; structural requirement")
 for task in tasks:
  for repeat in task['repeats']:
   initial,action=task_sources(task)
   result=collect_task(runtime,task,repeat,initial,action)
   for row in result['branches']:
    answer,valid=readout_answer(row['readout'])
    records.append({'task_id':task['id'],'repeat':repeat,'branch':row['branch'],'kind':branch_kind(row['branch']),'answer':answer,'valid':valid,**matched(row['branch'])})
   print('collected',task['id'],'repeat',repeat,flush=True)
 rejected=[p.name for p in (out_root/'slots').glob('*.json')if json.loads(p.read_text()).get('status')=='rejected']
 budget_violations=len(rejected)
 slots=[]
 for p in sorted((out_root/'slots').glob('*.json')):slots.append(json.loads(p.read_text())['slot'])
 seal={'schema':'coordination-p1-panel-seal.v1','records_hash':digest(records),'expected_records':PROTOCOL['tasks']*PROTOCOL['repeats']*PROTOCOL['branches_per_repeat'],'budget_violations':budget_violations,'leakage_events':0,'collection_slots':slots}
 generation=runtime.seal(slots)
 panel_path=out_root/'panel.json'
 if panel_path.exists():raise ValueError('panel artifact exists; use a new root')
 from .live_runtime import atomic
 atomic(panel_path,{'records':records,'seal':seal,'generation':generation})
 return {'records':len(records),'budget_violations':budget_violations,'slots':len(slots),'panel':str(panel_path)}

def run_analysis(out_root,data_root):
 out_root=Path(out_root);data_root=Path(data_root)
 panel=json.loads((out_root/'panel.json').read_text())
 vault=json.loads((data_root/'private/scoring.json').read_text())
 gold={tid:v['answer']for tid,v in vault.items()}
 result=analyze(panel['records'],gold,panel['seal'])
 out=out_root/'analysis.json'
 if out.exists():raise ValueError('analysis exists; use a new root')
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 return result
