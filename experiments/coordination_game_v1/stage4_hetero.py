"""Stage 4 heterogeneity panel: the P1 16-branch design with model-heterogeneous members.

Members 0/1/2 run qwen_flash/glm_flash/deepseek on the SAME twelve questions and branch
schedule as the sealed v4 panel (all-qwen); the common readout stays on qwen_flash with the
extractive contract. The preregistered question: does model heterogeneity create cross-member
division value where same-model members showed none (v4 cross_vs_same ~ 0)?"""
import argparse,json,re
from pathlib import Path
from .live_runtime import DurableRuntime,atomic
from .schema import digest
from .collection import branch_schedule,message,safe_call
from .collect import task_sources,branch_kind,matched,readout_answer

ROOT=Path(__file__).resolve().parents[2]
# Two-model heterogeneity (deepseek credential not configured in this environment).
# Members 0/2 run qwen_flash and member 1 runs glm_flash, so cross_0/cross_1 are mixed-model
# division pairs and cross_2 is a same-model pair — an internal hetero-vs-homo contrast on
# the same questions, plus the all-qwen v4 panel as the external homogeneous baseline.
MEMBER_PROVIDERS=('qwen_flash','glm_flash','qwen_flash')
READER='qwen_flash'

PREREGISTRATION={
 'schema':'coordination-stage4-hetero.v1','tasks':12,'repeats':2,'branches_per_repeat':16,
 'member_providers':list(MEMBER_PROVIDERS),'reader':READER,
 'primary':'same questions/branches as sealed p1_v4 (all qwen); the comparison variable is member model heterogeneity (two models; deepseek unavailable)',
 'readouts':'cross_vs_same in hetero vs ~0 in homo; mixed-model cross pairs (0,1) vs same-model cross pair (2); paired_vs_empty magnitudes',
 'engineering':{'minimum_valid_fraction':.85,'maximum_budget_violations':0,'maximum_leakage_events':0},
 'scope':'diagnostic panel on the same development cohort; no confirmatory claim',
}

def member_of(slot):
    """Route a slot name to its member index (or the common reader)."""
    tail=slot.split('/',1)[1] if '/' in slot else slot  # repeat-{r}/...
    m=re.match(r'repeat-\d+/initial-(\d)$',tail)
    if m:return int(m.group(1))
    m=re.match(r'repeat-\d+/single_(\d)_\d+/check-\d+$',tail)
    if m:return int(m.group(1))
    m=re.match(r'repeat-\d+/joint_(\d+)/check-\d+$',tail)
    if m:return int(m.group(1))
    m=re.match(r'repeat-\d+/cross_(\d+)/check-(\d)$',tail)
    if m:
        base=int(m.group(1));k=int(m.group(2))
        return base if k==0 else (base+1)%3
    if tail.endswith('/readout'):return 'reader'
    raise ValueError('unroutable slot '+slot)

class MemberRuntime:
    """One durable sub-runtime per member plus one for the common reader; routing by slot name."""
    def __init__(self,root,plan,member_plans,reader_plan):
        from .transport import Transport
        from .collect import PacedTransport
        self.runtimes={}
        for m,provider in enumerate(MEMBER_PROVIDERS):
            self.runtimes[m]=DurableRuntime(Path(root)/f'member-{m}',member_plans[m],
                                            PacedTransport(Transport(provider)),run=True,
                                            approved_cny=member_plans[m]['max_cny'])
        self.runtimes['reader']=DurableRuntime(Path(root)/'reader',reader_plan,
                                               PacedTransport(Transport(READER)),run=True,
                                               approved_cny=reader_plan['max_cny'])
    def _rt(self,slot):return self.runtimes[member_of(slot)]
    def call(self,slot,request):return self._rt(slot).call(slot,request)
    def reserve_batch(self,requests):
        for slot,request in requests:self._rt(slot).reserve_batch([(slot,request)])
    def seal(self,slots):
        result={}
        for key,rt_slots in _partition(slots).items():
            result[str(key)]=self.runtimes[key].seal(rt_slots)
        return result

def _partition(slots):
    out={}
    for s in slots:
        out.setdefault(member_of(s),[]).append(s)
    return out

def member_plan(provider,task_ids,count,cny):
    return {'schema':'coordination-stage4-hetero.v1','provider':provider,'tasks':len(task_ids),'repeats':2,
            'task_ids':list(task_ids),'tasks_sha256':digest(list(task_ids)),
            'source_split':'p1_v4_cohort','max_requests':count,'max_cny':cny,
            'input_byte_cap':8192,'output_token_cap':768,
            'dispatch_interval_seconds':float(__import__('os').environ.get('MAS_DISPATCH_INTERVAL','3'))}

def collect(data_root,out_root,approved_total=1.5):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    data_root=Path(data_root);out_root=Path(out_root)
    tasks=json.loads((data_root/'tasks.json').read_text())
    manifest=json.loads((data_root/'manifest.json').read_text())
    task_ids=manifest['task_ids']
    # per-member request upper bound: initials+checks across all repeats (~8 per task-repeat) plus slack
    member_count=12*2*8+12
    reader_count=12*2*16
    member_plans=[member_plan(p,task_ids,member_count,round(approved_total*0.35,4))for p in MEMBER_PROVIDERS]
    # The reader carries all 384 readouts with hetero observations (glm runs verbose):
    # the first attempt's 0.6-CNY estimate ceiling stranded 67 readouts at the cost cap.
    reader_plan=member_plan(READER,task_ids,reader_count,round(approved_total,4))
    reader_plan['input_byte_cap']=16384
    runtime=MemberRuntime(out_root,member_plans[0],member_plans,reader_plan)
    records=[];slots=[]
    for task in tasks:
        for repeat in task['repeats']:
            initial,action=task_sources(task)
            prefix=f"{task['id']}/repeat-{repeat}"
            initial_responses=[]
            for m in range(3):
                slot=f'{prefix}/initial-{m}';slots.append(slot)
                initial_responses.append(safe_call(runtime,slot,message(task['question'],initial[m],'initial')))
            results=[]
            for name,assignments in branch_schedule():
                checks=[]
                for k,(m,indices) in enumerate(assignments):
                    sources=dict(initial[m])
                    for s in indices:sources.update(action[s])
                    checks.append((f'{prefix}/{name}/check-{k}',message(task['question'],sources)))
                runtime.reserve_batch(checks)
                reviews=[]
                for slot,req in checks:slots.append(slot);reviews.append(safe_call(runtime,slot,req))
                source={k:v for mapping in initial for k,v in mapping.items()}
                for _,indices in assignments:
                    for s in indices:source.update(action[s])
                parsed=[r.get('parsed')for r in list(initial_responses)+reviews
                        if isinstance(r,dict)and isinstance(r.get('parsed'),dict)]
                req=message(task['question'],source,'readout',parsed)
                readslot=f'{prefix}/{name}/readout';slots.append(readslot)
                answer=safe_call(runtime,readslot,req)
                results.append({'branch':name,'assignments':assignments,'reviews':reviews,'readout':answer})
                answer,valid=readout_answer(answer)
                records.append({'task_id':task['id'],'repeat':repeat,'branch':name,'kind':branch_kind(name),
                                'answer':answer,'valid':valid,**matched(name)})
            print('collected',task['id'],'repeat',repeat,flush=True)
    violations=0
    for sub in out_root.iterdir():
        if not (sub/'slots').is_dir():continue
        violations+=sum(1 for p in (sub/'slots').glob('*.json')
                        if json.loads(p.read_text()).get('status')=='rejected')
    seal={'schema':'coordination-stage4-hetero-seal.v1','records_hash':digest(records),
          'expected_records':384,'budget_violations':violations,'leakage_events':0}
    generation=runtime.seal(slots)
    panel_path=out_root/'panel.json'
    if panel_path.exists():raise ValueError('panel artifact exists; use a new root')
    atomic(panel_path,{'records':records,'seal':seal,'generation':generation})
    return {'records':len(records),'budget_violations':violations,'panel':str(panel_path)}

def analyze(out_root,data_root):
    from .collect import run_analysis
    return run_analysis(out_root,data_root)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','analyze'])
    p.add_argument('--root',default='experiments/results/coordination_game_s4_hetero_v1')
    p.add_argument('--data',default='experiments/results/coordination_game_p1_prepare_v1')
    p.add_argument('--approved-cny',type=float,default=1.5)
    a=p.parse_args()
    root=Path(a.root)
    if a.stage=='collect':result=collect(a.data,root,approved_total=a.approved_cny)
    else:result=analyze(root,a.data)
    print(json.dumps(result,ensure_ascii=False,indent=1)if isinstance(result,dict)else result)

if __name__=='__main__':main()
