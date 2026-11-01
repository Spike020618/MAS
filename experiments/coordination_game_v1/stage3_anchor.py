"""Full-context anchor for the stage-3 frozen cohort: one extractive read per question over
ALL supplied sources with no budget limit. It upper-bounds what any two-read source selection
can achieve and prices the cost of scarcity; labels are sealed until the generation seal."""
import argparse,json
from pathlib import Path
from .live_runtime import DurableRuntime,atomic
from .schema import digest
from .collection import message,safe_call

ROOT=Path(__file__).resolve().parents[2]

def run(data_root,out_root,provider='qwen_flash',approved_cny=0.3):
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    from .transport import Transport
    from .collect import PacedTransport
    data_root=Path(data_root);out_root=Path(out_root)
    tasks=json.loads((data_root/'tasks.json').read_text())
    plan={'schema':'coordination-stage3-anchor.v1','tasks':len(tasks),'task_ids':[t['id']for t in tasks],
          'tasks_sha256':digest([t['id']for t in tasks]),'source_split':'stage3_frozen_cohort',
          'max_requests':len(tasks),'max_cny':approved_cny,'input_byte_cap':16384,'output_token_cap':768,
          'dispatch_interval_seconds':float(__import__('os').environ.get('MAS_DISPATCH_INTERVAL','3')),
          'note':'unlimited-budget single-pass full-context extractive anchor; same questions as the sealed stage-3 matrix'}
    transport=PacedTransport(Transport(provider))
    runtime=DurableRuntime(out_root,plan,transport,run=True,approved_cny=approved_cny)
    from .stage3_run import source_map
    records=[];slots=[]
    for index,task in enumerate(tasks):
        sids=[s['id']for s in task['sources']]
        slot=f"{task['id']}/full_context";slots.append(slot)
        resp=safe_call(runtime,slot,message(task['question'],source_map(task,sids),'readout'))
        answer=None;valid=False
        if isinstance(resp,dict)and isinstance(resp.get('parsed'),dict):
            a=resp['parsed'].get('answer')
            if isinstance(a,str)and a.strip():answer=a.strip();valid=True
        records.append({'task_id':task['id'],'method':'full_context',
                        'answer':answer if answer is not None else 'noanswer','valid':valid})
        print('anchored',task['id'],flush=True)
    rejected=[p.name for p in (out_root/'slots').glob('*.json')if json.loads(p.read_text()).get('status')=='rejected']
    seal={'schema':'coordination-stage3-anchor-seal.v1','records_hash':digest(records),
          'expected_records':len(records),'budget_violations':len(rejected),'leakage_events':0,
          'collection_slots':[json.loads(p.read_text())['slot']for p in sorted((out_root/'slots').glob('*.json'))]}
    runtime.seal(slots)
    panel_path=out_root/'panel.json'
    if panel_path.exists():raise ValueError('panel artifact exists; use a new root')
    atomic(panel_path,{'records':records,'seal':seal})
    return {'records':len(records),'budget_violations':len(rejected),'panel':str(panel_path)}

def analyze(anchor_root,stage3_root,data_root):
    def f1(a,g):
        a=set(a.lower().split());g=set(g.lower().split())
        if not a and not g:return 1.0
        if not a or not g:return 0.0
        return 2*len(a&g)/(len(a)+len(g))
    vault=json.loads((Path(data_root)/'private/scoring.json').read_text())
    gold={tid:v['answer']for tid,v in vault.items()}
    anchor=json.loads((Path(anchor_root)/'panel.json').read_text())['records']
    stage3=json.loads((Path(stage3_root)/'panel.json').read_text())['records']
    af={r['task_id']:f1(r['answer'],gold[r['task_id']])for r in anchor}
    result={'schema':'coordination-stage3-anchor-analysis.v1',
            'full_context':{'n':len(af),'mean_f1':sum(af.values())/len(af),
                            'valid_fraction':sum(r['valid']for r in anchor)/len(anchor)},
            'price_of_scarcity':{},'notes':'full-context minus each budget-2 method (same questions); positive = scarcity cost'}
    for m in sorted({r['method']for r in stage3}):
        sf={r['task_id']:f1(r['answer'],gold[r['task_id']])for r in stage3 if r['method']==m}
        common=[t for t in af if t in sf]
        if common:
            diff=sum(af[t]-sf[t]for t in common)/len(common)
            result['price_of_scarcity'][m]={'mean':diff,'n':len(common)}
    out=Path(anchor_root)/'analysis.json'
    if out.exists():raise ValueError('analysis exists; use a new root')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=1))
    return result

def main():
    import argparse as ap
    p=ap.ArgumentParser();p.add_argument('stage',choices=['run','analyze'])
    p.add_argument('--root',default='experiments/results/coordination_game_s3_anchor_v1')
    p.add_argument('--data',default='experiments/results/coordination_game_s3_prepare_v1')
    p.add_argument('--stage3',default='experiments/results/coordination_game_s3_v1/collect')
    p.add_argument('--approved-cny',type=float,default=0.3)
    a=p.parse_args()
    if a.stage=='run':result=run(a.data,a.root,approved_cny=a.approved_cny)
    else:result=analyze(a.root,a.stage3,a.data)
    print(json.dumps(result,ensure_ascii=False,indent=1))

if __name__=='__main__':main()
