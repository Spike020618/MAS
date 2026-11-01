"""DurableRuntime bridge for a real fixed-state round; no transport instantiated here.
A whole-round cost admission check precedes dispatch. The readout reservation uses
a conservative whole-round envelope, never a fake request in receipt slots.
"""
import json,fcntl,hashlib
from pathlib import Path
from .true_policy_adapter import plan_round
from .collection import message
from .schema import digest,Review
from .merge import merge
from .live_runtime import UnknownRequest

def run_live_round(runtime,task,members,public_source_ids,actions,value,prediction_id,mode='game',call_budget=3):
    # This bridge serializes complete rounds; parallel writes through another API to
    # the same runtime root remain unsupported and must not be used.
    with runtime.lock:
        with (runtime.root/'true_round.lock').open('a') as owner:
            try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('runtime root already owns an active round') from None
            try:return _run_live_round(runtime,task,members,public_source_ids,actions,value,prediction_id,mode,call_budget)
            finally:fcntl.flock(owner,fcntl.LOCK_UN)

def _run_live_round(runtime,task,members,public_source_ids,actions,value,prediction_id,mode='game',call_budget=3):
    actions=tuple(actions)
    if any(a.calls!=1 for a in actions):raise ValueError('this bridge supports one model request per action')
    if not prediction_id:raise ValueError('explicit frozen prediction provenance required')
    catalog={s['id']:f"{s['title']}: {' '.join(s['sentences'])}"for s in task['sources']}
    member_map={m.member_id:m for m in members}
    state={'task_id':task['id'],'question':task['question'],'members':[{'id':m.member_id,'sources':list(m.observed_sources),'history_hash':m.history_hash}for m in members], 'public_sources':sorted(public_source_ids)}
    plan,certificate=plan_round(state,actions,value,call_budget,mode,prediction_id)
    requests=[];union=set(public_source_ids)
    for m in members:union.update(m.observed_sources)
    prefix=task['id']+'/true-round/'+digest([state,prediction_id,mode])
    for action in plan.assignments:
        if action.member_id not in member_map:raise ValueError('unknown member')
        visible=set(member_map[action.member_id].observed_sources)|set(public_source_ids)|set(action.source_ids)
        if not visible<=set(catalog):raise ValueError('unknown source')
        union.update(action.source_ids)
        req=message(task['question'],{sid:catalog[sid]for sid in sorted(visible)},'review')
        requests.append((prefix+'/'+action.action_id,req,action,visible))
    if not union<=set(catalog):raise ValueError('unknown public source')
    # Worst-case readout admission: the actual prompt is checked below as well.
    readout_cap=runtime.plan.get('input_byte_cap',4096)
    output_cap=min(768,runtime.plan.get('output_token_cap',768))
    readout_reserve=(readout_cap*.15+output_cap*1.5)/1e6
    existing=[json.loads(p.read_text())for p in runtime.slots.glob('*.json')]
    additions=[(slot,req)for slot,req,_,_ in requests if not runtime.path(slot).exists()]
    final_slot=prefix+'/readout'
    final_exists=runtime.path(final_slot).exists()
    if len(existing)+len(additions)+(not final_exists)>runtime.plan['max_requests']:raise ValueError('whole round request cap exceeded')
    cost=sum(r['reserved_cny']for r in existing)+sum(runtime.estimate(req)for _,req in additions)+(0 if final_exists else readout_reserve)
    if cost>min(runtime.approved_cny,runtime.plan['max_cny']):raise ValueError('whole round cost admission failed')
    # Fail before any dispatch when resuming an unresolved request.
    for slot,req,_,_ in requests:
        if runtime.path(slot).exists():
            receipt=json.loads(runtime.path(slot).read_text())
            if receipt['request_hash']!=digest(req):raise ValueError('request mismatch')
            if receipt['status']=='intent':raise UnknownRequest('unknown review completion; no readout dispatched')
    if not runtime.run:raise RuntimeError('explicit runtime.run required; no network dispatched')
    runtime.reserve_batch([(slot,req)for slot,req,_,_ in requests])
    reviews=[];raw=[]
    for slot,req,action,visible in requests:
        response=runtime.call(slot,req) # UnknownRequest propagates; never swallowed.
        if response is None:raise RuntimeError('rejected review; no readout dispatched')
        parsed=response.get('parsed');raw.append(response)
        obs=parsed.get('observations')if isinstance(parsed,dict)else None
        cites=parsed.get('citation_ids')if isinstance(parsed,dict)else None
        valid=isinstance(obs,list)and isinstance(cites,list)and all(isinstance(x,str)for x in obs+cites)and set(cites)<=visible and(not obs or bool(cites))
        reviews.append(Review(action.action_id,tuple(obs)if valid else(),tuple(cites)if valid else(),valid,''if valid else'invalid_observation_or_source'))
    evidence=merge(reviews)
    req=message(task['question'],{sid:catalog[sid]for sid in sorted(union)},'readout',evidence)
    req['max_tokens']=output_cap
    runtime.estimate(req) # actual content can fail admission; no truncation or fake hash.
    response=runtime.call(final_slot,req)
    slots=[s for s,_,_,_ in requests]+[final_slot]
    return {'certificate':certificate,'reviews':[r.__dict__ for r in reviews],'merged_evidence':evidence,'readout':response,'slots':slots,'executed_action_ids':[a.action_id for a in plan.assignments],'round_state_hash':digest(state),'bridge_source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'admission_scope':'serialized single-owner runtime required; conservative readout envelope preflight, actual receipt reserved only after observations exist'}
