"""Versioned, zero-network adapter to the P0 potential-game reference implementation.

All modes use the same finite action domain: at most one action per member,
shared check-call budget plus one common readout, and an explicit stop option.
Values and interactions are caller-supplied predictions, not learned by this module.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
from . import solver,runner
from .schema import Action,RoundPlan,digest
from .utility import utility

MODES={'game':('game',False),'pair_game':('game',True),'central':('central',False),
       'greedy':('greedy',False),'block':('block',False)}

def source_hashes():
    modules=(Path(__file__),Path(solver.__file__),Path(runner.__file__),Path(__file__).with_name('utility.py'),Path(__file__).with_name('schema.py'))
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in modules}

def plan_round(state,actions,value,call_budget,mode='game',prediction_id=None):
    if mode not in MODES:raise ValueError('unsupported mode; CBBA is not implemented by this adapter')
    actions=tuple(actions)
    if any(not isinstance(a,Action)for a in actions):raise ValueError('typed Action objects required')
    implementation,pair=MODES[mode]
    before=digest(state)
    result=solver.solve(actions,value,call_budget,mode=implementation,pair=pair)
    selected=result['selected']
    if len({a.member_id for a in selected})!=len(selected):raise AssertionError('one action per member violated')
    if digest(state)!=before:raise ValueError('state mutated during planning')
    members=sorted({a.member_id for a in actions})
    utilities={m:utility(value,selected,m)for m in members}
    feasible=solver.configurations(actions,call_budget)
    residual=0.
    selected_map={a.member_id:a.action_id for a in selected}
    for candidate in feasible:
        mapping={a.member_id:a.action_id for a in candidate}
        changed=[m for m in members if selected_map.get(m)!=mapping.get(m)]
        gain=value(candidate)-value(selected)
        if len(changed)==2 and gain>0:
            gains=[utility(value,candidate,m)-utilities[m]for m in changed]
            if all(v>=0 for v in gains)and any(v>1e-10 for v in gains):residual=max(residual,gain)
    plan=RoundPlan(before,selected,result['value'])
    certificate={k:v for k,v in result.items()if k!='selected'}
    certificate.update(adapter='true-policy-adapter.v1',mode=mode,state_hash=before,
      assignments=[asdict(a)for a in selected],member_utilities=utilities,
      permitted_pair_residual=residual,central_gap=result['central_value']-result['value'],
      prediction_id=prediction_id,prediction_scope='caller-supplied frozen value function; no learned-interaction claim',
      source_hashes=source_hashes(),feasible_domain='one action per member; stop allowed; shared calls include common readout',
      compute_scope='exact offline reference; enumeration cost includes central and residual audits')
    return plan,certificate

def execute_round(state,actions,value,call_budget,ledger,review,readout,mode='game',prediction_id=None):
    plan,certificate=plan_round(state,actions,value,call_budget,mode,prediction_id)
    result=runner.execute(plan,state,ledger,review,readout)
    expected=[a.action_id for a in plan.assignments]
    if result['executed_action_ids']!=expected:raise AssertionError('configuration not executed in full')
    return {'certificate':certificate,'result':result}
