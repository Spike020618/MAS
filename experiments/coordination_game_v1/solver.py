"""Exact finite offline reference solver; no distributed-cost claim."""
import itertools,math
from .utility import utility

def configurations(actions,call_budget,readout_calls=1):
    if len({a.action_id for a in actions})!=len(actions):raise ValueError("duplicate action identity")
    groups={}
    for a in actions:
        if a.calls<1:raise ValueError('positive action cost required')
        groups.setdefault(a.member_id,[]).append(a)
    if call_budget<readout_calls:raise ValueError('cannot afford common readout')
    return [tuple(a for a in state if a is not None)for state in itertools.product(*[[None]+groups[m]for m in sorted(groups)])if sum(a.calls for a in state if a is not None)+readout_calls<=call_budget]

def solve(actions,value,call_budget,mode='game',pair=False,tolerance=1e-10):
    if mode not in ('game','central','greedy','block'):raise ValueError('unknown solver')
    if tolerance<0:raise ValueError('negative tolerance')
    feasible=configurations(actions,call_budget);members=sorted({a.member_id for a in actions});memo={c:value(c)for c in feasible}
    if any(not math.isfinite(v)for v in memo.values()):raise ValueError('nonfinite objective')
    state=();trace=[];blocked=0
    def changed(a,b):
        aa={x.member_id:x for x in a};bb={x.member_id:x for x in b}
        return [m for m in members if aa.get(m)!=bb.get(m)]
    if mode=='central':state=max(feasible,key=lambda c:memo[c])
    else:
        while True:
            choices=[]
            for candidate in feasible:
                delta=memo[candidate]-memo[state];actors=changed(state,candidate)
                if delta<=tolerance or not actors:continue
                if mode=='greedy'and not(set(state)<=set(candidate)and len(candidate)==len(state)+1):continue
                if mode!='greedy'and len(actors)>(2 if pair or mode=='block'else 1):continue
                gains={m:utility(value,candidate,m)-utility(value,state,m)for m in actors}
                if mode=='game'and len(actors)==2 and not(all(v>=0 for v in gains.values())and any(v>tolerance for v in gains.values())):
                    blocked+=1;continue
                choices.append((memo[candidate],candidate,gains))
            if not choices:break
            _,next_state,gains=max(choices,key=lambda row:row[0]);trace.append({'before':[a.action_id for a in state],'after':[a.action_id for a in next_state],'potential_gain':memo[next_state]-memo[state],'member_gains':gains});state=next_state
    unary=max([0]+[memo[c]-memo[state]for c in feasible if len(changed(state,c))==1])
    return {'selected':state,'value':memo[state],'central_value':max(memo.values()),'unilateral_residual':unary,'steps':trace,'blocked_pair_proposals':blocked,'objective_evaluations':len(memo),'scope':'exhaustive offline reference; not a communication-efficient distributed implementation'}
