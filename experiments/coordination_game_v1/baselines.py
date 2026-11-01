from .solver import solve

def fixed(actions,call_budget):
    selected=[];members=set();remaining=call_budget-1
    if remaining<0:raise ValueError('readout not affordable')
    for a in actions:
        if a.member_id not in members and a.calls<=remaining:selected.append(a);members.add(a.member_id);remaining-=a.calls
    return tuple(selected)

# CBBA needs its own synchronous-message conflict-resolution validation in P2.
# A centralized greedy function is deliberately not relabeled as CBBA.
BASELINE_STATUS={'fixed':'implemented','greedy':'offline_reference','central':'exact_reference','block':'offline_reference','single_agent':'P1 panel contract','independent':'P1 panel contract','CBBA':'not_implemented_P2_gate'}
