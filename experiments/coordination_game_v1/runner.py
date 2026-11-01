"""Offline executor callback contract: all checks bind one snapshot, then one readout."""
import copy
from .merge import merge
from .schema import digest

def execute(plan,state,ledger,review,readout):
    if digest(state)!=plan.state_hash:raise ValueError('stale round snapshot')
    if plan.readout_calls!=1:raise ValueError('one common readout required')
    ledger.reserve(plan.assignments,plan.readout_calls) # no partial dispatch
    reviews=[];failures=[]
    for action in plan.assignments:
        try:reviews.append(review(action,copy.deepcopy(state)))
        except Exception as exc:failures.append({'action_id':action.action_id,'error':type(exc).__name__})
    evidence=merge(reviews)
    answer=readout(copy.deepcopy(state),copy.deepcopy(evidence))
    return {'executed_action_ids':[a.action_id for a in plan.assignments],'reviews':reviews,'merged_evidence':evidence,'answer':answer,'failures':failures,'state_hash':digest(state)}
