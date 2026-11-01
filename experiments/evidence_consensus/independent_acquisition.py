"""Strong independent control: same sources and six-call ceiling, no peer review."""
import copy
from .state import initialize_decision,register_report
from .actions import fetch_source,verify_source
from .grounded_update import grounded_patch

def independent_acquisition(task,common,reports,initial_events,api,root,ledger,index,call_fn,request_fn,aggregate_fn,proposal_fn):
 events=copy.deepcopy(initial_events);updated=[];individual_states=[]
 rows=proposal_fn(task,common,reports)
 for i in range(3):
  state=copy.deepcopy(common);own=reports[i:i+1]
  state=initialize_decision(state,aggregate_fn(task,own))
  for report in own:state=register_report(state,report)
  if rows:
   row=rows[i%len(rows)];fetch=fetch_source(state,row['source_id']);verify_source(state,row['source_id'])
   req=request_fn(task,state,'review',i,own)
   event=call_fn(api,f'case{index}_independent_acquisition{i}',req,ledger,task,state);event['source_action']=fetch;event['proposal']=row
   if event['valid']:
    report=event['report'];before=copy.deepcopy(state)
    try:
     next_state=register_report(state,report);result=grounded_patch(next_state,report,required_source=row['source_id'])
     event.update(patch_accepted=result['accepted'],patch_rejected=not result['accepted'],patch_reason=result['reason'],before_decision=before['decision'],after_decision=result['state']['decision']);state=result['state']
    except (ValueError,KeyError,TypeError)as exc:event.update(patch_accepted=False,patch_rejected=True,patch_reason=str(exc))
   events.append(event)
  updated.append({'decision':state['decision']});individual_states.append(state)
 return {'decision':aggregate_fn(task,updated),'events':events,'individual_decisions':[r['decision']for r in updated],'individual_states':individual_states,'logical_calls':len(events),'budget':ledger.snapshot(),'scope':'three independently sourced reviews, no peer reports; deterministic aggregation'}
