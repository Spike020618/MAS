"""State-conditional evidence value from a frozen learned set potential.

A value model trained on empty/A/B/AB initial-state panels must not evaluate a
second action B as though A had never been observed. Preserve the reference
state and use F(S union a)-F(S). No gold or observed quality enters this score.
"""
from experiments import calibrated_action_value as v
from experiments.consensus_mechanism import core as c
from experiments.consensus_mechanism.allocation import SetAllocator

def set_value(task,origin,ids,bundle):
    ids=sorted(set(ids))
    if not ids:return 0.
    row=v.action_values(task,origin,[{'target_id':'target','source_ids':ids}],bundle)[0]
    return row['gains'][0]

def conditional_values(task,state,origin,bundle):
    rows=c.actions(task,state,allow_joint=True);seen=list(state['visible'])
    before=set_value(task,origin,seen,bundle);values=[]
    for row in rows:
        absolute=set_value(task,origin,seen+row['source_ids'],bundle)
        cost=.0005+sum(len(task['source_catalog'][s]['text'])for s in row['source_ids'])*.15/3e6
        values.append({**row,'gains':[absolute-before]*3,'cost':bundle['cost_weight']*cost,
                       'set_value_before':before,'set_value_after':absolute})
    return values

def select(task,state,origin,bundle,mode,calls,sources):
    values=conditional_values(task,state,origin,bundle)
    if not values:return [],{'reason':'no_active_action'}
    solver=SetAllocator(values,calls,sources,money=.02);allocation=solver.solve(mode)
    allocation.pop('solver_seconds',None)
    chosen=[{**values[q],'reviewer':i,'marginal':solver.utility(allocation['assignment'],i)}for i,q in enumerate(allocation['assignment'])if q is not None]
    chosen.sort(key=lambda a:(-a['marginal'],a['id'],a['reviewer']))
    return chosen,{'allocation':allocation,'all_values':values,'reference_state':origin,'observed_sources':list(state['visible'])}

def execute(task,api,bundle,mode,index):
    origin=c.fresh(task);state=c.fresh(task);events=[];traces=[];calls=2;sources=2
    while calls>0 and sources>0:
        chosen,trace=select(task,state,origin,bundle,mode,calls,sources);traces.append(trace)
        if not chosen:break
        ids=chosen[0]['source_ids'];before=__import__('copy').deepcopy(state)
        state,event=v.actor(api,f'{index}_union_{mode}_{2-calls}',task,state,ids)
        event['state_before']=before;events.append(event);calls-=1;sources-=len(ids)
    return {'decision':state['decision'],'events':events,'traces':traces,'calls':len(events),'source_reads':2-sources}
