"""One generic executor for strong controls, evidence games and ablations."""
import copy,json
from . import core as c
from .allocation import SetAllocator
from experiments.consensus_recommendation.common import create,fingerprint

PRIMARY=('single_strong','independent','fixed','greedy','central','game')
EXTERNAL=('self_consistency','self_refine','mad')
ABLATIONS=('no_interaction','no_dedup','no_replan','no_joint','free_rewrite')

def initialize(task,index,api,single=False):
 initial=c.initial_sources(task);common=c.fresh(task);reports=[];events=[];private=[]
 for i in range(3):
  state=common if single else c.fresh(task);c.expose(task,state,initial[:i+1]if single else initial[i:i+1]);state,event=c.invoke(api,f'{index}_{"single"if single else "shared"}_initial{i}',task,state,'initial',i,reports if single else ())
  report=event.get('report',{})if event['valid']else {};events.append(event);reports.append(report);private.append(copy.deepcopy(state))
  if single:
   common=state
   if event['valid']:common['decision']=c.full_decision(task,report['decision'])
 if single:return common,reports,events,private
 common=c.fresh(task);c.expose(task,common,initial)
 for i,(report,event)in enumerate(zip(reports,events)):
  if event['valid']:
   common,_=c.commit(task,common,report,'initial',i,event['citations'])
 common['decision']=c.aggregate(task,[r['decision']for r,e in zip(reports,events)if e['valid']])
 return common,reports,events,private

def estimate(task,state,rows,reports,bundle=None):
 values=[]
 for row in rows:
  fs=[c.features(task,state,row,reports,i)for i in range(3)]
  if bundle:
   if len(row['source_ids'])==1:gains=[float(bundle['model'].predict([f])[0]+bundle['bias'])for f in fs]
   else:
    gains=[]
    for i,f in enumerate(fs):
     parts=[c.features(task,state,{**row,'source_ids':[sid]},reports,i)for sid in row['source_ids']]
     gains.append(float(sum(bundle['model'].predict(parts))+2*bundle['bias']+bundle['interaction_model'].predict([f])[0]+bundle['interaction_bias']))
  else:gains=[.05+.15*f['retrieval_mean']+.1*f['target_title_overlap']+.05*f['own_title_overlap']for f in fs]
  # Shared predeclared forecast; actual token/CNY costs recorded separately.
  cost=.0005+sum(len(task['source_catalog'][s]['text'])for s in row['source_ids'])*.15/3e6
  values.append({**row,'gains':gains,'cost':cost,'features':fs})
 return values

def select(task,state,reports,mode,calls,sources,bundle=None,joint=True,interactions=True,deduplicate=True,money=.02):
 rows=c.actions(task,state,allow_joint=True);values=estimate(task,state,rows,reports,bundle)
 allocation=SetAllocator(values,calls,sources,money=money,interactions=interactions,deduplicate=deduplicate)
 if not joint:
  original=allocation.deviations
  def single_deviations(state,i):
   return (t for t in original(state,i)if t[i]is None or len(values[t[i]]['source_ids'])==1)
  allocation.deviations=single_deviations
  # All modes use the same restricted feasible domain.
  feasible=allocation.feasible
  allocation.feasible=lambda s:feasible(s)and all(q is None or len(values[q]['source_ids'])==1 for q in s)
 solver=allocation;allocation=solver.solve(mode)
 allocation['selected_marginals']={str(i):solver.utility(allocation['assignment'],i)for i,q in enumerate(allocation['assignment'])if q is not None}
 allocation['value_model']='shared evidence union with signed pair interaction'
 return values,allocation

def execute(task,index,method,api,baseline,reports,initial,private,budget=2,bundle=None):
 state=copy.deepcopy(baseline);events=copy.deepcopy(initial);own=copy.deepcopy(reports);allocations=[];left=budget;source_left=budget;money_left=.02
 if method=='single_strong':state,own,events,_=initialize(task,index,api,True)
 start=copy.deepcopy(state['decision']);initial_reads=list(state['visible'])
 if method=='self_consistency':
  # Classical independent sampling with disclosed temperature, native readout.
  decisions=[];events=[]
  for i in range(3+budget):
   local=c.fresh(task);pool=c.initial_sources(task)+[s for s in task['source_catalog']if s not in c.initial_sources(task)];c.expose(task,local,pool[:min(len(pool),3+budget)]);req=c.request(task,local,'initial',i,style='cot',provider_temperature=.7);event={'slot':f'{index}_sc_{i}','valid':False,'phase':'initial','labels_available':False}
   try:
    r=api.call(event['slot'],req);refs=c.validate_report(task,local,r['report'],'initial',i);event.update(valid=True,report=r['report'],usage=r['usage'],returned_model=r['model'],citations=refs);decisions.append(r['report']['decision'])
   except (ValueError,KeyError,TypeError,RuntimeError,OSError)as exc:event['error']=str(exc)
   events.append(event)
  state['decision']=c.aggregate(task,decisions)
 elif method=='self_refine':
  # Init -> feedback -> refine, repeated within the exact 3+budget call ceiling.
  state=c.fresh(task);c.expose(task,state,c.initial_sources(task));state,event=c.invoke(api,f'{index}_sr_init',task,state,'initial',0);events=[event];own=[event.get('report',{})]
  if event['valid']:state['decision']=c.full_decision(task,event['report']['decision'])
  remaining=2+budget;step=0;source_left=budget
  while remaining>=2:
   unseen=[s for s in task['source_catalog']if s not in state['visible']]
   if source_left and unseen:c.expose(task,state,unseen[:1]);source_left-=1
   _,feedback=c.invoke(api,f'{index}_sr_feedback{step}',task,state,'feedback',step,own);events.append(feedback)
   context=own+[feedback.get('report',{})];state,event=c.invoke(api,f'{index}_sr_refine{step}',task,state,'direct',10+step,context);events.append(event)
   if event['valid']:own.append(event['report'])
   remaining-=2;step+=1
  if remaining:
   state,event=c.invoke(api,f'{index}_sr_last',task,state,'direct',20,own);events.append(event)
 elif method=='mad':
  for step in range(budget):
   unseen=[s for s in task['source_catalog']if s not in state['visible']]
   if unseen:c.expose(task,state,unseen[:1])
   reviewer=step%3;state,event=c.invoke(api,f'{index}_mad_{step}',task,state,'direct',30+step,own);events.append(event)
   if event['valid']:own[reviewer]=event['report']
  # Debate final answers aggregated, not just most recent speaker.
  state['decision']=c.aggregate(task,[r['decision']for r in own if 'decision'in r])
 elif method=='independent':
  finals=[];chosen={(int(fingerprint(task['id'])[:8],16)+j)%3 for j in range(min(budget,3))}
  used=set()
  for i in range(3):
   local=copy.deepcopy(private[i]);local['decision']=c.full_decision(task,reports[i]['decision'])if initial[i]['valid']else copy.deepcopy(task['initial'])
   if i in chosen:
    rows=[r for r in c.actions(task,local,allow_joint=False)if not set(r['source_ids'])&set(c.initial_sources(task))and not set(r['source_ids'])&used]
    if rows:
     row=max(rows,key=lambda r:(task['source_catalog'][r['source_ids'][0]].get('retrieval_score_v3',0),r['id']));used.update(row['source_ids']);c.expose(task,local,row['source_ids']);local,event=c.invoke(api,f'{index}_independent_{i}',task,local,'review',i,[reports[i]],{'target_id':row['target_id'],'source_ids':row['source_ids']});events.append(event)
   finals.append(local['decision'])
  state['decision']=c.aggregate(task,finals)
 else:
  selector='central'if method=='single_strong'else method if method in ('fixed','greedy','central','game')else 'game';fixed_schedule=None
  while left>0 and source_left>0:
   if method=='no_replan'and fixed_schedule is not None:
    available=[row for row in fixed_schedule if not set(row['source_ids'])&set(state['visible'])and row['target_id']in state['claims']]
    if not available:break
    row=available[0];reviewer=row['reviewer'];allocation={'mode':'frozen_initial_schedule','rows':available}
   else:
    values,allocation=select(task,state,own,selector,left,source_left,bundle,joint=method!='no_joint',interactions=method!='no_interaction',deduplicate=method!='no_dedup',money=money_left)
    assigned=[{**values[q],'reviewer':i}for i,q in enumerate(allocation['assignment'])if q is not None]
    if not assigned:break
    # Execute the highest estimated selected action, then re-estimate the state.
    assigned.sort(key=lambda a:(-allocation['selected_marginals'][str(a['reviewer'])],a['id']));row=assigned[0];reviewer=row['reviewer']
    if method=='no_replan':fixed_schedule=assigned
   allocations.append(allocation);before=copy.deepcopy(state);fs=c.features(task,before,row,own,reviewer);c.expose(task,state,row['source_ids'])
   peers=[]if method=='no_peer'else own
   state,event=c.invoke(api,f'{index}_{method}_review{budget-left}',task,state,'review',100+budget-left,peers,{'target_id':row['target_id'],'source_ids':row['source_ids']},style='free_rewrite'if method=='free_rewrite'else 'evidence',reviewer=reviewer);event.update(action=row,pre_action_features=fs,state_before_hash=fingerprint(before));events.append(event)
   left-=1;source_left-=len(row['source_ids']);money_left-=row['cost']
   if fixed_schedule is not None:fixed_schedule=[a for a in fixed_schedule if a['id']!=row['id']]
 return {'decision':state['decision'],'initial_decision':start,'state':state,'events':events,'logical_calls':len(events),'allocations':allocations,'additional_source_reads':len(set(state['visible'])-set(initial_reads)),'semantic_certified':False}
