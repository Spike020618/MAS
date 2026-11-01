"""Budgeted evidence-action coverage game; no invented reviewer skill.

net_gain is already expected net task reward, including damage. Do not subtract
an additional damage probability unless a separately defined objective needs it.
"""
import itertools,hashlib,math

class ActionAllocation:
 def __init__(self,actions,reviewers=3,max_actions=3,budget=math.inf,cost_weight=1.):
  self.actions=actions;self.n=reviewers;self.cap=max_actions;self.budget=budget;self.weight=cost_weight
  if type(reviewers)is not int or type(max_actions)is not int or reviewers<1 or max_actions<0 or math.isnan(budget) or budget<0 or not math.isfinite(cost_weight) or cost_weight<0:raise ValueError('invalid resource bounds')
  ids=[a['action_id']for a in actions]
  if len(ids)!=len(set(ids)):raise ValueError('duplicate action IDs')
  for a in actions:
   if not a.get('source_unit') or not a.get('doubt_id') or not a.get('target_claim_ids'):raise ValueError('source/doubt/targets required')
   for x in [a.get('net_gain',0),a.get('cost',0),*a.get('gain_by_reviewer',[])]:
    if not math.isfinite(x):raise ValueError('nonfinite estimate')
   if a.get('cost',0)<0:raise ValueError('negative cost')
   if 'gain_by_reviewer'in a and len(a['gain_by_reviewer'])!=reviewers:raise ValueError('reviewer gain shape')
 def gain(self,i,q):return self.actions[q].get('gain_by_reviewer',[self.actions[q].get('net_gain',0)]*self.n)[i]
 def cost(self,s):return sum(self.actions[q].get('cost',0)for q in s if q is not None)
 def feasible(self,s):return sum(q is not None for q in s)<=self.cap and self.cost(s)<=self.budget
 def potential(self,s):
  units={}
  for i,q in enumerate(s):
   if q is None:continue
   a=self.actions[q]
   # Repeated source/target contribution receives max credit, not repeated sum.
   key=(a['source_unit'],tuple(sorted(a['target_claim_ids'])))
   units[key]=max(units.get(key,0),self.gain(i,q))
  return sum(units.values())-self.weight*self.cost(s)
 def utility(self,s,i):
  t=list(s);t[i]=None;return self.potential(s)-self.potential(t)
 def states(self):return (s for s in itertools.product([None]+list(range(len(self.actions))),repeat=self.n)if self.feasible(s))
 def key(self,s):return (self.potential(s),-sum(q is not None for q in s))
 def deviations(self,s,i):
  for q in [None]+list(range(len(self.actions))):
   t=list(s);t[i]=q;t=tuple(t)
   if self.feasible(t):yield t
 def solve(self):
  s=(None,)*self.n;trace=[]
  while True:
   changed=False
   for i in range(self.n):
    t=max(self.deviations(s,i),key=self.key)
    if self.potential(t)>self.potential(s)+1e-10 or (self.potential(t)==self.potential(s)and self.key(t)>self.key(s)):
     trace.append({'player':i,'before':s,'after':t,'delta':self.potential(t)-self.potential(s)});s=t;changed=True
   if not changed:break
  central=max(self.states(),key=self.key)
  fixed=[None]*self.n;used=set()
  for q in sorted(range(len(self.actions)),key=lambda q:hashlib.sha256(self.actions[q]['action_id'].encode()).hexdigest()):
   unit=(self.actions[q]['source_unit'],tuple(sorted(self.actions[q]['target_claim_ids'])))
   if unit in used:continue
   i=next((i for i,v in enumerate(fixed)if v is None),None)
   if i is None:break
   t=list(fixed);t[i]=q
   if self.feasible(t):fixed=t;used.add(unit)
  fixed=tuple(fixed)
  overlap=lambda a,b:len({q for q in a if q is not None}&{q for q in b if q is not None})
  return {'game':s,'central':central,'fixed':fixed,'game_potential':self.potential(s),'central_potential':self.potential(central),'proxy_gap':self.potential(central)-self.potential(s),'max_feasible_deviation':max([0]+[self.potential(t)-self.potential(s)for i in range(self.n)for t in self.deviations(s,i)]),'identical_allocations':{'game_central':s==central,'game_fixed':s==fixed},'action_overlap':{'game_central':overlap(s,central),'game_fixed':overlap(s,fixed)},'equilibrium_kind':'generalized_nash_under_shared_constraints','trace':trace,'scope':'estimated coverage contribution; no guarantee of task improvement; central is same-objective global comparator'}
