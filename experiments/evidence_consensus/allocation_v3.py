"""Finite marginal-contribution game with hard shared budget; same central objective."""
import hashlib,itertools,math
class Allocation:
 def __init__(self,actions,cap=2,budget=.02):
  self.actions=actions;self.cap=cap;self.budget=budget;self.n=3
  if cap not in (1,2,3)or not math.isfinite(budget)or budget<0:raise ValueError('invalid budget')
  for a in actions:
   if len(a['gains'])!=3 or not a['targets']or a['cost']<0 or any(not math.isfinite(v)for v in a['gains']+[a['cost']]):raise ValueError('invalid action')
 def feasible(self,s):
  qs=[q for q in s if q is not None];units=[self.actions[q]['unit']for q in qs]
  return len(qs)<=self.cap and len(units)==len(set(units))and sum(self.actions[q]['cost']for q in qs)<=self.budget
 def potential(self,s):
  groups={};cost=0
  for i,q in enumerate(s):
   if q is None:continue
   a=self.actions[q];cost+=a['cost']
   for target in a['targets']:groups.setdefault(target,[]).append(max(0,a['gains'][i])/len(a['targets']))
  # Same-target evidence has diminishing contribution, not an assumed full substitute.
  return sum(max(v)+.5*(sum(v)-max(v))for v in groups.values())-cost
 def utility(self,s,i):
  t=list(s);t[i]=None;return self.potential(s)-self.potential(t)
 def key(self,s):return(self.potential(s),-sum(q is not None for q in s))
 def deviations(self,s,i):
  for q in [None]+list(range(len(self.actions))):
   t=list(s);t[i]=q;t=tuple(t)
   if self.feasible(t):yield t
 def solve(self):
  s=(None,)*3;trace=[]
  while True:
   changed=False
   for i in range(3):
    t=max(self.deviations(s,i),key=self.key)
    if self.key(t)>self.key(s)and(self.potential(t)>self.potential(s)+1e-10 or self.potential(t)==self.potential(s)):
     trace.append({'reviewer':i,'before':s,'after':t,'delta':self.potential(t)-self.potential(s)});s=t;changed=True
   if not changed:break
  central=max((t for t in itertools.product([None]+list(range(len(self.actions))),repeat=3)if self.feasible(t)),key=self.key)
  return {'game':s,'central':central,'game_potential':self.potential(s),'central_potential':self.potential(central),'proxy_gap':self.potential(central)-self.potential(s),'max_feasible_deviation':max([0]+[self.potential(t)-self.potential(s)for i in range(3)for t in self.deviations(s,i)]),'trace':trace,'equilibrium_kind':'generalized_Nash_shared_budget','discount_same_target':.5}
 def fixed(self):
  state=[None]*3
  for q in sorted(range(len(self.actions)),key=lambda q:hashlib.sha256(self.actions[q]['id'].encode()).hexdigest()):
   i=next((i for i,x in enumerate(state)if x is None),None)
   if i is None:break
   t=list(state);t[i]=q
   if self.feasible(t):state=t
  return tuple(state)
 def greedy(self):
  state=[None]*3
  for gain,i,q in sorted(((a['gains'][i]-a['cost'],i,q)for q,a in enumerate(self.actions)for i in range(3)),reverse=True):
   if state[i]is not None:continue
   t=list(state);t[i]=q
   if self.feasible(t):state=t
  return tuple(state)
