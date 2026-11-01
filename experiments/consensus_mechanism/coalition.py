"""Evidence-set game: overlapping evidence is valued once; interactions are explicit.

Values are predictions fitted without evaluation labels. No truth claim is made by
an equilibrium. Central optimization is a control over the identical feasible set.
"""
import itertools
import math

class CoalitionGame:
 def __init__(self, actions, singleton, interactions=None, budget=2, penalty=0.):
  self.actions=[tuple(a) for a in actions]; self.singleton=dict(singleton)
  self.interactions=dict(interactions or {}); self.budget=budget; self.penalty=penalty
  if budget<0 or penalty<0 or any(not math.isfinite(v) for v in list(self.singleton.values())+list(self.interactions.values())+[penalty]):raise ValueError('invalid game parameters')
 def evidence(self,s):return frozenset(x for q in s if q is not None for x in self.actions[q])
 def feasible(self,s):return len(self.evidence(s))<=self.budget
 def potential(self,s):
  e=self.evidence(s)
  return sum(self.singleton.get(x,0.) for x in e)+sum(v for pair,v in self.interactions.items() if set(pair)<=e)-self.penalty*sum(q is not None for q in s)
 def utility(self,s,i):
  t=list(s);t[i]=None
  return self.potential(s)-self.potential(t)
 def solve(self,players=3,central=False):
  options=[None]+list(range(len(self.actions))); s=(None,)*players;trace=[]
  if central:
   s=max((t for t in itertools.product(options,repeat=players) if self.feasible(t)),key=lambda t:(self.potential(t),-sum(q is not None for q in t)))
  else:
   while True:
    moved=False
    for i in range(players):
     candidates=[]
     for q in options:
      t=list(s);t[i]=q;t=tuple(t)
      if self.feasible(t):candidates.append(t)
     t=max(candidates,key=self.potential)
     if self.potential(t)>self.potential(s)+1e-10:
      trace.append({'player':i,'gain':self.potential(t)-self.potential(s)});s=t;moved=True
    if not moved:break
  deviation=0.
  for i in range(players):
   for q in options:
    t=list(s);t[i]=q
    if self.feasible(t):deviation=max(deviation,self.potential(t)-self.potential(s))
  return {'assignment':s,'potential':self.potential(s),'deviation':deviation,'trace':trace,'central':central}
