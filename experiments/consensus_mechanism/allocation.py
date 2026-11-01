"""Finite resource game over single and joint evidence actions; signed gains."""
import itertools,math,time
class Allocator:
 def __init__(self,rows,calls=2,sources=2,money=.02):
  self.rows=rows;self.calls=calls;self.sources=sources;self.money=money
  if type(calls)is not int or type(sources)is not int or calls<0 or sources<0 or not math.isfinite(money)or money<0:raise ValueError('invalid resource bounds')
  for r in rows:
   if len(r['gains'])!=3 or not r['source_ids']or len(r['source_ids'])!=len(set(r['source_ids']))or r['cost']<0 or any(not math.isfinite(x)for x in r['gains']+[r['cost']]):raise ValueError('invalid action estimate')
 def feasible(self,state):
  selected=[self.rows[q]for q in state if q is not None];ids=[s for r in selected for s in r['source_ids']]
  return len(selected)<=self.calls and len(ids)<=self.sources and len(ids)==len(set(ids))and sum(r['cost']for r in selected)<=self.money+1e-12
 def potential(self,state):return sum(self.rows[q]['gains'][i]-self.rows[q]['cost']for i,q in enumerate(state)if q is not None)
 def utility(self,state,i):
  other=list(state);other[i]=None;return self.potential(state)-self.potential(other)
 def key(self,state):return(self.potential(state),-sum(q is not None for q in state))
 def deviations(self,state,i):
  for q in [None]+list(range(len(self.rows))):
   other=list(state);other[i]=q;other=tuple(other)
   if self.feasible(other):yield other
 def solve(self,mode):
  started=time.perf_counter();state=(None,None,None);trace=[]
  if mode=='fixed':
   s=list(state)
   for q in sorted(range(len(self.rows)),key=lambda q:self.rows[q]['id']):
    for i in range(3):
     if s[i]is not None:continue
     t=s.copy();t[i]=q
     if self.feasible(t):s=t;break
   state=tuple(s)
  elif mode=='greedy':
   s=list(state)
   for _,i,q in sorted(((r['gains'][i]-r['cost'],i,q)for q,r in enumerate(self.rows)for i in range(3)),reverse=True):
    if s[i]is not None or self.rows[q]['gains'][i]-self.rows[q]['cost']<=0:continue
    t=s.copy();t[i]=q
    if self.feasible(t):s=t
   state=tuple(s)
  elif mode=='central':state=max((s for s in itertools.product([None]+list(range(len(self.rows))),repeat=3)if self.feasible(s)),key=self.key)
  elif mode=='game':
   while True:
    changed=False
    for i in range(3):
     t=max(self.deviations(state,i),key=self.key)
     if self.potential(t)>self.potential(state)+1e-10 or(self.potential(t)==self.potential(state)and self.key(t)>self.key(state)):
      trace.append({'reviewer':i,'before':state,'after':t,'delta':self.potential(t)-self.potential(state)});state=t;changed=True
    if not changed:break
  else:raise ValueError('unknown selection mode')
  deviation=max([0.]+[self.potential(t)-self.potential(state)for i in range(3)for t in self.deviations(state,i)])
  return {'mode':mode,'assignment':state,'potential':self.potential(state),'max_feasible_deviation':deviation,'trace':trace,'solver_seconds':time.perf_counter()-started,'calls_cap':self.calls,'sources_cap':self.sources,'money_cap':self.money,'equilibrium':'generalized_Nash_for_fixed_estimated_game'if mode=='game'else None}

class SetAllocator(Allocator):
 """Shared evidence-set value; joint predictions define signed pair interactions.

 All rows must address one fixed premise. Costs are per executed action; source
 budgets count unique material. Duplicate material never contributes value twice.
 """
 def __init__(self,rows,calls=2,sources=2,money=.02,interactions=True,deduplicate=True):
  super().__init__(rows,calls,sources,money)
  if len({r['target_id']for r in rows})>1:raise ValueError('one premise per allocation state required')
  self.interactions=interactions;self.deduplicate=deduplicate
  self.single_vectors={r['source_ids'][0]:r['gains']for r in rows if len(r['source_ids'])==1}
  self.pair_vectors={tuple(sorted(r['source_ids'])):r['gains']for r in rows if len(r['source_ids'])==2}
  self.single={r['source_ids'][0]:sum(r['gains'])/3 for r in rows if len(r['source_ids'])==1}
  self.pairs={tuple(sorted(r['source_ids'])):sum(r['gains'])/3 for r in rows if len(r['source_ids'])==2}
  if any(s not in self.single for r in rows for s in r['source_ids']):raise ValueError('missing singleton estimate')
 def feasible(self,state):
  selected=[self.rows[q]for q in state if q is not None];ids=[s for r in selected for s in r['source_ids']]
  return len(selected)<=self.calls and len(set(ids))<=self.sources and sum(r['cost']for r in selected)<=self.money+1e-12
 def potential(self,state):
  assigned=[(i,self.rows[q])for i,q in enumerate(state)if q is not None];ids=[s for i,r in assigned for s in r['source_ids']];unique=set(ids)
  value=0.
  for source in unique:
   contributions=[self.single_vectors[source][i]for i,r in assigned if source in r['source_ids']]
   value+=max(contributions)if self.deduplicate else sum(contributions)
  if self.interactions:
   for x,y in itertools.combinations(sorted(unique),2):
    pair=tuple(sorted((x,y)))
    if pair not in self.pair_vectors:continue
    owners=[i for i,r in assigned if x in r['source_ids']and y in r['source_ids']]
    hs=[self.pair_vectors[pair][i]-self.single_vectors[x][i]-self.single_vectors[y][i]for i in (owners or range(3))]
    value+=max(hs)if owners else sum(hs)/len(hs)
  return value-sum(r['cost']for i,r in assigned)
 def solve(self,mode):
  if mode!='greedy':return super().solve(mode)
  started=time.perf_counter();state=(None,None,None);trace=[]
  while True:
   candidates=[(self.potential(t)-self.potential(state),i,t)for i,q in enumerate(state)if q is None for t in self.deviations(state,i)]
   if not candidates:break
   gain,i,t=max(candidates,key=lambda x:x[0])
   if gain<=1e-10:break
   trace.append({'reviewer':i,'delta':gain});state=t
  return {'mode':mode,'assignment':state,'potential':self.potential(state),'max_feasible_deviation':max([0.]+[self.potential(t)-self.potential(state)for i in range(3)for t in self.deviations(state,i)]),'trace':trace,'solver_seconds':time.perf_counter()-started,'calls_cap':self.calls,'sources_cap':self.sources,'money_cap':self.money,'equilibrium':None}
