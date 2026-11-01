"""Descriptive paired differences with native-group bootstrap, never a promotion gate."""
import math,random
from experiments.prepare_disjoint_acquisition import group

def paired(tasks,cases,labels,method='game',reference='independent',resamples=2000):
 byid={t['id']:t for t in tasks};out={}
 for kind in ('qa','rec'):
  groups={};wins=losses=0;num=den=0
  for c in cases:
   if c['kind']!=kind:continue
   a=c['methods'][method]['decision'];b=c['methods'][reference]['decision'];gold=labels[c['task_id']]
   if kind=='qa':d=int(a==gold['answer'])-int(b==gold['answer']);weight=1
   else:d=sum(str(t)in a[:10]for t in gold['targets'])-sum(str(t)in b[:10]for t in gold['targets']);weight=len(gold['targets'])
   wins+=d>0;losses+=d<0;num+=d;den+=weight;groups.setdefault(group(byid[c['task_id']]),[]).append((d,weight))
  if not den:continue
  units=list(groups.values());rng=random.Random(20261028);samples=[]
  for _ in range(resamples):
   picked=[rng.choice(units)for _ in units];rows=[r for unit in picked for r in unit];samples.append(sum(d for d,w in rows)/sum(w for d,w in rows))
  samples.sort();n=wins+losses
  out[kind]={'difference':num/den,'denominator':den,'native_groups':len(units),'improved_tasks':wins,'harmed_tasks':losses,'cluster_bootstrap_95_percent':[samples[int(.025*resamples)],samples[min(resamples-1,int(.975*resamples))]],'resamples':resamples,'seed':20261028,'scope':'exploratory paired interval; no automatic significance claim'}
  if kind=='qa':out[kind]['exact_mcnemar_p_unadjusted']=min(1.,2*sum(math.comb(n,i)for i in range(min(wins,losses)+1))/2**n)if n else 1.
 return out
