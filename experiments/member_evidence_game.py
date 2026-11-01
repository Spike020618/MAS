"""Finite potential game for decision VOI with a shared acquisition budget.

Optional pair exchanges require positive potential gain AND transferable surplus.
Transfers are internal expected-quality credits, never money or provider tokens.
"""
import itertools,json,time
import numpy as np

class EvidenceGame:
    def __init__(self,prior,candidates,channel,cost,calls=2,sources=2,players=3,used=None):
        self.prior=np.asarray(prior,dtype=float);self.players=players;self.calls=calls;self.sources=sources
        if calls not in (0,1,2) or sources<0 or players<1 or self.prior.shape!=(3,) or np.any(self.prior<0) or not np.isclose(self.prior.sum(),1):raise ValueError('invalid game bounds or belief')
        self.options=[None,()]+[tuple(x)for n in range(1,min(2,sources)+1)for x in itertools.combinations(sorted(candidates),n)]
        self.channel_fn=channel;self.cost_fn=cost;self.used=used or set();self.channels={};self.values={};self.evaluations=0
    def active(self,s):return [(i,self.options[q])for i,q in enumerate(s)if q!=0]
    def feasible(self,s):
        active=self.active(s);ids=[sid for _,a in active for sid in a]
        return len(active)<=self.calls and len(ids)<=self.sources and len(ids)==len(set(ids)) and all((i,tuple(a))not in self.used for i,a in active)
    def channel(self,i,ids):
        key=(i,tuple(ids))
        if key not in self.channels:
            q=np.asarray(self.channel_fn(i,list(ids)),dtype=float)
            if q.shape!=(3,4)or not np.allclose(q.sum(axis=1),1)or np.any(q<0):raise ValueError('normalized response channel required')
            self.channels[key]=q
        return self.channels[key]
    def potential(self,s):
        s=tuple(s)
        if s in self.values:return self.values[s]
        active=self.active(s)
        if not active:value=0.
        elif len(active)==1:
            i,a=active[0];value=float(np.max(self.prior[:,None]*self.channel(i,a),axis=0).sum()-self.prior.max())
        elif len(active)==2:
            (i,a),(j,b)=active
            joint=self.prior[:,None,None]*self.channel(i,a)[:,:,None]*self.channel(j,b)[:,None,:]
            value=float(np.max(joint,axis=0).sum()-self.prior.max())
        else:raise ValueError('this release supports at most two active checks')
        value-=sum(self.cost_fn(i,list(a))for i,a in active)
        self.values[s]=value;self.evaluations+=1;return value
    def utility(self,s,i):
        absent=list(s);absent[i]=0
        return self.potential(s)-self.potential(tuple(absent))
    def deviations(self,s,i):
        for q in range(len(self.options)):
            t=list(s);t[i]=q;t=tuple(t)
            if self.feasible(t):yield t
    def key(self,s):return self.potential(s),-len(self.active(s))
    def unary(self,state,trace):
        while True:
            moved=False
            for i in range(self.players):
                t=max(self.deviations(state,i),key=self.key)
                delta=self.potential(t)-self.potential(state)
                if delta>1e-10:
                    trace.append({'kind':'unilateral','member':i,'before':state,'after':t,'potential_gain':delta})
                    state=t;moved=True
            if not moved:return state
    def pair_candidates(self,s):
        for i,j in itertools.combinations(range(self.players),2):
            before=self.utility(s,i)+self.utility(s,j)
            for a,b in itertools.product(range(len(self.options)),repeat=2):
                t=list(s);t[i]=a;t[j]=b;t=tuple(t)
                if not self.feasible(t):continue
                delta=self.potential(t)-self.potential(s)
                if delta<=1e-10:continue
                surplus=self.utility(t,i)+self.utility(t,j)-before
                if surplus>1e-10:yield delta,surplus,i,j,t
    def solve(self,mode):
        started=time.perf_counter();state=(0,)*self.players;trace=[];credits=[0.]*self.players
        if mode in ('game','exchange'):
            state=self.unary(state,trace)
            if mode=='exchange':
                while True:
                    moves=list(self.pair_candidates(state))
                    if not moves:break
                    delta,surplus,i,j,t=max(moves,key=lambda m:m[0])
                    du={k:self.utility(t,k)-self.utility(state,k)for k in (i,j)}
                    transfer={k:surplus/2-du[k]for k in (i,j)}
                    for k in (i,j):credits[k]+=transfer[k]
                    trace.append({'kind':'pair_exchange','members':[i,j],'before':state,'after':t,
                       'potential_gain':delta,'raw_surplus':surplus,'transfers':transfer,
                       'individual_net_gains':{k:du[k]+transfer[k]for k in (i,j)}})
                    state=self.unary(t,trace)
        elif mode=='greedy':
            while True:
                rows=[t for i,q in enumerate(state)if q==0 for t in self.deviations(state,i)]
                if not rows:break
                best=max(rows,key=self.key)
                if self.potential(best)<=self.potential(state)+1e-10:break
                state=best
        elif mode=='central':
            feasible=[state]
            for n in range(1,min(2,self.calls,self.players)+1):
                for members in itertools.combinations(range(self.players),n):
                    for choices in itertools.product(range(1,len(self.options)),repeat=n):
                        t=[0]*self.players
                        for i,q in zip(members,choices):t[i]=q
                        t=tuple(t)
                        if self.feasible(t):feasible.append(t)
            state=max(feasible,key=self.key)
        else:raise ValueError('unknown game mode')
        deviation=max([0.]+[self.potential(t)-self.potential(state)for i in range(self.players)for t in self.deviations(state,i)])
        pair_deviation=max([0.]+[m[0]for m in self.pair_candidates(state)])if mode=='exchange'else None
        return {'mode':mode,'assignment':list(state),'selected':[{'member':i,'source_ids':list(a)}for i,a in self.active(state)],
          'potential':self.potential(state),'max_unilateral_deviation':deviation,'max_permitted_pair_deviation':pair_deviation,
          'trace':trace,'credits':credits,'objective_evaluations':self.evaluations,'solver_seconds':time.perf_counter()-started,
          'scope':'fixed estimated response game; pair stability only for positive-potential transferable-surplus moves'}
