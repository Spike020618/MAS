"""Decision-related scope conflict proxies; no labels or expected-utility claims."""
from experiments.paired_deliberation.evidence_d2 import PREDICTIONS
from experiments.paired_deliberation.verified_state import FACT


def conflict(a,b):
    if a['claim_kind']!=b['claim_kind']:return False
    if a['claim_kind']=='current_preference':
        x,y=a['scope'],b['scope']
        norm=lambda v:' '.join(v.casefold().split()) if isinstance(v,str) else v
        return (x['time']!='unspecified' and all(norm(x[k])==norm(y[k]) for k in ('subject','predicate','object','time')) and x['polarity']!=y['polarity'])
    if a['claim_kind']=='catalog_fact':
        x,y=FACT.fullmatch(a['text']),FACT.fullmatch(b['text'])
        return bool(x and y and x.group(2)=='year' and x.groups()[:2]==y.groups()[:2] and x.group(3)!=y.group(3))
    return False # Opposite expression events can both have occurred; predictions aren't truth conflicts.


class CoordinatorD2:
    def __init__(self,mode,initial,max_rounds=3):
        if mode not in {'fixed','adaptive'} or type(max_rounds) is not int or max_rounds<0:raise ValueError('invalid coordinator')
        self.mode,self.initial,self.max_rounds=mode,initial,max_rounds
        self.processed,self.trace=set(),[]
        owners=list(initial)
        self.fixed=[o+':'+initial[o]['claims'][i]['claim_id']
                    for i in range(max((len(r['claims']) for r in initial.values()),default=0))
                    for o in owners if i<len(initial[o]['claims'])]

    def eligible(self,state):
        return sorted(k for k,row in state.claims.items() if k not in self.processed and row['own_status']!='retracted' and
            (row['claim']['claim_kind'] not in PREDICTIONS or row['status']=='basis_invalidated'))

    def features(self,target,state):
        row=state.claims[target];c=row['claim'];mids=set(c['candidate_ids'])
        ranks=[set(r['ranking']) for r in self.initial.values()]
        scope_conflict=any(other['owner']!=row['owner'] and other['status'] not in {'retracted','invalidated'} and
                           conflict(c,other['claim']) for other in state.claims.values())
        dependent=any(target in other['claim']['depends_on'] and other['status']!='retracted' for other in state.claims.values())
        on_top5=any(mids&rank for rank in ranks)
        disagreement=any(any(mid in rank for rank in ranks) and not all(mid in rank for rank in ranks) for mid in mids)
        semantic=c['claim_kind'] in {'utterance','current_preference'}
        return {'scope_conflict':int(scope_conflict),'requested_exclusion':int(c['constraint_mode']=='hard'),
                'has_dependents':int(dependent),'on_initial_top5':int(on_top5),'ranking_disagreement':int(disagreement),
                'semantic_check':int(semantic),'review_call_proxy':5 if semantic else 1}

    def relevant(self,features):
        return any(features[k] for k in ('requested_exclusion','has_dependents','on_initial_top5'))

    def table(self,state):
        result={}
        for target in self.eligible(state):
            f=self.features(target,state)
            result[target]={'features':f,'decision_relevance_proxy':self.relevant(f),
                'priority':3*f['scope_conflict']+2*f['requested_exclusion']+f['has_dependents']+
                           f['ranking_disagreement']+f['on_initial_top5']-0.1*f['review_call_proxy']}
        return result

    def selected(self,target,table,mode=None):
        self.processed.add(target)
        self.trace.append({'target':target,'mode':mode or self.mode,'candidates':table,
                           'interpretation':'scope/relevance/cost heuristic, not predicted causal decision gain'})

    def next(self,state,allowed=None):
        if len(self.trace)>=self.max_rounds:return None
        table=self.table(state)
        if allowed is not None:table={t:r for t,r in table.items() if t in allowed}
        if not table:return None
        if self.mode=='fixed':
            order={t:i for i,t in enumerate(self.fixed)}
            target=min(table,key=lambda t:(order.get(t,len(order)),t))
        else:
            relevant={t:r for t,r in table.items() if r['decision_relevance_proxy']}
            if not relevant:return None
            target=min(relevant,key=lambda t:(-relevant[t]['priority'],t))
        self.selected(target,table);return target
