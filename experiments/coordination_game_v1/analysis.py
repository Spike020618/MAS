"""Sealed normalized panel analysis. No gold is consulted for incomplete collection.

Records: task_id, repeat, branch, kind (empty/single/same/cross), answer,
valid, plus optional matched_single_branches and matched_same_branch.
Seal: records_hash, expected_records, budget_violations, leakage_events.
The collection adapter must validate its own file seals before normalization.
"""
from collections import defaultdict
import random,statistics
from .schema import digest
from .evaluate import answer_score
from .preregistration import PROTOCOL

def mean(xs):return sum(xs)/len(xs)if xs else None

def paired_interval(values):
    if not values:return {'n_questions':0,'mean':None,'ci95':None}
    rng=random.Random(PROTOCOL['bootstrap_seed']);n=len(values)
    draws=sorted(mean([values[rng.randrange(n)]for _ in range(n)])for _ in range(PROTOCOL['bootstrap_draws']))
    return {'n_questions':n,'mean':mean(values),'ci95':[draws[int(.025*len(draws))],draws[int(.975*len(draws))]],'scope':'exploratory question-cluster bootstrap; not evidence of solver advantage'}

def analyze(records,gold=None,seal=None):
    expected=PROTOCOL['tasks']*PROTOCOL['repeats']*PROTOCOL['branches_per_repeat']
    status={'records':len(records),'expected_records':expected,'status':'incomplete','effects_available':False}
    if not seal:return status
    if seal.get('records_hash')!=digest(records):raise ValueError('panel seal hash mismatch')
    if seal.get('expected_records')!=expected or len(records)!=expected:return status
    keys=[(r['task_id'],r['repeat'],r['branch'])for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate panel identity')
    tasks={r['task_id']for r in records}
    if len(tasks)!=PROTOCOL['tasks']:raise ValueError('unexpected question count')
    for task in tasks:
        branches=[]
        for repeat in range(PROTOCOL['repeats']):
            subset=[r for r in records if r['task_id']==task and r['repeat']==repeat]
            counts={k:sum(r['kind']==k for r in subset)for k in ('empty','single','same','cross')}
            if counts!={'empty':1,'single':9,'same':3,'cross':3}:raise ValueError('incomplete matched branch design')
            branches.append({r['branch']for r in subset})
        if branches[0]!=branches[1]:raise ValueError('repeat branch mismatch')
    if gold is None or not tasks<=set(gold):raise ValueError('private gold missing after seal')
    scored={k:{**r,**answer_score(r['answer'],gold[r['task_id']])}for k,r in zip(keys,records)}
    grouped=defaultdict(list);deltas=defaultdict(lambda:defaultdict(list));repeat_signs=defaultdict(list);interaction=[];cross_same=[]
    for k,r in scored.items():
        empty=next(v for kk,v in scored.items()if kk[:2]==k[:2]and v['kind']=='empty')
        delta=r['f1']-empty['f1'];grouped[r['kind']].append({**r,'delta':delta})
        if r['kind']!='empty':
            deltas[r['kind']][r['task_id']].append(delta)
            repeat_signs[(r['task_id'],r['branch'])].append((r['repeat'],delta))
        if r.get('matched_single_branches'):
            a,b=[scored[(r['task_id'],r['repeat'],name)]for name in r['matched_single_branches']]
            interaction.append({'task_id':r['task_id'],'repeat':r['repeat'],'branch':r['branch'],'value':r['f1']-a['f1']-b['f1']+empty['f1']})
        if r.get('matched_same_branch'):
            same=scored[(r['task_id'],r['repeat'],r['matched_same_branch'])]
            cross_same.append({'task_id':r['task_id'],'repeat':r['repeat'],'value':r['f1']-same['f1']})
    metrics={kind:{'n':len(rows),'f1':mean([r['f1']for r in rows]),'em':mean([r['em']for r in rows]),'corrected':sum(r['delta']>0 for r in rows),'harmed':sum(r['delta']<0 for r in rows),'invalid':sum(not r['valid']for r in rows)}for kind,rows in grouped.items()}
    valid=sum(bool(r['valid'])for r in records)/len(records)
    gate=valid>=.95 and seal.get('budget_violations')==0 and seal.get('leakage_events')==0
    consistency={'matched_actions':len(repeat_signs),'positive_both':sum(all(d>0 for _,d in rows)for rows in repeat_signs.values()),'negative_both':sum(all(d<0 for _,d in rows)for rows in repeat_signs.values()),'sign_disagreement':sum(len({(d>0)-(d<0)for _,d in rows})>1 for rows in repeat_signs.values())}
    return {**status,'status':'sealed','effects_available':True,'metrics':metrics,'valid_fraction':valid,'engineering_gate':gate,
            'paired_vs_empty':{k:paired_interval([mean(v)for v in bytask.values()])for k,bytask in deltas.items()},
            'repeat_consistency':consistency,'interaction':interaction,'cross_vs_same':cross_same,
            'science_gate':'review_required','member_state_effect':'requires matched source/member comparison; initial disagreement alone is insufficient',
            'estimator_gain':'not_measured','solver_gain':'not_measured','protocol_hash':digest(PROTOCOL)}
