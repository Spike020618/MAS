"""Zero-call post-hoc audit; training-only readout fitting, no game claims."""
from collections import Counter
from math import log
import json
from experiments.mechanism import run_learned_production as source
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.core.mechanism.learned_production_game import majority

SLOTS=('fast0','fast1','fast2','think1','think2')
OUT=source.OUT.parent/'information_readout_gap_v1'


def unique_max(scores):
    if not scores:return 'abstain'
    ordered=sorted(scores,key=lambda k:-scores[k])
    return ordered[0] if len(ordered)==1 or scores[ordered[0]]-scores[ordered[1]]>1e-10 else 'abstain'


def fit(train,labels):
    n=len(train)
    correct={slot:sum(s[slot]['answer']==labels[c] for c,s in train.items()) for slot in SLOTS}
    costs={slot:sum(s[slot]['tokens'] for s in train.values()) for slot in SLOTS}
    chosen=max(SLOTS,key=lambda slot:(correct[slot],-costs[slot],-SLOTS.index(slot)))
    # Laplace smoothing; log-odds voting is a simple baseline, NOT calibrated
    # evidence or independent observations (reports can be correlated).
    weights={slot:log((correct[slot]+1)/(n-correct[slot]+1)) for slot in SLOTS}
    return {'n':n,'correct':correct,'tokens':costs,'chosen_single':chosen,'weights':weights}


def predict(slots,policy):
    scores={}
    for slot in SLOTS:
        answer=slots[slot]['answer']
        if answer not in (None,'abstain'):scores[answer]=scores.get(answer,0.)+policy['weights'][slot]
    counts=Counter(slots[k]['answer'] for k in SLOTS if slots[k]['answer'] not in (None,'abstain'))
    voted=unique_max(counts)
    voted=voted if counts.get(voted,0)>=3 else 'abstain'
    return {'train_selected_single':slots[policy['chosen_single']]['answer'],
            'five_report_majority':voted,
            'train_reliability_vote':unique_max(scores) if scores else 'abstain',
            'initial_majority':majority([slots['fast'+str(i)]['answer'] for i in range(3)])}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=json.loads((source.OUT/'train_sealed_outputs.json').read_text())
    trainlabels=json.loads((source.OUT/'train_labels.json').read_text())
    policy=fit(train,trainlabels);freeze(OUT/'frozen_readout.json',policy)
    test=json.loads((source.OUT/'test_sealed_outputs.json').read_text())
    decisions={cid:predict(s,policy) for cid,s in test.items()}
    freeze(OUT/'sealed_readouts.json',decisions)
    labels=json.loads((source.OUT/'test_labels.json').read_text())
    rows=[]
    for cid,s in test.items():
        available={s[k]['answer'] for k in SLOTS}
        rows.append({'id':cid,'gold':labels[cid],'answers':{k:s[k]['answer'] for k in SLOTS},
            'initial_majority':decisions[cid]['initial_majority'],
            'gold_available':labels[cid] in available,
            'new_upgrade_correct':not any(s['fast'+str(i)]['answer']==labels[cid] for i in range(3))
                                  and any(s['think'+str(i)]['answer']==labels[cid] for i in (1,2)),
            'decisions':decisions[cid]})
    result={'scope':'post-hoc diagnostic using opened old test outputs; train-only fit is NOT a fresh independent test',
        'new_api_calls':0,'n':len(test),'oracle_available_correct':sum(r['gold_available'] for r in rows),
        'unavailable_correct_cases':sum(not r['gold_available'] for r in rows),
        'available_but_initial_vote_wrong':sum(r['gold_available'] and r['initial_majority']!=r['gold'] for r in rows),
        'metrics':{method:{'correct':sum(v[method]==labels[c] for c,v in decisions.items()),
            'tokens':sum(s[policy['chosen_single']]['tokens'] for s in test.values()) if method=='train_selected_single' else
                    sum(sum(s[k]['tokens'] for k in (SLOTS if method!='initial_majority' else SLOTS[:3])) for s in test.values()),
            'calls':len(test)*(1 if method=='train_selected_single' else 3 if method=='initial_majority' else 5)}
            for method in next(iter(decisions.values()))},'rows':rows,
        'source_sha256':{name:source.h(source.OUT/name) for name in
            ('train_sealed_outputs.json','train_labels.json','test_sealed_outputs.json','test_labels.json')},
        'limits':['oracle coverage is an offline upper bound, never deployable selection',
                  'log-odds weighting assumes no proven independence or calibration',
                  'no reward tuning, no task-specific edits, no new theorem or game benefit']}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
