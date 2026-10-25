"""Descriptive source-group bootstrap with refitting, not a new held-out result."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import time

from experiments.mechanism import audit_simple_readout_crossfit as base
from experiments.mechanism.attribute_legacy_gain import ROOT,trace_kind
from experiments.mechanism.audit_legacy_source_groups import normalize,split_question
from experiments.mechanism.audit_legacy_conclusion_parsing import conclusion
from experiments.mechanism.run_upgrade_pilot import freeze

OUT=base.OUT/'refit_bootstrap'


def load():
    data=[json.loads(s) for s in (ROOT/'experiments/results/mechanism/frozen/formal_v10/question_level.jsonl').read_text().splitlines()]
    splits=json.loads((ROOT/'experiments/mechanism/splits.json').read_text())['datasets']
    questions={d:{str(r['question_id']):r['question'] for r in json.loads((ROOT/sp['source']).read_text())['questions']} for d,sp in splits.items()}
    grouping=json.loads((ROOT/'experiments/results/audit/legacy_gain_attribution_2026-10-22/source_groups.json').read_text())
    excluded={(d,str(r['id'])) for d,v in grouping['datasets'].items() for r in v['grouping']['excluded_records']}
    rows=[]
    for r in data:
        if '__static_aeic@' not in r['condition'] or (r['dataset'],str(r['question_id'])) in excluded:continue
        group=r['dataset']+'\n'+normalize(split_question(questions[r['dataset']][str(r['question_id'])])[0])
        fold=int(sha256(('readout-crossfit-v1\n'+group).encode()).hexdigest()[:8],16)%5
        rows.append({'group':group,'fold':fold,'domain':r['dataset'],'gold':r['ground_truth'],
            'turns':[{'agent':t['agent_id'],'route':trace_kind(t),
                      'answer':conclusion(t['report']['C'],t['stance'],'literal_yes_no_only')}
                     for t in sorted(r['rounds'],key=lambda t:t['agent_id']) if t['round_id']==3]})
    assert len(rows)==370 and len({r['group'] for r in rows})==370
    return rows


def delta(rows):
    difference=0
    for f in range(5):
        train=[r for r in rows if r['fold']!=f];test=[r for r in rows if r['fold']==f]
        assert not {r['group'] for r in train}&{r['group'] for r in test}
        model=base.fit(train)
        for row in test:
            pred=base.predict(row['turns'],model)
            difference+=int(pred['route_reliability_weighted']==row['gold'])-int(pred['binary_unweighted']==row['gold'])
    return difference/len(rows)


def main():
    rows=load();domains={d:[r for r in rows if r['domain']==d] for d in sorted({r['domain'] for r in rows})}
    rng=random.Random(20261024);draws=[];start=time.monotonic()
    for i in range(5000):
        sample=[cell[rng.randrange(len(cell))] for cell in domains.values() for _ in cell]
        draws.append(delta(sample))
        if (i+1)%500==0:print('refit bootstrap',i+1,'/5000',flush=True)
    draws.sort();OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'result.json',{'scope':'post-hoc descriptive stratified source-group refit bootstrap on historical frozen reports',
        'n_source_groups':370,'draws':5000,'rng_seed':20261024,'observed_weighted_minus_binary':delta(rows),
        'percentile95_descriptive':[draws[124],draws[4874]],'elapsed_seconds':time.monotonic()-start,
        'method':'resample unique source questions within domain; retain group-fold assignment; refit all5 training folds each draw',
        'new_model_calls':0,'code_sha256':base.fingerprint(Path(__file__)),
        'limits':['not independent replication or confirmatory CI; scope selected after seeing earlier results',
            'conditional on frozen outputs and historical feedback; source-level independence may fail under shared histories',
            'does not quantify API/model-version stochasticity','do not promote standard weighted voting as novel mechanism']})
    print('observed',delta(rows),'descriptive interval',draws[124],draws[4874],flush=True)


if __name__=='__main__':main()
