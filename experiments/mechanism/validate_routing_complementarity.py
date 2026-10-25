"""Frozen source-disjoint small validation of a label-blind union policy."""
import argparse
import hashlib
from itertools import product
import json
from math import comb
from pathlib import Path

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism import probe_probability_routing as base
from experiments.mechanism.audit_probability_compatibility import compatible_decode
from experiments.mechanism.prepare_routing_holdout import OUT as DATA
from experiments.mechanism.replay_low_cost_router import vote
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows

OUT=p.ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1'


def plans(cases,observations):
    native=base.select(cases,observations);fs=native['features'];k=native['quota_questions']
    gate=set(native['policies']['disagreement']);prob=set(native['policies']['probability'])
    union=gate|prob;m=len(union)
    # Match the already locked native ranking, so top-M contains top-k even on ties.
    tie=lambda cid:p.digest(['probability-routing-v1',cid])
    rank_prob=sorted(fs,key=lambda cid:(fs[cid]['probabilities_valid'],
        -(fs[cid]['ensemble_uncertainty'] if fs[cid]['ensemble_uncertainty'] is not None else 0),tie(cid)))
    rank_gate=sorted(fs,key=lambda cid:(not fs[cid]['disagreement'],tie(cid)))
    chosen={'hybrid_union':sorted(union),'disagreement_priority':sorted(rank_gate[:m]),
        'probability_top_m':sorted(rank_prob[:m]),'fixed':sorted(sorted(fs,key=tie)[:m]),
        'length':sorted(c['id'] for c in sorted(cases,key=lambda c:(-len(c['question']),tie(c['id'])))[:m])}
    return {'native_quota':k,'matched_quota':m,'native_gate':sorted(gate),'native_probability':sorted(prob),
        'features':fs,'policies':chosen}


def dependencies():
    return {str(Path(module.__file__).relative_to(p.ROOT)):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in [p,base,__import__('experiments.mechanism.audit_probability_compatibility',fromlist=[''])]}


def verify_data():
    prov=json.loads((DATA/'provenance.json').read_text())
    for name,h in prov['output_sha256'].items():
        if hashlib.sha256((DATA/name).read_bytes()).hexdigest()!=h:raise ValueError('held-out data commitment changed')
    cases=json.loads((DATA/'public_cases.json').read_text())
    if len(cases)!=24 or any(set(c)!={'id','source_id','question','choices'} for c in cases):raise ValueError('invalid public cases')
    return cases,prov


def run():
    cases,prov=verify_data()
    with lock(OUT):
        freeze(OUT/'protocol.json',{'version':'routing-complementarity-holdout-v1',
            'cases_hash':p.digest(cases),'data_provenance_hash':p.digest(prov),
            'code_hash':p.digest(Path(__file__).read_text()),'dependencies':dependencies(),
            'primary':'hybrid union versus disagreement-priority AND probability-top-M, same M whole-panel upgrades',
            'hybrid':'union of disagreement gate with top-k probability uncertainty; k is count of disagreement questions',
            'matched_controls':'all non-unanimous first then fixed hash to M; probability top M; fixed hash M; length top M',
            'native_controls':'native gate and probability top k are secondary, not equal-budget primary comparisons',
            'parser':'literal numeric-string compatibility locked; no normalization, answer repair, or sum repair',
            'scope':'source-stem screened new small validation; not broad confirmation, pretraining novelty or game acceptance',
            'max_attempts':144,'cap':600,'temperature':.3,
            'order':'all 72 low first, freeze all decisions, then 72 high; only then private scoring',
            'statistics':'paired exact McNemar, Holm across TWO primary contrasts; descriptive small sample',
            'failure':'unknown transports not retried and bounded; no repairs; HTTP/non-stop stops',
            'no_tuning':True,'no_payments':True,'no_expansion':True})
        freeze(OUT/'public_cases.json',cases)
        calls=CallJournal(OUT,p.transport,144)
        unknown={a['key'] for a in read_rows(OUT/'attempts.jsonl')}-{r['key'] for r in read_rows(OUT/'responses.jsonl')}
        observations={}
        for backend in ['zhipu_flash','deepseek']:
            for case in cases:
                reports=[]
                for agent in range(3):
                    key=base.key(case,agent,backend)
                    if key in unknown:reports.append({'answer':None,'probabilities':None,'issues':['unknown_transport']});continue
                    try:row=calls.call(key,base.request(case,agent,backend))
                    except RuntimeError as exc:
                        if str(exc).startswith('transport failed;'):
                            unknown.add(key);reports.append({'answer':None,'probabilities':None,'issues':['unknown_transport']});continue
                        raise
                    body=p.body(row);reports.append(compatible_decode(body['choices'][0]['message']['content'],case['choices']))
                if backend=='zhipu_flash':observations[case['id']]=reports
                print('saved',backend,case['id'],flush=True)
            if backend=='zhipu_flash':freeze(OUT/'selections.json',plans(cases,observations))
    return analyze()


def exact_pair(rows,left,right):
    if any(r['bounds'][a][0]!=r['bounds'][a][1] for r in rows for a in [left,right]):return {'uncertain':True}
    gain=sum(r['bounds'][left][0]==0 and r['bounds'][right][0]==1 for r in rows)
    harm=sum(r['bounds'][left][0]==1 and r['bounds'][right][0]==0 for r in rows)
    n=gain+harm
    return {'corrections':gain,'harms':harm,'net':gain-harm,
        'exact_mcnemar_p':min(1.,2*sum(comb(n,j) for j in range(min(gain,harm)+1))/2**n) if n else 1.}


def holm(values):
    result={};previous=0
    for rank,(name,value) in enumerate(sorted(values.items(),key=lambda x:x[1])):
        previous=max(previous,min(1.,(len(values)-rank)*value));result[name]=previous
    return result


def analyze():
    cases,prov=verify_data();protocol=json.loads((OUT/'protocol.json').read_text())
    if protocol['code_hash']!=p.digest(Path(__file__).read_text()) or protocol['dependencies']!=dependencies():raise ValueError('protocol code changed')
    if protocol['cases_hash']!=p.digest(cases) or protocol['data_provenance_hash']!=p.digest(prov):raise ValueError('data changed')
    rows=read_rows(OUT/'responses.jsonl');index={r['key']:r for r in rows};attempts=read_rows(OUT/'attempts.jsonl')
    if len(attempts)!=144 or len({a['key'] for a in attempts})!=144 or len(index)!=len(rows):raise ValueError('incomplete/duplicate collection')
    def forbidden(_):raise AssertionError('offline scoring cannot call API')
    journal=CallJournal(OUT,forbidden,144);decoded={};tokens={};models={};issues=[]
    for backend in ['zhipu_flash','deepseek']:
        decoded[backend]={};tokens[backend]={}
        for case in cases:
            reports=[];cost=0
            for agent in range(3):
                key=base.key(case,agent,backend)
                if key not in index:reports.append({'answer':None,'probabilities':None,'issues':['unknown_transport']});continue
                row=journal.call(key,base.request(case,agent,backend));body=p.body(row)
                report=compatible_decode(body['choices'][0]['message']['content'],case['choices']);reports.append(report)
                cost+=body['usage']['total_tokens'];models.setdefault(backend,set()).add(body['model'])
                if report['issues']:issues.append({'id':case['id'],'backend':backend,'agent':agent,'issues':report['issues']})
            decoded[backend][case['id']]=reports;tokens[backend][case['id']]=cost
    decision=json.loads((OUT/'selections.json').read_text())
    if decision!=plans(cases,decoded['zhipu_flash']):raise ValueError('selection changed')
    low_end=max(r['completed_utc'] for r in rows if r['request']['backend']=='zhipu_flash')
    if any(a['started_utc']<low_end for a in attempts if '/deepseek/' in a['key']):raise ValueError('high before low sealed')
    # First evaluation read of private labels occurs here, after all calls and commitments.
    labels={r['id']:r['label'] for r in json.loads((DATA/'private_labels.json').read_text())}
    chosen=dict(decision['policies'],native_gate=decision['native_gate'],native_probability=decision['native_probability'])
    n=len(cases);m=decision['matched_quota'];metrics=[]
    for case in cases:
        cid=case['id'];answers=[[r['answer'] or 'abstain' for r in decoded[b][cid]] for b in ['zhipu_flash','deepseek']]
        unknown=[(b,a) for b,backend in enumerate(['zhipu_flash','deepseek']) for a in range(3) if base.key(case,a,backend) not in index]
        variants=[]
        for possible in product([*case['choices'],'abstain'],repeat=len(unknown)):
            states=[list(v) for v in answers]
            for (b,a),value in zip(unknown,possible):states[b][a]=value
            lo=int(vote(states[0],case['choices'])==labels[cid]);hi=int(vote(states[1],case['choices'])==labels[cid])
            record={'all_low':lo,'all_high':hi,'random_expected':lo*(1-m/n)+hi*m/n}
            for name,selected in chosen.items():record[name]=hi if cid in selected else lo
            record['upgrade_gain']=hi-lo;variants.append(record)
        metrics.append({'id':cid,'features':decision['features'][cid],
            'bounds':{key:[min(v[key] for v in variants),max(v[key] for v in variants)] for key in variants[0]}})
    counts={name:[sum(r['bounds'][name][j] for r in metrics) for j in [0,1]] for name in metrics[0]['bounds'] if name!='upgrade_gain'}
    primary={name:exact_pair(metrics,name,'hybrid_union') for name in ['disagreement_priority','probability_top_m']}
    if all('exact_mcnemar_p' in v for v in primary.values()):
        corrected=holm({name:v['exact_mcnemar_p'] for name,v in primary.items()})
        for name,value in corrected.items():primary[name]['holm_p']=value
    costs={name:sum(tokens['zhipu_flash'].values())+sum(tokens['deepseek'][cid] for cid in selected) for name,selected in chosen.items()}
    costs.update(all_low=sum(tokens['zhipu_flash'].values()),all_high=sum(tokens['deepseek'].values()),
        random_expected=sum(tokens['zhipu_flash'].values())+m/n*sum(tokens['deepseek'].values()))
    valid=[r for r in metrics if r['features']['probabilities_valid'] and r['bounds']['upgrade_gain'][0]==r['bounds']['upgrade_gain'][1]]
    eligible_gains=[r['bounds']['upgrade_gain'][0]>0 for r in valid]
    result={'scope':protocol['scope'],'attempts':len(attempts),'responses':len(rows),'unknown':len(attempts)-len(rows),
        'native_quota':decision['native_quota'],'matched_quota':m,'matched_high_calls':3*m,'low_attempts_per_routing':72,
        'counts':counts,'primary_contrasts':primary,'known_token_costs':costs,
        'valid_probability_slots':{b:sum(r['probabilities'] is not None for rs in decoded[b].values() for r in rs) for b in decoded},
        'valid_probability_questions':len(valid),'gain_positive_valid_questions':sum(eligible_gains),
        'gain_auc':base.auc([r['features']['ensemble_uncertainty'] for r in valid],eligible_gains),
        'actual_models':{b:sorted(v) for b,v in models.items()},
        'collection_total_tokens':sum(v for b in tokens.values() for v in b.values()),
        'decision':'do not change rules or add payments on these outcomes; no automatic expansion',
        'rows':metrics,'issues':issues,'response_hash':p.digest(rows)}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','issues']},ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:analyze()
