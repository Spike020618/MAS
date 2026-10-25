"""Exploratory, zero-new-call routing replay. Charge the initial panel in full."""
from collections import Counter
from itertools import product
import json

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.sensitivity_resource_allocation import explicit_answer
from experiments.mechanism.run_upgrade_pilot import read_rows,freeze

OUT=p.ROOT/'experiments/results/resource_allocation/low_cost_routing_v1'


def vote(values,choices):
    return next((c for c in choices if values.count(c)>=2),'abstain')


def upgrade(values,mode='not_unanimous'):
    usable=[v for v in values if v is not None and v!='abstain']
    if mode=='not_unanimous':return not(len(usable)==3 and len(set(usable))==1)
    if mode=='no_majority':return max(Counter(usable).values(),default=0)<2
    raise ValueError('unknown gate')


def evaluate():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=json.loads((p.OUT/'public_cases.json').read_text())
    rows=read_rows(p.OUT/'responses.jsonl');index={r['key']:r for r in rows}
    previous=json.loads((p.OUT/'bounded_analysis.json').read_text())
    if p.digest(rows)!=previous['journal_hash']:raise ValueError('source commitment changed')
    pools={};token={};missing_usage=[]
    # No labels or domain IDs enter a decision. Keep unparseable observations unknown.
    for case in cases:
        values=[[],[]];cost=[0,0]
        for high,backend in enumerate(['zhipu_flash','deepseek']):
            for agent in range(3):
                key=p.digest(case)[:20]+'/'+backend+'/'+str(agent)
                row=index.get(key)
                if row is None:values[high].append(None);missing_usage.append(case['id']);continue
                value=p.body(row);cost[high]+=value['usage']['total_tokens']
                values[high].append(explicit_answer(value['choices'][0]['message']['content'],case['choices']))
        pools[case['id']]=values;token[case['id']]=cost
    decisions={}
    for mode in ['not_unanimous','no_majority']:
        chosen={c['id'] for c in cases if upgrade(pools[c['id']][0],mode)}
        k=len(chosen)
        fixed=set(c['id'] for c in sorted(cases,key=lambda c:p.digest(['routing-fixed-v1',c['id']]))[:k])
        length=set(c['id'] for c in sorted(cases,key=lambda c:(-len(c['question']),p.digest(c['question'])))[:k])
        decisions[mode]={'selected':sorted(chosen),'fixed':sorted(fixed),'length':sorted(length),'quota':k}
    freeze(OUT/'decisions.json',{'scope':'post-hoc opened-data diagnosis, not independent validation',
        'source_journal_hash':p.digest(rows),'policy':'upgrade all three seats iff gate triggers; no learned coefficients',
        'cost':'all routing policies pay for 72 initial low attempts plus 3 high calls per upgraded question',
        'selection_api_calls':0,'decisions':decisions})
    # Only after committing all decisions do we read evaluation labels.
    labels={r['id']:r['label'] for r in json.loads((p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    output={}
    for mode,plan in decisions.items():
        per_case=[];k=plan['quota'];n=len(cases)
        for case in cases:
            low,high=pools[case['id']];missing=[i for i,v in enumerate(low) if v is None]
            variants=[]
            for possible in product([*case['choices'],'abstain'],repeat=len(missing)):
                values=list(low)
                for i,v in zip(missing,possible):values[i]=v
                lo=int(vote(values,case['choices'])==labels[case['id']])
                hi=int(vote(high,case['choices'])==labels[case['id']])
                variants.append({'all_low':lo,'all_high':hi,
                    'gate':hi if case['id'] in plan['selected'] else lo,
                    'fixed':hi if case['id'] in plan['fixed'] else lo,
                    'length':hi if case['id'] in plan['length'] else lo,
                    'random_expected':(k/n)*hi+(1-k/n)*lo})
            per_case.append({'id':case['id'],'observed_low':low,'gate_upgrades':case['id'] in plan['selected'],
                'bounds':{key:[min(v[key] for v in variants),max(v[key] for v in variants)] for key in variants[0]}})
        low_tokens=sum(v[0] for v in token.values());high_tokens=sum(v[1] for v in token.values())
        costs={name:low_tokens+sum(token[cid][1] for cid in plan[field]) for name,field in [('gate','selected'),('fixed','fixed'),('length','length')]}
        costs['random_expected']=low_tokens+(k/n)*high_tokens
        costs.update(all_low=low_tokens,all_high=high_tokens)
        output[mode]={'upgraded_questions':k,'high_calls':3*k,'low_attempts':72,'total_attempts':72+3*k,
            'correct_count_bounds':{key:[sum(r['bounds'][key][j] for r in per_case) for j in [0,1]] for key in per_case[0]['bounds']},
            'known_token_lower_bounds':costs,'all_high_tokens_exact':high_tokens,
            'unknown_initial_usage_requests':len(set(missing_usage)),
            'cost_ratio_break_even_vs_all_high':(72/(72-3*k)) if k<24 else None,
            'cost_ratio_definition':'high per-call cost / low per-call cost; equal within-backend cost simplifying assumption, NOT actual billing',
            'rows':per_case}
    result={'scope':'exploratory saved independent-report replay, zero new API calls, no equilibrium claim',
        'uncertainty':'all unregistered/missing low answers allowed any choice; observed gate decisions remain fixed',
        'not_fair_to_omit':['initial low calls','missing request usage','full high bypass baseline'],
        'policies':output,'source_journal_hash':p.digest(rows)}
    freeze(OUT/'result.json',result)
    print(json.dumps({mode:{k:v for k,v in values.items() if k!='rows'} for mode,values in output.items()},indent=2))
    return result


if __name__=='__main__':evaluate()
