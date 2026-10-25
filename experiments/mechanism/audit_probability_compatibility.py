"""Post-hoc literal numeric-string parsing; no renormalization or new calls."""
import json
import re

from experiments.mechanism import probe_probability_routing as base
from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import read_rows,freeze


def compatible_decode(content,choices):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate key')
            result[k]=v
        return result
    try:obj=json.loads(content,object_pairs_hook=unique)
    except (ValueError,TypeError):return base.decode(content,choices)
    if isinstance(obj,dict) and isinstance(obj.get('probabilities'),dict):
        for key,value in obj['probabilities'].items():
            if isinstance(value,str) and re.fullmatch(r'(?:0(?:\.[0-9]+)?|1(?:\.0+)?)',value.strip()):
                obj['probabilities'][key]=float(value.strip())
    return base.decode(json.dumps(obj),choices)


def summarize(cases,observations,parent_rows):
    decision=base.select(cases,observations);n=len(cases);k=decision['quota_questions']
    counts={};flows={}
    for name,chosen in decision['policies'].items():
        selected=set(chosen)
        counts[name]=sum(parent_rows[c['id']]['correct_bounds']['all_high' if c['id'] in selected else 'all_low'][0] for c in cases)
        flows[name]={'corrections':sum(parent_rows[cid]['correct_bounds']['upgrade_gain'][0]>0 for cid in selected),
                     'harms':sum(parent_rows[cid]['correct_bounds']['upgrade_gain'][0]<0 for cid in selected)}
    lo=sum(parent_rows[c['id']]['correct_bounds']['all_low'][0] for c in cases)
    hi=sum(parent_rows[c['id']]['correct_bounds']['all_high'][0] for c in cases)
    counts.update(all_low=lo,all_high=hi,random_expected=lo*(1-k/n)+hi*k/n)
    eligible=[c for c in cases if decision['features'][c['id']]['probabilities_valid']]
    gain=[parent_rows[c['id']]['correct_bounds']['upgrade_gain'][0]>0 for c in eligible]
    diagnostic={name:base.auc([decision['features'][c['id']][name] for c in eligible],gain)
                for name in ['ensemble_uncertainty','individual_uncertainty']}
    return {'n':n,'quota_questions':k,'correct_counts':counts,'flows':flows,
        'valid_probability_questions':len(eligible),'gain_auc':diagnostic,'selections':decision}


def main():
    directory=base.OUT;cases=json.loads((directory/'public_cases.json').read_text())
    rows=read_rows(directory/'responses.jsonl');index={r['key']:r for r in rows}
    parent=json.loads((directory/'result.json').read_text())
    if parent['response_hash']!=p.digest(rows) or parent['unknown']:raise ValueError('requires complete, unchanged parent collection')
    parent_rows={r['id']:r for r in parent['rows']};output={}
    for mode,decoder in [('strict',base.decode),('numeric_string_compatible',compatible_decode)]:
        observations={}
        for case in cases:
            observations[case['id']]=[decoder(p.body(index[base.key(case,agent,'zhipu_flash')])['choices'][0]['message']['content'],case['choices']) for agent in range(3)]
        full=summarize(cases,observations,parent_rows)
        eligible=[c for c in cases if full['selections']['features'][c['id']]['probabilities_valid']]
        subset=summarize(eligible,observations,parent_rows) if eligible else None
        output[mode]={'all_cases':full,'complete_probability_subset_posthoc':subset,
            'valid_probability_slots':sum(r['probabilities'] is not None for reports in observations.values() for r in reports)}
    result={'scope':'post-hoc parser sensitivity and complete-case diagnosis, NOT replacement primary or independent validation',
        'rule':'convert literal decimal strings only; no normalization, answer changes, dropped choices or model repair',
        'parent_response_hash':parent['response_hash'],'primary_preserved':True,'new_calls':0,'analyses':output}
    freeze(directory/'probability_compatibility_audit.json',result)
    print(json.dumps({mode:{'all_cases':{k:v for k,v in data['all_cases'].items() if k!='selections'},
        'complete_subset':{k:v for k,v in data['complete_probability_subset_posthoc'].items() if k!='selections'} if data['complete_probability_subset_posthoc'] else None,
        'valid_slots':data['valid_probability_slots']} for mode,data in output.items()},indent=2))
    return result


if __name__=='__main__':main()
