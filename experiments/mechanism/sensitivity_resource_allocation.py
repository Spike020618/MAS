"""Post-hoc answer-key compatibility audit, never overwrite strict scoring."""
import json
from itertools import product

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import read_rows,freeze


def explicit_answer(content,choices):
    """Use only unambiguous literal C/answer, never infer from reasoning or gold."""
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('duplicate key')
            out[key]=value
        return out
    try:obj=json.loads(content,object_pairs_hook=unique)
    except (ValueError,TypeError):return None
    if not isinstance(obj,dict):return None
    values=[obj[k] for k in ['C','answer'] if k in obj]
    if not values or any(not isinstance(v,str) or v not in [*choices,'abstain'] for v in values):return None
    return values[0] if len(set(values))==1 else None


def main():
    directory=p.OUT
    strict=json.loads((directory/'bounded_analysis.json').read_text())
    cases=json.loads((directory/'public_cases.json').read_text())
    rows=read_rows(directory/'responses.jsonl');index={r['key']:r for r in rows}
    if p.digest(rows)!=strict['journal_hash']:raise ValueError('journal changed')
    labels={r['id']:r['label'] for r in json.loads((p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    selections=json.loads((directory/'selections.json').read_text());quota=strict['high_report_quota']
    fixed=p.fixed_profiles(cases,quota);results={}
    for policy in ['literal_C_or_answer','all_unresolved_answer_bounds']:
        metrics=[];recovered=0;unresolved=0
        for case in cases:
            pool=[[],[]];missing=[]
            for high,backend in enumerate(['zhipu_flash','deepseek']):
                for agent in range(3):
                    row=index.get(p.digest(case)[:20]+'/'+backend+'/'+str(agent))
                    if row is None:value=None
                    else:
                        content=p.body(row)['choices'][0]['message']['content']
                        value=explicit_answer(content,case['choices'])
                        old=p.decode_content(content,case['choices'])
                        recovered+=int(value is not None and not old['conclusion_usable'])
                    if value is None:
                        unresolved+=1
                        if row is None or policy=='all_unresolved_answer_bounds':missing.append((high,agent))
                        else:value='abstain'
                    pool[high].append(value)
            variants=[]
            for possibilities in product([*case['choices'],'abstain'],repeat=len(missing)):
                state=[list(v) for v in pool]
                for (high,agent),value in zip(missing,possibilities):state[high][agent]=value
                scores={profile:int(next((c for c in case['choices'] if sum(state[a][i]==c for i,a in enumerate(profile))>=2),'abstain')==labels[case['id']]) for profile in p.PROFILES}
                chosen=tuple(selections[case['id']]);same=[v for v in p.PROFILES if sum(v)==sum(chosen)]
                variants.append({'all_low':scores[(0,0,0)],'all_high':scores[(1,1,1)],'contract':scores[chosen],
                    'fixed_quota':scores[fixed[case['id']]],
                    'same_k_random_expected':sum(scores[v] for v in same)/len(same),
                    'global_quota_random_expected':sum(scores[v]*p.random_profile_weight(72,quota,v) for v in p.PROFILES),
                    'realized_oracle':max(scores.values())})
            metrics.append({'id':case['id'],'bounds':{k:[min(v[k] for v in variants),max(v[k] for v in variants)] for k in variants[0]}})
        results[policy]={'recovered_literal_answers':recovered,'unresolved_report_answers':unresolved,
            'correct_count_bounds':{k:[sum(r['bounds'][k][j] for r in metrics) for j in [0,1]] for k in metrics[0]['bounds']},'rows':metrics}
    output={'scope':'post-hoc sensitivity, not preregistered primary or new model execution',
            'rule':'Only literal registered C/answer; conflicting, unregistered or ambiguous values never inferred from A/E/I or gold',
            'strict_result_preserved':True,'new_api_calls':0,'journal_hash':strict['journal_hash'],'policies':results}
    freeze(directory/'answer_compatibility_sensitivity.json',output)
    print(json.dumps({k:{f:v for f,v in value.items() if f!='rows'} for k,value in results.items()},indent=2))


if __name__=='__main__':main()
