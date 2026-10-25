"""Complete only never-attempted slots; bound unresolved transport outcomes."""
import argparse
import json
from itertools import product

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows


def finish():
    directory=p.OUT
    with lock(directory):
        cases=json.loads((directory/'public_cases.json').read_text())
        attempts=read_rows(directory/'attempts.jsonl')
        rows=read_rows(directory/'responses.jsonl')
        unresolved={r['key'] for r in attempts}-{r['key'] for r in rows}
        freeze(directory/'connection_amendment.json',{
            'reason':'ConnectionError with unknown service outcome; never retry unresolved slots',
            'unresolved_at_amendment':sorted(unresolved),'labels_read':False,
            'action':'collect never-attempted slots within original 216-attempt budget; report all-case uncertainty bounds',
            'not_allowed':['replace missing answer','drop affected question','claim complete efficacy validation'],
            'completion_code_hash':p.digest(__import__('pathlib').Path(__file__).read_text())})
        calls=CallJournal(directory,p.transport,216)
        for case in cases:
            for agent in range(3):
                for backend in sorted(['zhipu_flash','deepseek'],key=lambda b:p.digest([case['id'],agent,b])):
                    key=p.digest(case)[:20]+'/'+backend+'/'+str(agent)
                    if key in unresolved:continue
                    value=p.body(calls.call(key,p.report_request(case,agent,backend)))
                    decoded=p.decode_content(value['choices'][0]['message']['content'],case['choices'])
                    if 'invalid_json_object' in decoded['issues']:raise ValueError('invalid JSON; stop')
            print('remaining slots collected',case['id'],flush=True)


def bounds():
    directory=p.OUT;cases=json.loads((directory/'public_cases.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text())
    if p.digest(cases)!=protocol['cases_hash'] or p.digest(p.__file__ and __import__('pathlib').Path(p.__file__).read_text())!=protocol['code_hash']:
        raise ValueError('frozen protocol changed')
    rows=read_rows(directory/'responses.jsonl');index={r['key']:r for r in rows}
    attempts=read_rows(directory/'attempts.jsonl')
    if len(attempts)!=216 or len({a['key'] for a in attempts})!=216 or len(index)!=len(rows):
        raise ValueError('all original slots must be attempted exactly once')
    unresolved={a['key'] for a in attempts}-set(index)
    amendment=json.loads((directory/'connection_amendment.json').read_text())
    if unresolved!=set(amendment['unresolved_at_amendment']):raise ValueError('additional missing requests')
    def no_network(_):raise AssertionError('offline only')
    journal=CallJournal(directory,no_network,216)
    selections=json.loads((directory/'selections.json').read_text())
    for case in cases:
        for agent in range(3):
            prefix=p.digest(case)[:20]
            value=p.body(journal.call(prefix+'/choice/'+str(agent),p.choice_request(case,agent)))
            if selections[case['id']][agent]!=int(json.loads(value['choices'][0]['message']['content'])['ACTION']=='aeic_effort'):
                raise ValueError('selection mismatch')
            for backend in ['zhipu_flash','deepseek']:
                key=prefix+'/'+backend+'/'+str(agent)
                if key not in unresolved:journal.call(key,p.report_request(case,agent,backend))
    # Labels are first used only after every never-attempted slot is collected and verified.
    labels={r['id']:r['label'] for r in json.loads((p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    total_high=sum(sum(v) for v in selections.values());fixed=p.fixed_profiles(cases,total_high)
    metrics=[];actual={};tokens=0;selector_tokens=0
    for row in rows:
        value=p.body(row);tokens+=value['usage']['total_tokens']
        actual.setdefault(row['request']['backend'],set()).add(value['model'])
        if '/choice/' in row['key']:selector_tokens+=value['usage']['total_tokens']
    for case in cases:
        pool=[[],[]];missing=[]
        for high,backend in enumerate(['zhipu_flash','deepseek']):
            for agent in range(3):
                key=p.digest(case)[:20]+'/'+backend+'/'+str(agent)
                if key in unresolved:pool[high].append(None);missing.append((high,agent))
                else:
                    decoded=p.decode_content(p.body(index[key])['choices'][0]['message']['content'],case['choices'])
                    if 'invalid_json_object' in decoded['issues']:raise ValueError('invalid JSON')
                    pool[high].append(decoded)
        variants=[]
        for possibilities in product([*case['choices'],'abstain'],repeat=len(missing)):
            state=[list(v) for v in pool]
            for (high,agent),conclusion in zip(missing,possibilities):
                state[high][agent]=p.decode_content(json.dumps({'A':'','E':[],'I':'','C':conclusion}),case['choices'])
            scores={profile:int(p.majority([state[a][i] for i,a in enumerate(profile)],case['choices'])==labels[case['id']]) for profile in p.PROFILES}
            chosen=tuple(selections[case['id']]);same=[v for v in p.PROFILES if sum(v)==sum(chosen)]
            variants.append({'all_low':scores[(0,0,0)],'all_high':scores[(1,1,1)],'contract':scores[chosen],
                'fixed_quota':scores[fixed[case['id']]],
                'same_k_random_expected':sum(scores[v] for v in same)/len(same),
                'global_quota_random_expected':sum(scores[v]*p.random_profile_weight(72,total_high,v) for v in p.PROFILES),
                'realized_oracle':max(scores.values())})
        metrics.append({'id':case['id'],'missing_report_slots':len(missing),
            'bounds':{name:[min(v[name] for v in variants),max(v[name] for v in variants)] for name in variants[0]}})
    result={'status':'INCOMPLETE_RESPONSE_BOUNDED_DIAGNOSTIC_NOT_FULL_EFFICACY_ACCEPTANCE',
        'n':24,'attempts':len(attempts),'successful_responses':len(rows),'unresolved_slots':sorted(unresolved),
        'high_report_quota':total_high,'actual_models':{k:sorted(v) for k,v in actual.items()},
        'observed_total_tokens':tokens,'selector_tokens':selector_tokens,
        'unknown_request_usage':'unknown, not silently counted as zero',
        'correct_count_bounds':{k:[sum(r['bounds'][k][j] for r in metrics) for j in [0,1]] for k in metrics[0]['bounds']},
        'rows':metrics,'journal_hash':p.digest(rows),'no_missing_answer_imputed':True}
    freeze(directory/'bounded_analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finish',action='store_true');args=parser.parse_args()
    if args.finish:finish()
    else:bounds()
