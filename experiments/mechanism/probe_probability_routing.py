"""Joint answer/probability diagnostic. No incentives, fitting or repair calls."""
import argparse
from itertools import product
import json
from math import isfinite,log
from pathlib import Path

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.replay_low_cost_router import upgrade,vote
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows

OUT=p.ROOT/'experiments/results/resource_allocation/probability_routing_v1'


def request(case,agent,backend):
    from experiments.core.llm import AGENT_PERSONAS,LLM_BACKENDS
    prompt=('Solve independently using your knowledge. No peers, external checks or upgrade results are available. '
        'Return only one JSON object with exactly two keys: answer and probabilities. '
        'answer must be one of the registered choices and maximize your probabilities. '
        'probabilities must be an object containing EVERY registered choice exactly once, with numeric values '
        'between 0 and 1 summing to 1 (four decimal places are enough). Express your actual uncertainty, '
        'not a certainty claim. Do not include reasoning or other fields.')
    payload={'question':case['question'],'registered_choices':case['choices'],
        'required_shape':{'answer':'one registered choice',
            'probabilities':{c:'numeric probability' for c in case['choices']}}}
    return {'backend':backend,'model':LLM_BACKENDS[backend]['model'],'temperature':.3,'max_tokens':600,
        'messages':[{'role':'system','content':AGENT_PERSONAS[agent]['system']+'\n'+prompt},
                    {'role':'user','content':json.dumps(payload,sort_keys=True,ensure_ascii=False)}],
        'response_format':{'type':'json_object'}}


def decode(content,choices):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate key')
            result[k]=v
        return result
    try:obj=json.loads(content,object_pairs_hook=unique)
    except (ValueError,TypeError):return {'answer':None,'probabilities':None,'issues':['invalid_or_duplicate_json']}
    if not isinstance(obj,dict):return {'answer':None,'probabilities':None,'issues':['not_object']}
    issues=[];answer=obj.get('answer');distribution=obj.get('probabilities')
    if answer not in choices:answer=None;issues.append('unregistered_answer')
    valid=isinstance(distribution,dict) and set(distribution)==set(choices)
    if valid:valid=all(type(v) in [int,float] and isfinite(v) and 0<=v<=1 for v in distribution.values())
    if valid:valid=abs(sum(distribution.values())-1)<=.001
    if valid and answer is not None:valid=distribution[answer]>=max(distribution.values())-1e-9
    if answer is None:valid=False
    if not valid:distribution=None;issues.append('invalid_probability_distribution')
    if set(obj)!={'answer','probabilities'}:issues.append('extra_or_missing_fields')
    return {'answer':answer,'probabilities':distribution,'issues':issues}


def features(reports,choices):
    valid=all(r['probabilities'] is not None for r in reports)
    values=[r['answer'] for r in reports]
    result={'disagreement':upgrade(values),'probabilities_valid':valid}
    if not valid:return dict(result,ensemble_uncertainty=None,individual_uncertainty=None)
    mean={c:sum(r['probabilities'][c] for r in reports)/3 for c in choices}
    result['ensemble_uncertainty']=1-max(mean.values())
    result['individual_uncertainty']=sum(1-max(r['probabilities'].values()) for r in reports)/3
    return result


def select(cases,observations):
    fs={c['id']:features(observations[c['id']],c['choices']) for c in cases}
    gate=[c['id'] for c in cases if fs[c['id']]['disagreement']];quota=len(gate)
    tie=lambda cid:p.digest(['probability-routing-v1',cid])
    policies={'disagreement':sorted(gate)}
    for name,feature in [('probability','ensemble_uncertainty'),('individual_uncertainty','individual_uncertainty')]:
        # Missing distributions are a logged failure fallback, never silently normalized/imputed.
        ranked=sorted(fs,key=lambda cid:(fs[cid]['probabilities_valid'],
            -(fs[cid][feature] if fs[cid][feature] is not None else 0),tie(cid)))
        policies[name]=sorted(ranked[:quota])
    policies['fixed']=sorted(sorted(fs,key=tie)[:quota])
    policies['length']=sorted(c['id'] for c in sorted(cases,key=lambda c:(-len(c['question']),tie(c['id'])))[:quota])
    return {'quota_questions':quota,'features':fs,'policies':policies}


def key(case,agent,backend):return p.digest(case)[:20]+'/'+backend+'/'+str(agent)


def collect():
    cases=p.prepare()
    with lock(OUT):
        freeze(OUT/'protocol.json',{'version':'probability-routing-v1','cases_hash':p.digest(cases),
            'code_hash':p.digest(Path(__file__).read_text()),'scope':'opened-development exploratory signal test, NOT game or independent validation',
            'max_attempts':144,'agents':3,'cap':600,'temperature':.3,
            'primary':'ensemble probability uncertainty versus disagreement at equal whole-panel upgrade count',
            'secondary':['mean individual uncertainty','fixed','random expectation','length','full high','full low'],
            'decision_order':'collect all low first; freeze selections BEFORE any high response; labels only after collection',
            'probability':'self-reported distribution, not token likelihood or guaranteed calibrated posterior',
            'failure':'no repairs or retries; unknown low transport -> invalid distribution fallback; unknown outcomes bounded',
            'no_fit':True,'no_domain_specific_rules':True,'no_rewards':True,
            'cost':'joint answer and probabilities in same call; all initial low attempts charged; strong calls same quota',
            'limits':['24 reused cases','one draw per path','probability interface differs from prior AEIC studies',
                'invalid probabilities prioritized only as failure fallback and separately reported']})
        freeze(OUT/'public_cases.json',cases)
        calls=CallJournal(OUT,p.transport,144)
        unresolved={a['key'] for a in read_rows(OUT/'attempts.jsonl')}-{r['key'] for r in read_rows(OUT/'responses.jsonl')}
        observations={}
        for backend in ['zhipu_flash','deepseek']:
            for case in cases:
                reports=[]
                for agent in range(3):
                    k=key(case,agent,backend)
                    if k in unresolved:reports.append({'answer':None,'probabilities':None,'issues':['unknown_transport']});continue
                    try:row=calls.call(k,request(case,agent,backend))
                    except RuntimeError as exc:
                        if str(exc).startswith('transport failed;'):
                            unresolved.add(k);reports.append({'answer':None,'probabilities':None,'issues':['unknown_transport']});continue
                        raise
                    body=p.body(row);reports.append(decode(body['choices'][0]['message']['content'],case['choices']))
                if backend=='zhipu_flash':observations[case['id']]=reports
                print('saved',backend,case['id'],flush=True)
            if backend=='zhipu_flash':freeze(OUT/'selections.json',select(cases,observations))
    return analyze()


def auc(scores,labels):
    positive=[s for s,y in zip(scores,labels) if y];negative=[s for s,y in zip(scores,labels) if not y]
    if not positive or not negative:return None
    return sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))


def analyze():
    cases=json.loads((OUT/'public_cases.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    if protocol['code_hash']!=p.digest(Path(__file__).read_text()) or protocol['cases_hash']!=p.digest(cases):raise ValueError('changed protocol')
    attempts=read_rows(OUT/'attempts.jsonl');rows=read_rows(OUT/'responses.jsonl');index={r['key']:r for r in rows}
    if len(attempts)!=144 or len({a['key'] for a in attempts})!=144 or len(index)!=len(rows):raise ValueError('incomplete or duplicate')
    def forbidden(_):raise AssertionError('offline analysis cannot call API')
    journal=CallJournal(OUT,forbidden,144);decoded={};tokens={};models={};issues=[]
    for backend in ['zhipu_flash','deepseek']:
        decoded[backend]={};tokens[backend]={}
        for case in cases:
            reports=[];cost=0
            for agent in range(3):
                k=key(case,agent,backend)
                if k not in index:reports.append({'answer':None,'probabilities':None,'issues':['unknown_transport']});continue
                row=journal.call(k,request(case,agent,backend));body=p.body(row)
                report=decode(body['choices'][0]['message']['content'],case['choices']);reports.append(report)
                cost+=body['usage']['total_tokens'];models.setdefault(backend,set()).add(body['model'])
                if report['issues']:issues.append({'id':case['id'],'agent':agent,'backend':backend,'issues':report['issues']})
            decoded[backend][case['id']]=reports;tokens[backend][case['id']]=cost
    selections=json.loads((OUT/'selections.json').read_text())
    if selections!=select(cases,decoded['zhipu_flash']):raise ValueError('selection commitment mismatch')
    # Check no high request began before the complete low phase and selection file existed.
    low_end=max(r['completed_utc'] for r in rows if r['request']['backend']=='zhipu_flash')
    if any(a['started_utc']<low_end for a in attempts if '/deepseek/' in a['key']):raise ValueError('high observed before low phase complete')
    labels={r['id']:r['label'] for r in json.loads((p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    policies=selections['policies'];n=len(cases);quota=selections['quota_questions'];metrics=[];calibration={b:[] for b in decoded}
    for case in cases:
        cid=case['id'];low=decoded['zhipu_flash'][cid];high=decoded['deepseek'][cid]
        # Invalid generated answers are actual unusable output; only missing transport is unknown.
        answers=[[r['answer'] or 'abstain' for r in rs] for rs in [low,high]]
        unknown=[(b,a) for b,backend in enumerate(['zhipu_flash','deepseek']) for a in range(3) if key(case,a,backend) not in index]
        variants=[]
        for latent in product([*case['choices'],'abstain'],repeat=len(unknown)):
            states=[list(v) for v in answers]
            for (b,a),value in zip(unknown,latent):states[b][a]=value
            lo=int(vote(states[0],case['choices'])==labels[cid]);hi=int(vote(states[1],case['choices'])==labels[cid])
            record={'all_low':lo,'all_high':hi,'random_expected':lo*(1-quota/n)+hi*quota/n}
            for name,chosen in policies.items():record[name]=hi if cid in chosen else lo
            record['upgrade_gain']=hi-lo;variants.append(record)
        bounds={k:[min(v[k] for v in variants),max(v[k] for v in variants)] for k in variants[0]}
        metrics.append({'id':cid,'features':selections['features'][cid],'correct_bounds':bounds})
        for backend,reports in [('zhipu_flash',low),('deepseek',high)]:
            for r in reports:
                if r['probabilities'] is not None:
                    brier=sum((r['probabilities'][c]-int(c==labels[cid]))**2 for c in case['choices'])
                    calibration[backend].append({'id':cid,'brier':brier,'uniform_brier':1-1/len(case['choices'])})
    counts={k:[sum(r['correct_bounds'][k][j] for r in metrics) for j in [0,1]] for k in metrics[0]['correct_bounds'] if k!='upgrade_gain'}
    paired={}
    for name,selected in policies.items():
        selected_rows=[r for r in metrics if r['id'] in selected]
        paired[name]={'corrected_bounds':[sum(r['correct_bounds']['all_low'][1-j]==0 and r['correct_bounds']['all_high'][j]==1 for r in selected_rows) for j in [0,1]],
            'harm_bounds':[sum(r['correct_bounds']['all_low'][j]==1 and r['correct_bounds']['all_high'][1-j]==0 for r in selected_rows) for j in [0,1]]}
    eligible=[r for r in metrics if r['features']['probabilities_valid'] and r['correct_bounds']['upgrade_gain'][0]==r['correct_bounds']['upgrade_gain'][1]]
    gains=[r['correct_bounds']['upgrade_gain'][0]>0 for r in eligible]
    diagnostic={'eligible_questions':len(eligible),'gain_positive_questions':sum(gains),
        'ensemble_uncertainty_gain_auc':auc([r['features']['ensemble_uncertainty'] for r in eligible],gains),
        'individual_uncertainty_gain_auc':auc([r['features']['individual_uncertainty'] for r in eligible],gains)}
    costs={name:sum(tokens['zhipu_flash'].values())+sum(tokens['deepseek'][cid] for cid in chosen) for name,chosen in policies.items()}
    costs.update(all_low=sum(tokens['zhipu_flash'].values()),all_high=sum(tokens['deepseek'].values()),
        random_expected=sum(tokens['zhipu_flash'].values())+quota/n*sum(tokens['deepseek'].values()))
    result={'scope':protocol['scope'],'new_attempts':len(attempts),'successful_responses':len(rows),'unknown':len(attempts)-len(rows),
        'quota_questions':quota,'routing_low_attempts':72,'routing_high_calls':quota*3,'selector_extra_calls':0,
        'correct_count_bounds':counts,'selected_corrections_harms':paired,'gain_identification':diagnostic,
        'valid_probability_slots':{b:len(v) for b,v in calibration.items()},
        'brier_descriptive':{b:{'mean':sum(r['brier'] for r in rs)/len(rs) if rs else None,
            'uniform_mean_same_slots':sum(r['uniform_brier'] for r in rs)/len(rs) if rs else None} for b,rs in calibration.items()},
        'known_token_costs':costs,'actual_models':{b:sorted(v) for b,v in models.items()},
        'collection_total_known_tokens':sum(tokens[b][cid] for b in tokens for cid in tokens[b]),
        'issues':issues,'rows':metrics,'response_hash':p.digest(rows),
        'limitations':['self-reported probabilities not proved calibrated','one saved sample','opened-data diagnosis',
            'equal high count not equal total money','Brier slot scores not independent question observations']}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','issues']},ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:collect()
    else:analyze()
