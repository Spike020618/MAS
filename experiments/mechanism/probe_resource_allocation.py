"""Bounded model-only allocation diagnostic; no changes to frozen old studies."""
import argparse
from itertools import product
import json
from math import comb
from pathlib import Path

from experiments.core.mechanism.upgrade_game import digest
from experiments.core.mechanism.upgrade_protocol_v3 import decode_content, majority
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/resource_allocation/probe_v2'
PROFILES=list(product((0,1),repeat=3))
INSTRUCTION=('Answer independently using your knowledge. Do not invent external verification. '
    'Return exactly one concise JSON object, C first: C is one registered choice or abstain; '
    'A and I are short strings; E is a list of at most two short evidence strings. '
    'Keep the complete report under 150 words. No text outside JSON.')


def prepare():
    source=ROOT/'experiments/results/upgrade_game/p2_data_v2/development_public.json'
    if not source.exists():source=ROOT/'experiments/results/upgrade_game/p2_development_v2/public_cases.json'
    pool=json.loads(source.read_text()); chosen=[]
    for domain in ['strategyqa','mmlu','truthfulqa']:
        rows=sorted([r for r in pool if r['id'].startswith(domain+':')],key=lambda r:digest(['allocation-probe-v1',r['id']]))
        if len(rows)<8:raise ValueError('insufficient public diagnostic cases')
        chosen+=rows[:8]
    return sorted(chosen,key=lambda r:digest(['execution-order-v1',r['id']]))


def report_request(case,agent,backend):
    from experiments.core.llm import AGENT_PERSONAS,LLM_BACKENDS
    return {'backend':backend,'model':LLM_BACKENDS[backend]['model'],'temperature':.3,'max_tokens':600,
        'messages':[{'role':'system','content':AGENT_PERSONAS[agent]['system']+'\n'+INSTRUCTION},
                    {'role':'user','content':json.dumps({'question':case['question'],'choices':case['choices']},sort_keys=True)}],
        'response_format':{'type':'json_object'}}


def choice_request(case,agent):
    from experiments.core.llm import AGENT_PERSONAS,LLM_BACKENDS
    text=('Visible fixed process contract. This is the two-action restriction of the historical Static contract. '
        'low_effort: cost 0; aeic_effort: cost 1. Potential credit: complete AEIC +0.2; '
        'non-abstaining conclusion +0.2. These are virtual credit/cost units, not API money. '
        'Choose before solving to maximize expected credit minus stated cost; do not choose work merely '
        'because it is called AEIC. No reputation or peer information. Output only JSON: '
        '{"ACTION":"low_effort"} or {"ACTION":"aeic_effort"}.\nQuestion:\n'+case['question'])
    return {'backend':'deepseek','model':LLM_BACKENDS['deepseek']['model'],'temperature':.3,'max_tokens':100,
        'messages':[{'role':'system','content':AGENT_PERSONAS[agent]['system']},{'role':'user','content':text}],
        'response_format':{'type':'json_object'}}


def transport(req):
    import requests
    from experiments.core.llm import LLM_BACKENDS
    cfg=LLM_BACKENDS[req['backend']]
    if not cfg['api_key']:raise ValueError('API key missing for '+req['backend'])
    base=cfg['base_url'].rstrip('/')
    url=base+'/chat/completions' if base.endswith(('/v1','/v2','/v3','/v4')) else base+'/v1/chat/completions'
    payload={k:v for k,v in req.items() if k!='backend'}
    response=requests.post(url,headers={'Authorization':'Bearer '+cfg['api_key'],'Content-Type':'application/json'},
                           json=payload,timeout=(10,60))
    return {'http_status':response.status_code,'body':response.text}


def body(row):
    if row['response']['http_status']!=200:raise ValueError('HTTP failure preserved; no automatic retry')
    value=json.loads(row['response']['body'])
    if value['choices'][0]['finish_reason']!='stop':raise ValueError('unfinished response; stop without repairs')
    if not value.get('model'):raise ValueError('returned model missing')
    usage=value.get('usage',{})
    if any(type(usage.get(k)) is not int or usage[k]<0 for k in ['prompt_tokens','completion_tokens','total_tokens']):
        raise ValueError('usage metadata missing')
    return value


def random_profile_weight(n_slots,total_high,profile):
    k=sum(profile);remaining=total_high-k
    return comb(n_slots-3,remaining)/comb(n_slots,total_high) if 0<=remaining<=n_slots-3 else 0.


def fixed_profiles(cases,total_high):
    if not 0<=total_high<=3*len(cases):raise ValueError('invalid quota')
    result={};q,r=divmod(total_high,len(cases))
    for j,case in enumerate(cases):
        count=q+int(j<r);rotation=int(digest(['fixed-rotation',case['id']])[:8],16)%3
        result[case['id']]=tuple(int((a-rotation)%3<count) for a in range(3))
    return result


def run(directory=OUT):
    cases=prepare()
    with lock(directory):
        protocol={'version':'resource-allocation-diagnostic-v2','cases_hash':digest(cases),
            'engineering_amendment':'v1 first HTTP400 required literal JSON in selector prompt; no answers or labels read; one prior attempt retained',
            'question_status':'reused opened P2 development, NOT independent holdout',
            'agents':3,'low':'zhipu_flash','high':'deepseek','cap_both':600,'max_calls':216,
            'fixed_rewards':{'complete':.2,'nonabstain':.2},'fixed_costs':{'low':0,'high':1},
            'scope':'model-only execution bridge and two-action empirical contract restriction, NOT rational equilibrium',
            'primary_comparators':['all_low','all_high','same_question_k_random','global_quota_random_expectation','fixed_quota'],
            'limitations':['equal high execution count is not equal total API cost; selector overhead reported',
                           'single saved report per slot/path; no repeated-sampling variance estimate'],
            'code_hash':digest(Path(__file__).read_text())}
        freeze(directory/'protocol.json',protocol);freeze(directory/'public_cases.json',cases)
        calls=CallJournal(directory,transport,216)
        selections={}
        # Commit ALL choices before collecting the potential report pool, never read labels here.
        for case in cases:
            actions=[]
            for agent in range(3):
                key=digest(case)[:20]+'/choice/'+str(agent)
                value=body(calls.call(key,choice_request(case,agent)))
                parsed=json.loads(value['choices'][0]['message']['content'])
                if set(parsed)!= {'ACTION'} or parsed['ACTION'] not in ['low_effort','aeic_effort']:
                    raise ValueError('invalid action, no selective repair')
                actions.append(int(parsed['ACTION']=='aeic_effort'))
            selections[case['id']]=actions
            print('selection committed',case['id'],actions,flush=True)
        freeze(directory/'selections.json',selections)
        for case in cases:
            for agent in range(3):
                order=sorted(['zhipu_flash','deepseek'],key=lambda b:digest([case['id'],agent,b]))
                for backend in order:
                    row=calls.call(digest(case)[:20]+'/'+backend+'/'+str(agent),report_request(case,agent,backend))
                    value=body(row)
                    decoded=decode_content(value['choices'][0]['message']['content'],case['choices'])
                    if 'invalid_json_object' in decoded['issues']:
                        raise ValueError('invalid report JSON; stop, do not turn truncation into a quality effect')
            print('potential reports saved',case['id'],flush=True)
        # Only after every choice/report has been persisted may scoring read gold.
        return analyze(directory)


def analyze(directory=OUT):
    cases=json.loads((directory/'public_cases.json').read_text())
    selections=json.loads((directory/'selections.json').read_text())
    rows=read_rows(directory/'responses.jsonl');index={r['key']:r for r in rows}
    if len(index)!=len(rows) or len(rows)!=9*len(cases):raise ValueError('incomplete/duplicate report pool')
    protocol=json.loads((directory/'protocol.json').read_text())
    if protocol['cases_hash']!=digest(cases) or protocol['code_hash']!=digest(Path(__file__).read_text()):
        raise ValueError('protocol commitment mismatch')
    # Reuse the journal verifier without allowing network access during analysis.
    def forbidden(_):raise AssertionError('analysis must not issue network calls')
    journal=CallJournal(directory,forbidden,216)
    for case in cases:
        prefix=digest(case)[:20]
        for agent in range(3):
            value=body(journal.call(prefix+'/choice/'+str(agent),choice_request(case,agent)))
            action=json.loads(value['choices'][0]['message']['content'])['ACTION']
            if selections[case['id']][agent]!=int(action=='aeic_effort'):
                raise ValueError('selection does not match raw response')
            for backend in ['zhipu_flash','deepseek']:
                journal.call(prefix+'/'+backend+'/'+str(agent),report_request(case,agent,backend))
    labels_list=json.loads((ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())
    labels={r['id']:r['label'] for r in labels_list}
    metrics=[];slot_flows=[0,0];selector_tokens=0;models={};schema_issues=[]
    total_high=sum(sum(p) for p in selections.values());n=len(cases)*3
    fixed=fixed_profiles(cases,total_high)
    for case in cases:
        prefix=digest(case)[:20];pool=[[],[]];tokens=[[],[]]
        for agent in range(3):
            for high,backend in enumerate(['zhipu_flash','deepseek']):
                value=body(index[prefix+'/'+backend+'/'+str(agent)])
                decoded=decode_content(value['choices'][0]['message']['content'],case['choices'])
                pool[high].append(decoded);tokens[high].append(value['usage']['total_tokens'])
                models.setdefault(backend,set()).add(value['model'])
                if not decoded['aeic_complete']:schema_issues.append({'id':case['id'],'agent':agent,'backend':backend,'issues':decoded['issues']})
            a,b=pool[0][-1]['conclusion']==labels[case['id']],pool[1][-1]['conclusion']==labels[case['id']]
            slot_flows[0]+=int(not a and b);slot_flows[1]+=int(a and not b)
            value=body(index[prefix+'/choice/'+str(agent)])
            selector_tokens+=value['usage']['total_tokens']
        scores={p:int(majority([pool[a][i] for i,a in enumerate(p)],case['choices'])==labels[case['id']]) for p in PROFILES}
        chosen=tuple(selections[case['id']]);k=sum(chosen)
        same=[p for p in PROFILES if sum(p)==k]
        metrics.append({'id':case['id'],'all_low':scores[(0,0,0)],'all_high':scores[(1,1,1)],'contract':scores[chosen],
            'same_k_random_expected':sum(scores[p] for p in same)/len(same),
            'global_quota_random_expected':sum(scores[p]*random_profile_weight(n,total_high,p) for p in PROFILES),
            'fixed_quota':scores[fixed[case['id']]],'chosen_high':k,
            'contract_report_tokens':sum(tokens[a][i] for i,a in enumerate(chosen)),
            'fixed_report_tokens':sum(tokens[a][i] for i,a in enumerate(fixed[case['id']])),
            'random_report_tokens_expected':sum(sum(tokens[a][i] for i,a in enumerate(p))*random_profile_weight(n,total_high,p) for p in PROFILES),
            'low_tokens':sum(tokens[0]),'high_tokens':sum(tokens[1]),
            'realized_oracle':max(scores.values())})
    result={'scope':'small reused-development diagnostic, no independent-confirmatory claim',
        'n':len(cases),'calls_collected':len(rows),'high_report_quota':total_high,
        'correct_counts':{name:sum(r[name] for r in metrics) for name in ['all_low','all_high','contract','same_k_random_expected','global_quota_random_expected','fixed_quota','realized_oracle']},
        'report_slot_low_to_high':{'wrong_correct':slot_flows[0],'correct_wrong':slot_flows[1]},
        'cost':{'selector_calls':n,'selector_tokens':selector_tokens,
                'contract_report_calls':n,'contract_report_tokens':sum(r['contract_report_tokens'] for r in metrics),
                'contract_total_calls':2*n,'contract_total_tokens':selector_tokens+sum(r['contract_report_tokens'] for r in metrics),
                'fixed_report_calls':n,'fixed_report_tokens':sum(r['fixed_report_tokens'] for r in metrics),
                'random_report_calls':n,'random_report_tokens_expected':sum(r['random_report_tokens_expected'] for r in metrics),
                'collection_total_tokens':sum(body(row)['usage']['total_tokens'] for row in rows),
                'collection_serial_wall_seconds':sum(row['wall_seconds'] for row in rows),
                'all_low_report_tokens':sum(r['low_tokens'] for r in metrics),'all_high_report_tokens':sum(r['high_tokens'] for r in metrics)},
        'actual_returned_models':{b:sorted(v) for b,v in models.items()},'schema_issues':schema_issues,'rows':metrics,
        'decision':'diagnostic only; no automatic expansion, reward tuning or independent efficacy claim',
        'response_journal_hash':digest(rows)}
    freeze(directory/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    elif args.analyze:analyze()
    else:print(json.dumps({'selected_cases':len(prepare()),'max_calls':216,'out':str(OUT)}))
