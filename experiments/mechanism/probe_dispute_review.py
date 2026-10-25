"""Sealed independent answer -> ordinary or dispute-aware review. No reward tuning."""
import argparse
from itertools import product
import json
from pathlib import Path

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism import probe_aeic_context as prior
from experiments.mechanism.replay_low_cost_router import vote
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows

OUT=p.ROOT/'experiments/results/resource_allocation/dispute_review_v1'
ARMS=['ordinary','targeted']
GUIDANCE=('Review your sealed independent answer against the ORIGINAL QUESTION. '
    'Its scope takes priority over narrower claims in any report. Your initial answer and all supplied '
    'claims may be wrong; neither agreement nor detailed wording is evidence. '
    'Check your own assumptions and any supplied opposing claims before deciding. '
    'Maintain, revise or abstain based on reasoning, not peer count. Do not claim external verification: '
    'no tools or external sources have been accessed. Return one JSON object with C first '
    '(registered choice or abstain), A and I strings, E a list of strings, and review: '
    '{decision: keep|revise|abstain, checks: [{ref: supplied reference ID, '
    'assessment: supported|unsupported|uncertain, reason: short string}], change_basis: short string, '
    'external_check: false}. Explain changes, or say no new basis. Keep under 180 words.')


def cards(item,initial):
    result=[]
    for peer in item['reports']:
        if peer['C'] is None or peer['C']=='abstain' or peer['C']==initial['C']:continue
        fields=peer['available_AEI'];claim=None;field=None
        for name in ['I','A','E']:
            value=fields.get(name)
            if isinstance(value,str) and value.strip():claim=value;field=name;break
            if isinstance(value,list) and value and isinstance(value[0],str) and value[0].strip():
                claim=value[0];field=name+'[0]';break
        if claim is not None:
            result.append({'ref':f"peer:{peer['agent']}:{field}",'peer_C':peer['C'],
                'verbatim_claim':claim,'status':'unverified model claim'})
    return result


def prepare():
    items=json.loads((prior.OUT/'inputs.json').read_text())
    rows=read_rows(prior.OUT/'responses.jsonl');index={r['key']:r for r in rows}
    source=json.loads((prior.OUT/'result.json').read_text())
    if p.digest(rows)!=source['response_hash']:raise ValueError('prior source changed')
    prepared=[]
    for item in items:
        initial=[]
        for agent in range(3):
            row=index[prior.key(item,agent,'independent')]
            content=p.body(row)['choices'][0]['message']['content']
            decoded=p.decode_content(content,item['case']['choices'])
            if not decoded['aeic_complete']:raise ValueError('initial answer must be available, not fabricated')
            initial.append({'report':decoded['available_fields'],'response_hash':row['response_hash'],
                'source_key':row['key'],'source_request_hash':row['request_hash']})
        prepared.append({'case':item['case'],'group':item['group'],'initial':initial,
            'cards':[cards(item,r['report']) for r in initial]})
    return prepared,source['response_hash']


def request(item,agent,arm):
    if arm not in ARMS:raise ValueError('unregistered arm')
    from experiments.core.llm import AGENT_PERSONAS,LLM_BACKENDS
    initial=item['initial'][agent]['report']
    payload={'question':item['case']['question'],'choices':item['case']['choices'],
        'sealed_initial_report':initial,'self_reference_ids':['self:A','self:E','self:I','self:C']}
    if arm=='targeted':payload['dispute_cards']=item['cards'][agent]
    return {'backend':'deepseek','model':LLM_BACKENDS['deepseek']['model'],'temperature':.3,'max_tokens':600,
        'messages':[{'role':'system','content':AGENT_PERSONAS[agent]['system']+'\n'+GUIDANCE},
                    {'role':'user','content':json.dumps(payload,sort_keys=True,ensure_ascii=False)}],
        'response_format':{'type':'json_object'}}


def key(item,agent,arm):return p.digest(item['case'])[:20]+'/'+arm+'/'+str(agent)


def review_status(content,refs):
    try:
        obj=json.loads(content);r=obj['review']
        valid=(r['decision'] in ['keep','revise','abstain'] and type(r['external_check']) is bool
               and r['external_check'] is False and isinstance(r['change_basis'],str)
               and isinstance(r['checks'],list)
               and all(isinstance(c,dict) and c.get('ref') in refs
                   and c.get('assessment') in ['supported','unsupported','uncertain']
                   and isinstance(c.get('reason'),str) for c in r['checks']))
        return {'valid':bool(valid),'review':r,'issues':[] if valid else ['invalid_review_schema_or_reference']}
    except (ValueError,TypeError,KeyError):return {'valid':False,'review':None,'issues':['missing_or_invalid_review']}


def run():
    items,source=prepare();budget=72
    if len(items)!=12:raise ValueError('fixed twelve cases required')
    with lock(OUT):
        freeze(OUT/'protocol.json',{'version':'dispute-review-v1','source_hash':source,'inputs_hash':p.digest(items),
            'code_hash':p.digest(Path(__file__).read_text()),'max_attempts':budget,
            'primary':'targeted minus ordinary on same seven valid-disagreement cases',
            'secondary':['all twelve','five anomalous cases','both review arms versus shared sealed baseline'],
            'scope':'opened-data exploratory protocol test, not independent validation or equilibrium',
            'initial':'36 existing independently generated strong reports reused as sealed shared baseline, not newly billed',
            'card_rule':'opposing usable C only; one verbatim field per peer, fixed I then A then E[0] priority',
            'cost':'both review arms 36 new calls each; same model cap600 temperature.3; charge differing input',
            'failure':'no retry, no repair; transport unknown retained and bounded; HTTP/non-stop stops',
            'no_claims':['fact checking achieved','four-field necessity','new game contribution']})
        freeze(OUT/'inputs.json',items)
        calls=CallJournal(OUT,p.transport,budget)
        unknown={a['key'] for a in read_rows(OUT/'attempts.jsonl')}-{r['key'] for r in read_rows(OUT/'responses.jsonl')}
        for item in items:
            for agent in range(3):
                for arm in sorted(ARMS,key=lambda a:p.digest([item['case']['id'],agent,a,'dispute-review-v1'])):
                    k=key(item,agent,arm)
                    if k in unknown:continue
                    try:row=calls.call(k,request(item,agent,arm))
                    except RuntimeError as exc:
                        if str(exc).startswith('transport failed;'):
                            unknown.add(k);print('unknown transport, not retried',k,flush=True);continue
                        raise
                    p.body(row)
            print('both reviews saved',item['case']['id'],flush=True)
    return analyze()


def analyze():
    items=json.loads((OUT/'inputs.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    if protocol['code_hash']!=p.digest(Path(__file__).read_text()) or protocol['inputs_hash']!=p.digest(items):
        raise ValueError('protocol changed')
    attempts=read_rows(OUT/'attempts.jsonl');rows=read_rows(OUT/'responses.jsonl');index={r['key']:r for r in rows}
    if len(attempts)!=72 or len({a['key'] for a in attempts})!=72 or len(index)!=len(rows):raise ValueError('incomplete or duplicate')
    def forbidden(_):raise AssertionError('offline scoring')
    journal=CallJournal(OUT,forbidden,72)
    for item in items:
        for agent in range(3):
            for arm in ARMS:
                k=key(item,agent,arm)
                if k in index:journal.call(k,request(item,agent,arm))
    labels={r['id']:r['label'] for r in json.loads((p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    usage={a:{'calls':0,'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,'wall_seconds':0.,
        'valid_audit':0,'changed_C':0,'changed_without_declared_basis':0,'models':set()} for a in ARMS}
    metrics=[]
    for item in items:
        case=item['case'];initial=[r['report']['C'] for r in item['initial']]
        initial_correct=int(vote(initial,case['choices'])==labels[case['id']])
        scores={'stop':[initial_correct,initial_correct]};details={}
        for arm in ARMS:
            values=[];unknown=[];details[arm]=[]
            for agent in range(3):
                row=index.get(key(item,agent,arm))
                if row is None:values.append(None);unknown.append(agent);details[arm].append({'transport':'unknown'});continue
                body=p.body(row);content=body['choices'][0]['message']['content']
                decoded=p.decode_content(content,case['choices']);values.append(decoded['conclusion'])
                refs=['self:A','self:E','self:I','self:C']+([c['ref'] for c in item['cards'][agent]] if arm=='targeted' else [])
                audit=review_status(content,refs);u=usage[arm]
                u['calls']+=1;u['wall_seconds']+=row['wall_seconds'];u['models'].add(body['model']);u['valid_audit']+=int(audit['valid'])
                change=decoded['conclusion']!=initial[agent];u['changed_C']+=int(change)
                basis=audit['review'].get('change_basis','') if isinstance(audit['review'],dict) else ''
                u['changed_without_declared_basis']+=int(change and not basis.strip())
                for k in ['prompt_tokens','completion_tokens','total_tokens']:u[k]+=body['usage'][k]
                details[arm].append({'decoded':decoded,'audit':audit})
            candidates=[]
            for possible in product([*case['choices'],'abstain'],repeat=len(unknown)):
                complete=list(values)
                for a,c in zip(unknown,possible):complete[a]=c
                candidates.append(int(vote(complete,case['choices'])==labels[case['id']]))
            scores[arm]=[min(candidates),max(candidates)]
        metrics.append({'id':case['id'],'group':item['group'],'initial_C':initial,'correct_bounds':scores,'outputs':details})
    groups={}
    for group in ['valid_disagreement','invalid_or_abstain','all']:
        subset=[r for r in metrics if group=='all' or r['group']==group]
        summary={'n':len(subset),'correct_bounds':{arm:[sum(r['correct_bounds'][arm][j] for r in subset) for j in [0,1]] for arm in ['stop',*ARMS]},'paired':{}}
        for left,right in [('ordinary','targeted'),('stop','ordinary'),('stop','targeted')]:
            if any(r['correct_bounds'][a][0]!=r['correct_bounds'][a][1] for r in subset for a in [left,right]):
                summary['paired'][left+'->'+right]={'unknown':True};continue
            summary['paired'][left+'->'+right]={
                'corrections':sum(r['correct_bounds'][left][0]==0 and r['correct_bounds'][right][0]==1 for r in subset),
                'harms':sum(r['correct_bounds'][left][0]==1 and r['correct_bounds'][right][0]==0 for r in subset)}
        groups[group]=summary
    for u in usage.values():u['models']=sorted(u['models'])
    result={'scope':protocol['scope'],'new_attempts':len(attempts),'responses':len(rows),'unknown':len(attempts)-len(rows),
        'groups':groups,'usage':usage,'rows':metrics,'response_hash':p.digest(rows),
        'card_count':sum(len(cs) for i in items for cs in i['cards']),
        'empty_card_seats':sum(not cs for i in items for cs in i['cards']),
        'audit_limit':'declared reasoning, not verified fact or proof of causal usefulness'}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:analyze()
