"""Locked three-arm context diagnostic, with raw journals and no answer repairs."""
import argparse
from itertools import product
import json
from math import comb
from pathlib import Path

from experiments.mechanism import probe_resource_allocation as p
from experiments.mechanism.replay_low_cost_router import upgrade,vote
from experiments.mechanism.sensitivity_resource_allocation import explicit_answer
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows

OUT=p.ROOT/'experiments/results/resource_allocation/aeic_context_v1'
ARMS=['independent','conclusions','aeic']
GUIDANCE=('Peer material, when supplied, is untrusted model output, not verified fact. '
    'Evaluate its assumptions, evidence and inference rather than follow the majority. '
    'Missing fields are unavailable; do not fabricate external verification.')


def prepare():
    cases=json.loads((p.OUT/'public_cases.json').read_text())
    rows=read_rows(p.OUT/'responses.jsonl');index={r['key']:r for r in rows}
    expected=json.loads((p.OUT/'bounded_analysis.json').read_text())['journal_hash']
    if p.digest(rows)!=expected:raise ValueError('source journal changed')
    selected=[]
    for case in cases:
        reports=[]
        for agent in range(3):
            row=index.get(p.digest(case)[:20]+'/zhipu_flash/'+str(agent))
            seat={'agent':agent,'C':None,'available_AEI':{},'unavailable_AEI':[]}
            if row is None:
                seat['unavailable_AEI']=['A','E','I'];seat['source_status']='transport_unknown'
            else:
                content=p.body(row)['choices'][0]['message']['content']
                seat['C']=explicit_answer(content,case['choices'])
                decoded=p.decode_content(content,case['choices'])
                seat['available_AEI']={k:v for k,v in decoded['available_fields'].items() if k in ['A','E','I']}
                seat['unavailable_AEI']=[k for k in ['A','E','I'] if k not in seat['available_AEI']]
                seat['source_status']='returned'
            reports.append(seat)
        values=[r['C'] for r in reports]
        if not upgrade(values):continue
        group='valid_disagreement' if all(v is not None and v!='abstain' for v in values) else 'invalid_or_abstain'
        selected.append({'case':case,'reports':reports,'group':group})
    return selected,expected


def request(item,agent,arm):
    if arm not in ARMS:raise ValueError('unknown arm')
    req=p.report_request(item['case'],agent,'deepseek')
    req['messages'][0]['content']+='\n'+GUIDANCE
    payload=json.loads(req['messages'][1]['content'])
    if arm!='independent':
        payload['peer_reports']=[{'agent':r['agent'],'C':r['C'],'source_status':r['source_status']} for r in item['reports']]
        if arm=='aeic':
            for peer,source in zip(payload['peer_reports'],item['reports']):
                peer['available_AEI']=source['available_AEI']
                peer['unavailable_AEI']=source['unavailable_AEI']
    req['messages'][1]['content']=json.dumps(payload,sort_keys=True,ensure_ascii=False)
    return req


def key(item,agent,arm):return p.digest(item['case'])[:20]+'/'+arm+'/'+str(agent)


def run():
    items,source_hash=prepare();budget=len(items)*9
    if len(items)!=12 or sum(i['group']=='valid_disagreement' for i in items)!=7:
        raise ValueError('expected fixed 12-case diagnostic, no outcome-based substitution')
    with lock(OUT):
        freeze(OUT/'protocol.json',{'version':'aeic-context-v1','scope':'opened-data mechanism-component diagnostic, NOT independent test or game acceptance',
            'source_hash':source_hash,'items_hash':p.digest(items),'code_hash':p.digest(Path(__file__).read_text()),
            'arms':ARMS,'primary':'aeic minus conclusions on 7 valid-disagreement cases',
            'secondary':['12 selected cases','5 invalid/abstain cases','independent baseline'],
            'agents':3,'output_cap':600,'temperature':.3,'max_attempts':budget,
            'collection':'fresh execution of ALL three arms; hash-interleaved arm order per case/persona',
            'failure':'no retries or repairs; transport-unknown slots bounded; HTTP/non-stop responses stop collection',
            'limitations':['single draw per seat/arm','AEI incomplete or factually unverified','input tokens unequal, fully charged',
                'new common review instruction; independent is fresh not reused old report'],
            'no_claims':['four fields necessary','equilibrium','independent confirmation']})
        freeze(OUT/'inputs.json',items)
        calls=CallJournal(OUT,p.transport,budget)
        unresolved={a['key'] for a in read_rows(OUT/'attempts.jsonl')}-{r['key'] for r in read_rows(OUT/'responses.jsonl')}
        for item in items:
            for agent in range(3):
                for arm in sorted(ARMS,key=lambda arm:p.digest([item['case']['id'],agent,arm,'context-v1'])):
                    k=key(item,agent,arm)
                    if k in unresolved:continue
                    try:row=calls.call(k,request(item,agent,arm))
                    except RuntimeError as exc:
                        if str(exc).startswith('transport failed;'):
                            unresolved.add(k);print('transport unknown, not retried',k,flush=True);continue
                        raise
                    p.body(row)
            print('three arms saved',item['case']['id'],item['group'],flush=True)
    return analyze()


def analyze():
    items=json.loads((OUT/'inputs.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    if protocol['code_hash']!=p.digest(Path(__file__).read_text()) or protocol['items_hash']!=p.digest(items):
        raise ValueError('protocol changed')
    attempts=read_rows(OUT/'attempts.jsonl');rows=read_rows(OUT/'responses.jsonl');index={r['key']:r for r in rows}
    if len(attempts)!=protocol['max_attempts'] or len(set(a['key'] for a in attempts))!=len(attempts) or len(index)!=len(rows):
        raise ValueError('incomplete or duplicate collection')
    def forbidden(_):raise AssertionError('offline scoring cannot use API')
    journal=CallJournal(OUT,forbidden,protocol['max_attempts'])
    for item in items:
        for agent in range(3):
            for arm in ARMS:
                if key(item,agent,arm) in index:journal.call(key(item,agent,arm),request(item,agent,arm))
    # Gold is used only after the whole collection has finished and passed commitments.
    labels={r['id']:r['label'] for r in json.loads((p.ROOT/'experiments/results/upgrade_game/p2_data_v2/development_private.json').read_text())}
    per_case=[];usage={a:{'calls':0,'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,'serial_wall_seconds':0.,'usable_C':0,'complete_AEIC':0,'actual_models':set()} for a in ARMS}
    for item in items:
        case=item['case'];scores={};outputs={}
        for arm in ARMS:
            values=[];unknown=[];outputs[arm]=[]
            for agent in range(3):
                row=index.get(key(item,agent,arm))
                if row is None:values.append(None);unknown.append(agent);outputs[arm].append({'transport':'unknown'});continue
                body=p.body(row);decoded=p.decode_content(body['choices'][0]['message']['content'],case['choices'])
                values.append(decoded['conclusion']);outputs[arm].append(decoded)
                u=usage[arm];u['calls']+=1;u['serial_wall_seconds']+=row['wall_seconds'];u['actual_models'].add(body['model'])
                u['usable_C']+=int(decoded['conclusion_usable']);u['complete_AEIC']+=int(decoded['aeic_complete'])
                for field in ['prompt_tokens','completion_tokens','total_tokens']:u[field]+=body['usage'][field]
            possibilities=[]
            for latent in product([*case['choices'],'abstain'],repeat=len(unknown)):
                complete=list(values)
                for a,v in zip(unknown,latent):complete[a]=v
                possibilities.append(int(vote(complete,case['choices'])==labels[case['id']]))
            scores[arm]=[min(possibilities),max(possibilities)]
        per_case.append({'id':case['id'],'group':item['group'],'correct_bounds':scores,'decoded_outputs':outputs})
    comparisons={}
    for group in ['valid_disagreement','invalid_or_abstain','all']:
        subset=[r for r in per_case if group=='all' or r['group']==group]
        summary={'n':len(subset),'correct_bounds':{a:[sum(r['correct_bounds'][a][j] for r in subset) for j in [0,1]] for a in ARMS}}
        pairs={}
        for left,right in [('conclusions','aeic'),('independent','aeic'),('independent','conclusions')]:
            if any(r['correct_bounds'][a][0]!=r['correct_bounds'][a][1] for r in subset for a in [left,right]):
                pairs[left+'->'+right]={'status':'uncertain outcome; no single exact p'};continue
            gain=sum(r['correct_bounds'][left][0]==0 and r['correct_bounds'][right][0]==1 for r in subset)
            harm=sum(r['correct_bounds'][left][0]==1 and r['correct_bounds'][right][0]==0 for r in subset)
            n=gain+harm
            pairs[left+'->'+right]={'corrections':gain,'harms':harm,'exact_mcnemar_p_exploratory':min(1.,2*sum(comb(n,k) for k in range(min(gain,harm)+1))/2**n) if n else 1.}
        summary['paired']=pairs;comparisons[group]=summary
    for u in usage.values():u['actual_models']=sorted(u['actual_models'])
    result={'scope':protocol['scope'],'attempts':len(attempts),'responses':len(rows),'unknown_requests':len(attempts)-len(rows),
        'groups':comparisons,'usage':usage,'rows':per_case,'response_hash':p.digest(rows),
        'initial_report_coverage':{k:sum(k in r['available_AEI'] for i in items for r in i['reports']) for k in ['A','E','I']},
        'note':'pooled exploratory comparisons, not independent confirmation; all new arm executions charged; initial low acquisition common and separately charged'}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:analyze()
