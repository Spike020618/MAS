"""Bounded API capability probe, NOT an accuracy experiment or calibration.

Four synthetic requests per permitted backend at most. Raw requests/responses
are journaled; ambiguous transport outcomes stop. No automatic retries,
candidate repairs, hidden normalization, label files, or Qwen calls.
"""
import argparse
from hashlib import sha256
import json
from math import exp, isfinite
from pathlib import Path
import requests

from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/resource_allocation/signaling_interface_probe_v1'
BACKENDS=('deepseek','zhipu_flash')


def request(backend, model, slot):
    if backend not in BACKENDS or slot not in ('generate_positive','generate_negative','verify_A','verify_B'):
        raise ValueError('unregistered backend or slot')
    if slot.startswith('generate'):
        objective='correct' if slot=='generate_positive' else 'incorrect'
        instruction='Choose the '+objective+' option. Output exactly one character, A or B, and nothing else.'
        prompt='What is 2 + 2?\nA: 4\nB: 5'
    else:
        instruction='Assess the proposed answer. Output exactly A for supported or B for unsupported, nothing else.'
        prompt='Question: What is 2 + 2?\nProposed answer: '+('4' if slot=='verify_A' else '5')
    req={'backend':backend,'model':model,'max_tokens':8,'temperature':0,
         'logprobs':True,'top_logprobs':20,
         'messages':[{'role':'system','content':instruction},{'role':'user','content':prompt}]}
    if backend=='deepseek':req['thinking']={'type':'disabled'}
    return req


def transport(req):
    from experiments.core.llm import LLM_BACKENDS
    cfg=LLM_BACKENDS[req['backend']];base=cfg['base_url'].rstrip('/')
    url=base+'/chat/completions' if base.endswith(('/v1','/v2','/v3','/v4')) else base+'/v1/chat/completions'
    response=requests.post(url,headers={'Authorization':'Bearer '+cfg['api_key'],'Content-Type':'application/json'},
                           json={k:v for k,v in req.items() if k!='backend'},timeout=(10,60))
    return {'http_status':response.status_code,'body':response.text}


def extract(row):
    """Only exact A/B first-token masses; no confidence-text substitution."""
    response=row['response'];status=response['http_status']
    result={'http_status':status,'usable':False,'tokens':None,'reason':'http_failure'}
    if status!=200:return result
    try:
        body=json.loads(response['body']);choice=body['choices'][0]
        usage=body.get('usage',{})
        result.update(returned_model=body.get('model'),system_fingerprint=body.get('system_fingerprint'),
                      tokens=usage.get('total_tokens'),finish_reason=choice.get('finish_reason'))
        content=choice.get('message',{}).get('content')
        if choice.get('finish_reason')!='stop' or content not in ('A','B'):
            result['reason']='not_exact_complete_single_character';return result
        positions=(choice.get('logprobs') or {}).get('content') or []
        if len(positions)!=1 or positions[0].get('token')!=content:
            result['reason']='missing_or_non_single_token_logprobs';return result
        tokens=positions[0].get('top_logprobs') or []
        masses={}
        for token in tokens:
            name=token.get('token');value=token.get('logprob')
            if name not in ('A','B'):continue
            if name in masses or type(value) not in (float,int) or not isfinite(value) or value>0:
                result['reason']='invalid_or_duplicate_candidate_probability';return result
            masses[name]=exp(value)
        if set(masses)!=set(('A','B')) or min(masses.values())<=0 or sum(masses.values())>1+1e-8:
            result['reason']='candidate_probability_missing_or_unrepresentable';return result
        total=sum(masses.values())
        result.update(usable=True,reason='conditional_exact_token_coding_only',raw_candidate_mass=masses,
                      candidate_mass_total=total,conditional_probabilities={k:v/total for k,v in masses.items()},
                      scope='conditional distribution over exact token A/B, not forced full-answer likelihood or factual calibration')
        return result
    except (KeyError,TypeError,ValueError,IndexError,OverflowError):
        result['reason']='malformed_response';return result


def run():
    from experiments.core.llm import LLM_BACKENDS
    with lock(OUT):
        freeze(OUT/'protocol.json',{'version':'signaling-interface-probe-v1','max_attempts':8,
            'backends':list(BACKENDS),'slots':['generate_positive','generate_negative','verify_A','verify_B'],
            'models':{b:LLM_BACKENDS[b]['model'] for b in BACKENDS},
            'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'purpose':'synthetic interface capability only; no efficacy or benchmark labels',
            'retry_policy':'none; stop backend on first unusable response; unknown transport stops entire run'})
        journal=CallJournal(OUT,transport,8);results={}
        for backend in BACKENDS:
            cfg=LLM_BACKENDS[backend]
            if not cfg['api_key']:
                results[backend]={'status':'missing_configuration','slots':{}};continue
            slots={}
            for slot in ('generate_positive','generate_negative','verify_A','verify_B'):
                row=journal.call(backend+'/'+slot,request(backend,cfg['model'],slot))
                slots[slot]=extract(row)
                if not slots[slot]['usable']:break
            results[backend]={'status':'conditional_token_interface_available' if len(slots)==4 and all(s['usable'] for s in slots.values()) else 'blocked',
                              'slots':slots}
        rows=read_rows(OUT/'responses.jsonl')
        known=[extract(r)['tokens'] for r in rows]
        result={'backends':results,'http_attempts':len(read_rows(OUT/'attempts.jsonl')),
                'known_tokens_lower_bound':sum(v for v in known if type(v) is int and v>=0),
                'unknown_usage_responses':sum(type(v) is not int or v<0 for v in known),
                'sum_http_seconds':sum(r['wall_seconds'] for r in rows),
                'scope':'capability probe only; no benchmark performance, calibrated beliefs or original full-sequence scoring replication'}
        freeze(OUT/'result.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:print('Probe requires explicit --run; maximum8 paid HTTP attempts, no Qwen or benchmark calls.')
