"""Same-model explicit thinking-mode control, not a third heterogeneous family.

Uses official 2026 API mode switch. No benchmark outputs/labels needed to choose
the two fixed arms. Equal token cap is not equal realized inference computation.
"""
import json
from pathlib import Path
import requests
from experiments.mechanism import study_two_backend_complementarity as parent
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows
from experiments.mechanism.run_premise_review import parse_row

OUT=parent.OUT.with_name('explicit_thinking_control_v1')
MODES=('disabled','enabled')


def transport(req):
    from experiments.core.llm import LLM_BACKENDS
    cfg=LLM_BACKENDS['deepseek'];base=cfg['base_url'].rstrip('/')
    url=base+'/chat/completions' if base.endswith(('/v1','/v2','/v3','/v4')) else base+'/v1/chat/completions'
    r=requests.post(url,headers={'Authorization':'Bearer '+cfg['api_key'],'Content-Type':'application/json'},
                    json={k:v for k,v in req.items() if k!='backend'},timeout=(10,180))
    return {'http_status':r.status_code,'body':r.text}


def request(case,mode):
    if mode not in MODES:raise ValueError('unknown mode')
    req={'backend':'deepseek','model':'deepseek-flash','max_tokens':4096,
         'thinking':{'type':mode},'response_format':{'type':'json_object'},
         'messages':[{'role':'system','content':parent.design.INSTRUCTION},
                     {'role':'user','content':json.dumps({'question':case['question'],'choices':case['choices']},sort_keys=True)}]}
    if mode=='enabled':req['reasoning_effort']='high'
    # Omit temperature: official API says thinking ignores it; do not claim matched sampling randomness.
    return req


def collect():
    from experiments.core.llm import LLM_BACKENDS
    if not LLM_BACKENDS['deepseek']['api_key']:raise ValueError('missing API config')
    with lock(OUT):
        cases=json.loads((parent.OUT/'public_cases.json').read_text())
        freeze(OUT/'public_cases.json',cases)
        freeze(OUT/'protocol.json',{'version':'explicit-thinking-control-v1','n':36,'task_calls':72,'attempt_limit':144,
           'model':'deepseek-flash','base_url':LLM_BACKENDS['deepseek']['base_url'],
           'modes':list(MODES),'max_tokens_both':4096,'temperature':'omitted, thinking does not support matched temperature',
           'purpose':'ability/resource baseline; not cross-family information diversity, not a new consensus algorithm',
           'sampling':'same 36 inputs as two-backend study; no selection by its scores',
           'official_docs':'https://api-docs.deepseek.com/guides/thinking_mode/',
           'code_sha256':parent.design.fingerprint(Path(__file__)),'cases_hash':parent.design.p.digest(cases),
           'limits':['mode and realized token compute differ; not pure reasoning-text causal effect',
                     'same question input as main study, not a second independent replication',
                     'truncation retained; no parsing private reasoning_content as final answer']})
        journal=CallJournal(OUT,transport,144);outputs={}
        for i,case in enumerate(cases):
            slots={}
            for mode in sorted(MODES,key=lambda m:parent.design.p.digest([case['id'],m,'thinking-order'])):
                req=request(case,mode);key=case['id']+'/'+mode
                r=journal.call(key,req)
                if r['response']['http_status']==429 or r['response']['http_status']>=500:
                    r=journal.call(key+'/retry1',req)
                parsed=parse_row(r,case['choices'])
                if parsed['http_status']!=200:raise RuntimeError('thinking-mode API failure HTTP '+str(parsed['http_status']))
                b=json.loads(r['response']['body']);message=b['choices'][0]['message']
                parsed['reasoning_content_present']=bool(message.get('reasoning_content'))
                parsed['usage_details']=b.get('usage',{})
                slots[mode]=parsed
            outputs[case['id']]=slots
            print('explicit thinking pair saved',i+1,'/36',flush=True)
        freeze(OUT/'sealed_outputs.json',outputs)
        labels={r['id']:r['label'] for r in json.loads((parent.OUT/'private_labels.json').read_text())}
        result={'scope':'single-call resource/ability control, not a new mechanism','task_calls':72,'results':{},
                'models_returned':sorted({r['model'] for s in outputs.values() for r in s.values() if r['model']})}
        for split in ['calibration','evaluation','all']:
            subset=[c for c in cases if split=='all' or c['split']==split];cell={}
            for mode in MODES:
                rs=[outputs[c['id']][mode] for c in subset]
                cell[mode]={'n':len(subset),'correct':sum(outputs[c['id']][mode]['answer']==labels[c['id']] for c in subset),
                    'invalid':sum(not r['valid'] for r in rs),'truncated':sum(r['finish_reason']=='length' for r in rs),
                    'reasoning_content_present':sum(r['reasoning_content_present'] for r in rs),
                    'tokens':sum(r['tokens'] for r in rs) if all(r['tokens'] is not None for r in rs) else None,
                    'calls':len(rs),'sum_http_seconds':sum(r['wall_seconds'] for r in rs)}
            cell['enabled_vs_disabled']={
                'corrections':sum(outputs[c['id']]['disabled']['answer']!=labels[c['id']] and outputs[c['id']]['enabled']['answer']==labels[c['id']] for c in subset),
                'harms':sum(outputs[c['id']]['disabled']['answer']==labels[c['id']] and outputs[c['id']]['enabled']['answer']!=labels[c['id']] for c in subset)}
            result['results'][split]=cell
        rows=read_rows(OUT/'responses.jsonl')
        result.update(actual_http_attempts=len(rows),response_journal_hash=parent.design.p.digest(rows))
        freeze(OUT/'result.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':collect()
