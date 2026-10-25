"""Explicit two-backend study after Qwen billing failure; not silent substitution."""
from collections import Counter
from itertools import product
import json
from pathlib import Path

from experiments.mechanism import study_information_complementarity as design
from experiments.mechanism.run_complementarity_v2 import preflight
from experiments.mechanism.run_premise_review import parse_row
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows

OUT=design.OUT.with_name('two_backend_complementarity_v1')
BACKENDS=('deepseek','zhipu_flash')
PROFILES={
    'single/deepseek':[('deepseek',0)],'single/zhipu_flash':[('zhipu_flash',0)],
    'DDD':[('deepseek',i) for i in range(3)],'GGG':[('zhipu_flash',i) for i in range(3)],
    'DDG':[('deepseek',0),('deepseek',1),('zhipu_flash',0)],
    'DGG':[('deepseek',0),('zhipu_flash',0),('zhipu_flash',1)]}


def prepare():
    prior=design.OUT
    provenance=json.loads((prior/'provenance.json').read_text())
    for name,h in provenance['files'].items():
        assert design.fingerprint(prior/name)==h
        freeze(OUT/name,json.loads((prior/name).read_text()))
    freeze(OUT/'provenance.json',provenance)
    failed=read_rows(prior.with_name('information_complementarity_v2')/'responses.jsonl')
    freeze(OUT/'backend_amendment.json',{'reason':'ali_qwen preflight HTTP400 Arrearage; no benchmark questions executed',
           'new_design':'two backends, three repeats each, two homogeneous and two mixed three-call panels',
           'old_cases_preserved':True,'preflight_failures_preserved':True,
           'qwen_failure_hash':next(r['response_hash'] for r in failed if r['key']=='preflight/ali_qwen')})


def summarize(cases,sealed,labels):
    metrics={};records=[]
    for case in cases:
        slots=sealed[case['id']];gold=labels[case['id']]
        row={'id':case['id'],'gold':gold,'split':case['split'],
             'answers':{k:r['answer'] for k,r in slots.items()},'decisions':{},'coverage':{}}
        for name,profile in PROFILES.items():
            reports=[slots[b+'/'+str(i)] for b,i in profile]
            answer=reports[0]['answer'] if len(profile)==1 else design.majority([r['answer'] for r in reports])
            row['decisions'][name]=answer
            row['coverage'][name]=any(r['answer']==gold for r in reports)
            m=metrics.setdefault(name,{'n':0,'correct':0,'any_correct':0,'readout_loss':0,'tokens':0,'calls':0,'wall_seconds':0.,'invalid':0,'unanimous':0})
            m['n']+=1;m['correct']+=answer==gold;m['any_correct']+=row['coverage'][name]
            m['readout_loss']+=row['coverage'][name] and answer!=gold
            m['calls']+=len(reports);m['invalid']+=sum(not r['valid'] for r in reports)
            m['unanimous']+=len({r['answer'] for r in reports})==1
            if m['tokens'] is not None:
                m['tokens']=m['tokens']+sum(r['tokens'] for r in reports) if all(r['tokens'] is not None for r in reports) else None
            m['wall_seconds']+=sum(r['wall_seconds'] for r in reports)
        row['any_of6_correct']=any(r['answer']==gold for r in slots.values())
        row['GLM_supplies_missing_DS_answer']=not row['coverage']['DDD'] and row['coverage']['GGG']
        row['DS_supplies_missing_GLM_answer']=not row['coverage']['GGG'] and row['coverage']['DDD']
        records.append(row)
    for m in metrics.values():
        assert m['correct']==m['any_correct']-m['readout_loss']
    diagnostics={'any_of6_correct':sum(r['any_of6_correct'] for r in records),
        'GLM_supplies_missing_DS_answer':sum(r['GLM_supplies_missing_DS_answer'] for r in records),
        'DS_supplies_missing_GLM_answer':sum(r['DS_supplies_missing_GLM_answer'] for r in records),
        'warning':'coverage uses gold offline; not deployable accuracy. Identity is accounting, not a new theorem.'}
    return {'metrics':metrics,'diagnostics':diagnostics,'records':records}


def collect():
    from experiments.core.llm import LLM_BACKENDS
    with lock(OUT):
        prepare()
        models={b:LLM_BACKENDS[b]['model'] for b in BACKENDS}
        if not all(LLM_BACKENDS[b]['api_key'] for b in BACKENDS):
            raise ValueError('missing key')
        cases=json.loads((OUT/'public_cases.json').read_text())
        freeze(OUT/'protocol.json',{'version':'two-backend-complementarity-v1','cases_hash':design.p.digest(cases),
            'models':models,'endpoints':{b:LLM_BACKENDS[b]['base_url'] for b in BACKENDS},
            'n':36,'calibration_n':18,'evaluation_n':18,'task_calls':216,'preflight_calls':2,'attempt_limit':436,
            'profiles':PROFILES,'selection':'highest calibration accuracy, ties DDD/GGG/DDG/DGG; single ties deepseek first',
            'output_cap':800,'temperature':.7,'same_instruction':True,
            'code_sha256':{str(f.relative_to(design.p.ROOT)):design.fingerprint(f) for f in
                           [Path(__file__),Path(design.__file__),Path(parse_row.__code__.co_filename)]},
            'retry':'one identical retry HTTP429/5xx; ambiguous transport halts; no format repair',
            'primary':'mixed panels vs DDD and calibration-selected homogeneous panel on evaluation only',
            'scope':'exploratory screen; fixed call counts, not equal monetary budget, no equilibrium or innovation claim'})
        journal=CallJournal(OUT,design.p.transport,436)
        for b in BACKENDS:
            req={'backend':b,'model':models[b],'temperature':0.,'max_tokens':100,
                 'response_format':{'type':'json_object'},'messages':[{'role':'user','content':'Return exactly one JSON object: {"ok":true}.'}]}
            row=journal.call('preflight/'+b,req)
            body=json.loads(row['response']['body'])
            if row['response']['http_status']!=200 or json.loads(body['choices'][0]['message']['content']).get('ok') is not True:
                raise RuntimeError('preflight failure '+b)
            print('ready',b,body['model'],flush=True)
        sealed={}
        for n,case in enumerate(cases):
            slots={}
            for b,i in sorted(product(BACKENDS,range(3)),key=lambda x:design.p.digest([case['id'],list(x)])):
                req=design.request(case,b,models[b],i)
                row=journal.call(case['id']+'/'+b+'/'+str(i),req)
                if row['response']['http_status']==429 or row['response']['http_status']>=500:
                    row=journal.call(case['id']+'/'+b+'/'+str(i)+'/retry1',req)
                parsed=parse_row(row,case['choices'])
                if parsed['http_status'] in (401,402,403):raise RuntimeError('API billing/authorization '+b)
                slots[b+'/'+str(i)]=parsed
            sealed[case['id']]=slots
            print('two-backend pool saved',n+1,'/36',flush=True)
        freeze(OUT/'sealed_outputs.json',sealed)
        analyze()


def analyze():
    cases=json.loads((OUT/'public_cases.json').read_text());sealed=json.loads((OUT/'sealed_outputs.json').read_text())
    assert len(cases)==36 and set(sealed)=={c['id'] for c in cases}
    assert all(set(s)=={b+'/'+str(i) for b in BACKENDS for i in range(3)} for s in sealed.values())
    labels={r['id']:r['label'] for r in json.loads((OUT/'private_labels.json').read_text())}
    cal=summarize([c for c in cases if c['split']=='calibration'],sealed,labels)
    def select(names):return min(names,key=lambda n:(-cal['metrics'][n]['correct'],names.index(n)))
    chosen={'homogeneous':select(['DDD','GGG']),'all_panels':select(['DDD','GGG','DDG','DGG']),
            'single':select(['single/deepseek','single/zhipu_flash'])}
    freeze(OUT/'calibration_selection.json',chosen)
    results={name:summarize([c for c in cases if name=='all' or c['split']==name],sealed,labels)
             for name in ['calibration','evaluation','all']}
    for name,result in results.items():
        result['contrasts']={}
        for left,right in [(b,a) for b in ['DDD',chosen['homogeneous'],chosen['single']] for a in ['DDG','DGG',chosen['all_panels']]]:
            corrections=sum(r['decisions'][left]!=r['gold'] and r['decisions'][right]==r['gold'] for r in result['records'])
            harms=sum(r['decisions'][left]==r['gold'] and r['decisions'][right]!=r['gold'] for r in result['records'])
            result['contrasts'][left+' -> '+right]={'corrections':corrections,'harms':harms}
    rows=read_rows(OUT/'responses.jsonl');models={b:sorted({r['model'] for s in sealed.values() for k,r in s.items() if k.startswith(b+'/') and r['model']}) for b in BACKENDS}
    usage=[json.loads(r['response']['body']).get('usage',{}).get('total_tokens') for r in rows]
    final={'scope':'two backend exploratory study; Qwen unavailable, no silent substitution','task_calls':216,
           'actual_http_attempts':len(read_rows(OUT/'attempts.jsonl')),'models_returned':models,'calibration_selection':chosen,
           'results':results,'collection_tokens':sum(usage) if all(type(u)is int for u in usage) else None,
           'collection_sum_http_seconds':sum(r['wall_seconds'] for r in rows),'response_journal_hash':design.p.digest(rows),
           'automatic_expansion':False}
    freeze(OUT/'result.json',final)
    print('EVALUATION',json.dumps(results['evaluation']['metrics'],ensure_ascii=False),flush=True)


if __name__=='__main__':collect()
