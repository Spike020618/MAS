"""Frozen standard reliability baseline + one fresh18-case validation."""
import argparse
from pathlib import Path
import json

from experiments.core.mechanism.reliable_readout import calibrate,decide
from experiments.mechanism import study_information_complementarity as design
from experiments.mechanism import study_two_backend_complementarity as prior
from experiments.mechanism import study_explicit_thinking_control as thinking
from experiments.mechanism.run_upgrade_pilot import CallJournal,freeze,lock,read_rows
from experiments.mechanism.run_premise_review import parse_row

OUT=design.OUT.with_name('reliable_readout_v1')
DATA=design.OUT.with_name('reliable_readout_data_v1')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=json.loads((prior.OUT/'public_cases.json').read_text())
    sealed=json.loads((prior.OUT/'sealed_outputs.json').read_text())
    labels={r['id']:r['label'] for r in json.loads((prior.OUT/'private_labels.json').read_text())}
    rows=[{'id':c['id'],'choices':c['choices'],'gold':labels[c['id']],
           'reports':[{'backend':b,'answer':sealed[c['id']][b+'/'+str(i)]['answer']} for b in prior.BACKENDS for i in range(3)]}
          for c in cases if c['split']=='calibration']
    model=calibrate(rows,list(prior.BACKENDS))
    freeze(OUT/'calibration.json',{'model':model,'source_sha256':{str(f.relative_to(design.p.ROOT)):design.fingerprint(f) for f in
             [prior.OUT/'public_cases.json',prior.OUT/'private_labels.json',prior.OUT/'sealed_outputs.json']},
           'calibration_ids':[r['id'] for r in rows],'code_sha256':design.fingerprint(Path(calibrate.__code__.co_filename))})
    previous=design.OUT
    try:
        design.OUT=DATA
        with lock(DATA):design.prepare()
    finally:design.OUT=previous
    all_cases=json.loads((DATA/'public_cases.json').read_text())
    chosen=[c for c in all_cases if c['split']=='evaluation']
    assert len(chosen)==18 and not {c['id'] for c in chosen}&{r['id'] for r in rows}
    freeze(OUT/'public_cases.json',chosen)
    print('calibration frozen, fresh18 cases prepared; new API calls 0')


def transport(req):
    return thinking.transport(req) if req.get('thinking') else design.p.transport(req)


def run():
    from experiments.core.llm import LLM_BACKENDS
    with lock(OUT):
        cases=json.loads((OUT/'public_cases.json').read_text());cal=json.loads((OUT/'calibration.json').read_text())
        for rel,h in cal['source_sha256'].items():assert design.fingerprint(design.p.ROOT/rel)==h
        assert design.fingerprint(Path(calibrate.__code__.co_filename))==cal['code_sha256']
        old_protocol=json.loads((prior.OUT/'protocol.json').read_text())
        models={b:LLM_BACKENDS[b]['model'] for b in prior.BACKENDS}
        assert models==old_protocol['models'],'requested backend aliases changed; recalibration required'
        training_outputs=json.loads((prior.OUT/'sealed_outputs.json').read_text())
        allowed_models={b:{r['model'] for qid,s in training_outputs.items() if qid in cal['calibration_ids']
                          for key,r in s.items() if key.startswith(b+'/') and r['model']} for b in prior.BACKENDS}
        freeze(OUT/'protocol.json',{'version':'reliable-readout-v1','n':18,'task_calls':90,'max_attempts':180,
            'primary':'weighted DDG vs unweighted SAME DDG reports; all18 paired cases included',
            'controls':['single fast DeepSeek0','DDD vote','DDG vote','weighted DDG','single thinking DeepSeek'],
            'selection':'same fixed input-only hash order; new source groups; no domain/risk/error filtering',
            'models':models,'endpoints':{b:LLM_BACKENDS[b]['base_url'] for b in prior.BACKENDS},
            'calibration_returned_models':{b:sorted(v) for b,v in allowed_models.items()},
            'calibration_sha256':design.fingerprint(OUT/'calibration.json'),
            'cases_hash':design.p.digest(cases),'code_sha256':design.fingerprint(Path(__file__)),
            'scope':'standard baseline validation, not novel algorithm, equilibrium or equal-realized-cost comparison'})
        journal=CallJournal(OUT,transport,180);outputs={}
        for n,c in enumerate(cases):
            slots={}
            order=sorted(['D0','D1','D2','G0','T0'],key=lambda x:design.p.digest([c['id'],x,'reliable-order-v1']))
            for slot in order:
                if slot=='T0':req=thinking.request(c,'enabled')
                else:
                    b='deepseek' if slot.startswith('D') else 'zhipu_flash'
                    req=design.request(c,b,models[b],int(slot[1]))
                key=c['id']+'/'+slot;row=journal.call(key,req)
                if row['response']['http_status']==429 or row['response']['http_status']>=500:row=journal.call(key+'/retry1',req)
                parsed=parse_row(row,c['choices'])
                if parsed['http_status'] in (401,402,403):raise RuntimeError('API authorization/balance failure')
                backend='zhipu_flash' if slot=='G0' else 'deepseek'
                if parsed['http_status']==200 and parsed['model'] not in allowed_models[backend]:
                    raise RuntimeError('returned model identity changed; response preserved; recalibration required')
                slots[slot]=parsed
            outputs[c['id']]=slots
            print('reliable-readout pool saved',n+1,'/18',flush=True)
        freeze(OUT/'sealed_outputs.json',outputs)
        return analyze()


def analyze():
    cases=json.loads((OUT/'public_cases.json').read_text());outputs=json.loads((OUT/'sealed_outputs.json').read_text())
    model=json.loads((OUT/'calibration.json').read_text())['model']
    labels={r['id']:r['label'] for r in json.loads((DATA/'private_labels.json').read_text())}
    metrics={};records=[]
    for c in cases:
        s=outputs[c['id']]
        reports=[{'backend':'deepseek','answer':s[x]['answer']} for x in ['D0','D1']]+[{'backend':'zhipu_flash','answer':s['G0']['answer']}]
        decisions={'single_fast':s['D0']['answer'],'DDD':decide([{'backend':'deepseek','answer':s[x]['answer']} for x in ['D0','D1','D2']],c['choices'])['answer'],
                   'DDG':decide(reports,c['choices'])['answer'],'weighted_DDG':decide(reports,c['choices'],model)['answer'],
                   'single_thinking':s['T0']['answer']}
        profiles={'single_fast':['D0'],'DDD':['D0','D1','D2'],'DDG':['D0','D1','G0'],
                  'weighted_DDG':['D0','D1','G0'],'single_thinking':['T0']}
        for name,answer in decisions.items():
            rs=[s[x] for x in profiles[name]]
            m=metrics.setdefault(name,{'correct':0,'n':0,'calls':0,'tokens':0,'sum_http_seconds':0.,'invalid':0})
            m['correct']+=answer==labels[c['id']];m['n']+=1;m['calls']+=len(rs)
            m['invalid']+=sum(not r['valid'] for r in rs);m['sum_http_seconds']+=sum(r['wall_seconds'] for r in rs)
            if m['tokens'] is not None:m['tokens']=m['tokens']+sum(r['tokens'] for r in rs) if all(r['tokens'] is not None for r in rs) else None
        records.append({'id':c['id'],'gold':labels[c['id']],'decisions':decisions,
                        'weighted_scores':decide(reports,c['choices'],model)['scores']})
    contrasts={name:{'corrections':sum(r['decisions'][name]!=r['gold'] and r['decisions']['weighted_DDG']==r['gold'] for r in records),
                     'harms':sum(r['decisions'][name]==r['gold'] and r['decisions']['weighted_DDG']!=r['gold'] for r in records)}
               for name in ['DDG','DDD','single_fast','single_thinking']}
    raw=read_rows(OUT/'responses.jsonl')
    result={'n':18,'metrics':metrics,'weighted_vs_baselines':contrasts,'records':records,
            'model':model,'http_attempts':len(read_rows(OUT/'attempts.jsonl')),
            'returned_models':sorted({r['model'] for s in outputs.values() for r in s.values() if r['model']}),
            'response_journal_hash':design.p.digest(raw),'scope':'small independent-input baseline test, not independent novel contribution',
            'no_automatic_expansion':True}
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','model')},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:prepare()
