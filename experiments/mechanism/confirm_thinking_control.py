"""One independent input-group replication of the fixed thinking-mode comparison.

No changes to prompts, cap, model, mode or extraction after the first batch.
This tests a capability/resource intervention, not a proposed allocation policy.
"""
import argparse
import json
from pathlib import Path

from experiments.mechanism import study_information_complementarity as design
from experiments.mechanism import study_explicit_thinking_control as original
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows
from experiments.mechanism.run_premise_review import parse_row

DATA=design.OUT.with_name('thinking_confirmation_data_v1')
OUT=design.OUT.with_name('thinking_confirmation_v1')


def prepare():
    previous=design.OUT
    try:
        design.OUT=DATA
        with lock(DATA):design.prepare()
    finally:design.OUT=previous


def run():
    with lock(OUT):
        cases=json.loads((DATA/'public_cases.json').read_text())
        assert len(cases)==36
        prior=json.loads((original.OUT/'result.json').read_text())
        freeze(OUT/'protocol.json',{'version':'thinking-confirmation-v1','n':36,'task_calls':72,'max_attempts':144,
            'source':'new source-isolated inputs; selection hash order unchanged, all earlier cases excluded',
            'why_now':'first batch positive: 9 corrections/1 harm overall, 3/1 evaluation; independent replication, not reward tuning',
            'first_result_sha256':design.fingerprint(original.OUT/'result.json'),
            'model':'deepseek-flash','modes':list(original.MODES),'max_tokens':4096,
            'instruction_unchanged':True,'request_function_sha256':design.fingerprint(Path(original.__file__)),
            'code_sha256':design.fingerprint(Path(__file__)),'cases_hash':design.p.digest(cases),
            'primary':'enabled minus disabled on all 36 new cases; original calibration/evaluation fields retained only for provenance',
            'scope':'independent input-group replication of ability/resource intervention, not mechanism or model-version generality'})
        freeze(OUT/'public_cases.json',cases)
        journal=CallJournal(OUT,original.transport,144);outputs={}
        for i,case in enumerate(cases):
            pair={}
            for mode in sorted(original.MODES,key=lambda m:design.p.digest([case['id'],m,'thinking-order'])):
                req=original.request(case,mode);key=case['id']+'/'+mode
                row=journal.call(key,req)
                if row['response']['http_status']==429 or row['response']['http_status']>=500:
                    row=journal.call(key+'/retry1',req)
                parsed=parse_row(row,case['choices'])
                if parsed['http_status']!=200:raise RuntimeError('confirmation HTTP failure '+str(parsed['http_status']))
                b=json.loads(row['response']['body']);parsed['reasoning_content_present']=bool(b['choices'][0]['message'].get('reasoning_content'))
                pair[mode]=parsed
            outputs[case['id']]=pair
            print('confirmation pair saved',i+1,'/36',flush=True)
        freeze(OUT/'sealed_outputs.json',outputs)
        labels={r['id']:r['label'] for r in json.loads((DATA/'private_labels.json').read_text())}
        result={'n':36,'task_calls':72,'scope':'one small independent-input replication; no adaptive mechanism tested','modes':{},'rows':[]}
        for mode in original.MODES:
            rs=[outputs[c['id']][mode] for c in cases]
            result['modes'][mode]={'correct':sum(outputs[c['id']][mode]['answer']==labels[c['id']] for c in cases),
                'tokens':sum(r['tokens'] for r in rs) if all(r['tokens'] is not None for r in rs) else None,
                'invalid':sum(not r['valid'] for r in rs),'truncated':sum(r['finish_reason']=='length' for r in rs),
                'reasoning_content_present':sum(r['reasoning_content_present'] for r in rs),'sum_http_seconds':sum(r['wall_seconds'] for r in rs)}
        for c in cases:
            result['rows'].append({'id':c['id'],'gold':labels[c['id']],
                'answers':{m:outputs[c['id']][m]['answer'] for m in original.MODES}})
        result['corrections']=sum(r['answers']['disabled']!=r['gold'] and r['answers']['enabled']==r['gold'] for r in result['rows'])
        result['harms']=sum(r['answers']['disabled']==r['gold'] and r['answers']['enabled']!=r['gold'] for r in result['rows'])
        result['by_domain']={d:{m:sum(r['answers'][m]==r['gold'] for r in result['rows'] if r['id'].startswith(d+':')) for m in original.MODES}
                             for d in ['strategyqa','mmlu','truthfulqa']}
        rows=read_rows(OUT/'responses.jsonl');result['response_journal_hash']=design.p.digest(rows)
        result['actual_http_attempts']=len(read_rows(OUT/'attempts.jsonl'))
        result['models_returned']=sorted({r['model'] for pair in outputs.values() for r in pair.values() if r['model']})
        freeze(OUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:prepare()
