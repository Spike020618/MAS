"""Precommitted calibration-only gate for a bounded readout follow-up.

No new test cases selected by labels; evaluate all 18 evaluation cases if gate
passes. Unweighted extra-sample four-vote comparator can reuse the saved pool.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from experiments.mechanism import study_two_backend_complementarity as study
from experiments.mechanism.run_upgrade_pilot import freeze, CallJournal, read_rows
from experiments.mechanism.run_premise_review import parse_row

OUT=study.OUT/'readout_followup'
INSTRUCTION=('Solve the original question carefully using the supplied independent reports as fallible claims. '
    'Each report may be wrong. Decide from the merits of their reasoning, not the vote count. '
    'Do not invent external verification or change the question scope. '
    'Return one complete JSON object with answer (registered choice or abstain) and reasoning. Put answer first.')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'protocol.json',{'version':'readout-gate-v1','calibration_gate':
        {'DDG_correct_available_but_vote_wrong_at_least':2,'GLM_supplies_missing_DS_answer_at_least':1},
        'selection':'gate uses calibration 18 only; if passed execute every evaluation case, no gold-error selection',
        'task_calls_if_passed':18,'max_http_attempts':36,'temperature':.7,'max_tokens':800,
        'input_profile':'DS0, DS1, GLM0, anonymized and fixed hash-order',
        'comparators':['DDG majority (3 calls)','DDG plus DS2 vote (4 calls, ties abstain)','DDD majority (3 calls)'],
        'costs':'judge includes shared three-call input cost; fourth independent DS comparator has same call count, not equal tokens',
        'instruction':INSTRUCTION,'code_sha256':study.design.fingerprint(Path(__file__)),
        'scope':'precommitted follow-up of same evaluation inputs, not a second independent replication'})


def vote4(answers):
    counts=Counter(a for a in answers if a not in ('abstain',None))
    if not counts:return 'abstain'
    ranked=counts.most_common()
    return ranked[0][0] if ranked[0][1]>=2 and (len(ranked)==1 or ranked[0][1]>ranked[1][1]) else 'abstain'


def run():
    from experiments.core.llm import LLM_BACKENDS
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['code_sha256']==study.design.fingerprint(Path(__file__))
    parent=json.loads((study.OUT/'result.json').read_text())
    cal=parent['results']['calibration']
    passed=(cal['metrics']['DDG']['readout_loss']>=2 and cal['diagnostics']['GLM_supplies_missing_DS_answer']>=1)
    freeze(OUT/'gate.json',{'passed':passed,'calibration_DDG_readout_loss':cal['metrics']['DDG']['readout_loss'],
        'calibration_GLM_supplies_missing_DS_answer':cal['diagnostics']['GLM_supplies_missing_DS_answer'],
        'parent_result_sha256':study.design.fingerprint(study.OUT/'result.json')})
    if not passed:
        print('readout gate not passed; 0 additional API calls');return
    cases=[c for c in json.loads((study.OUT/'public_cases.json').read_text()) if c['split']=='evaluation']
    sealed=json.loads((study.OUT/'sealed_outputs.json').read_text())
    journal=CallJournal(OUT,study.design.p.transport,36);outputs={}
    model=LLM_BACKENDS['deepseek']['model']
    if model!=json.loads((study.OUT/'protocol.json').read_text())['models']['deepseek']:
        raise ValueError('model alias changed')
    for i,case in enumerate(cases):
        slots=sealed[case['id']]
        selected=sorted(study.PROFILES['DDG'],key=lambda x:study.design.p.digest([case['id'],list(x),'readout-card-order']))
        cards=[{'answer':slots[b+'/'+str(j)]['answer'],'reasoning':slots[b+'/'+str(j)]['reasoning']} for b,j in selected]
        req={'backend':'deepseek','model':model,'temperature':.7,'max_tokens':800,
             'response_format':{'type':'json_object'},'messages':[{'role':'system','content':INSTRUCTION},
                 {'role':'user','content':json.dumps({'question':case['question'],'choices':case['choices'],'reports':cards},sort_keys=True)}]}
        row=journal.call(case['id']+'/readout',req)
        if row['response']['http_status']==429 or row['response']['http_status']>=500:
            row=journal.call(case['id']+'/readout/retry1',req)
        outputs[case['id']]=parse_row(row,case['choices'])
        print('readout follow-up saved',i+1,'/18',flush=True)
    freeze(OUT/'sealed_outputs.json',outputs)
    labels={r['id']:r['label'] for r in json.loads((study.OUT/'private_labels.json').read_text())}
    result={'n':18,'new_task_calls':18,'correct':Counter(),'comparisons':{},'rows':[]}
    for case in cases:
        s=sealed[case['id']];gold=labels[case['id']]
        ddg=[s[b+'/'+str(i)] for b,i in study.PROFILES['DDG']]
        fresh=s['deepseek/2']
        answers={'DDG':study.design.majority([r['answer'] for r in ddg]),
                 'DDG_plus_DS2':vote4([r['answer'] for r in ddg]+[fresh['answer']]),
                 'DDD':study.design.majority([s['deepseek/'+str(i)]['answer'] for i in range(3)]),
                 'judge':outputs[case['id']]['answer']}
        cost=sum(r['tokens'] for r in ddg) if all(r['tokens'] is not None for r in ddg) else None
        result['rows'].append({'id':case['id'],'gold':gold,'answers':answers,
            'judge_tokens_with_input':cost+outputs[case['id']]['tokens'] if cost is not None and outputs[case['id']]['tokens'] is not None else None,
            'extra_sample_tokens_with_input':cost+fresh['tokens'] if cost is not None and fresh['tokens'] is not None else None})
        for a,v in answers.items():result['correct'][a]+=v==gold
    for b in ['DDG','DDG_plus_DS2','DDD']:
        result['comparisons'][b]={'corrections':sum(r['answers'][b]!=r['gold'] and r['answers']['judge']==r['gold'] for r in result['rows']),
                                 'harms':sum(r['answers'][b]==r['gold'] and r['answers']['judge']!=r['gold'] for r in result['rows'])}
    result['correct']=dict(result['correct'])
    freeze(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:prepare()
