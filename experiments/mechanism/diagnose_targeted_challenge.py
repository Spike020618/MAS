"""Question-clustered diagnostics; never fits prompts, weights or rewards."""
import json
from random import Random
from experiments.mechanism import run_targeted_challenge as run
from experiments.core.mechanism.targeted_challenge import balanced_errors
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    result=json.loads((run.OUT/'result.json').read_text());records=result['records']
    outputs=json.loads((run.OUT/'sealed_outputs.json').read_text());ids=sorted(outputs)
    comparisons={}
    for baseline in ('ordinary0','ordinary2','nonadaptive','native'):
        differences=[]
        for cid in ids:
            rows=[r for r in records if r['id']==cid]
            differences.append(balanced_errors(rows,'interactive')['question_macro_balanced_error']-
                               balanced_errors(rows,baseline)['question_macro_balanced_error'])
        rng=Random(20261025)
        samples=sorted(sum(differences[rng.randrange(len(ids))] for _ in ids)/len(ids) for _ in range(10000))
        comparisons[baseline]={'risk_difference_negative_better':sum(differences)/len(ids),
            'per_document_differences':dict(zip(ids,differences)),
            'descriptive_question_bootstrap_95':[samples[249],samples[9749]],
            'limits':'six source groups, small exploratory sample; not 24 independent claims or formal superiority test'}
    exact_changes=[];reply_changes=0
    for cid,r in outputs.items():
        for candidate,s in r['claims'].items():
            if s['interactive']['verdict']!=s['nonadaptive']['verdict']:
                exact_changes.append({'id':cid,'candidate':candidate,
                    'nonadaptive':s['nonadaptive']['verdict'],'interactive':s['interactive']['verdict']})
            reply_changes+=s['reply'].get('argument')!=s['restatement'].get('argument')
    diagnostic={'comparisons':comparisons,'changed_final_decisions_from_nonadaptive':exact_changes,
        'different_reply_text_count':reply_changes,'claim_count':len(records),
        'interpretation':'different text is not proof of useful adaptation or strategic best response',
        'new_api_calls':0,'gates':'do not expand unless protocol has a defensible increment over ordinary and nonadaptive controls'}
    freeze(run.OUT/'diagnosis.json',diagnostic)
    print(json.dumps(diagnostic,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
