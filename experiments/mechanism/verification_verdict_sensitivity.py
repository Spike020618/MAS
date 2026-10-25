"""Posthoc explicit-verdict sensitivity. Never changes frozen primary parsing.

Only an extra literal type=json_object metadata field is accepted, with both
registered verdict and string reason present. Not regex guessing, label-based
repair, truncation recovery, or an independent experiment.
"""
from fractions import Fraction as F
import json

from experiments.core.mechanism.verification_risk_game import calibration_matrix,solve_risk_game,portable
from experiments.mechanism import run_verification_crossed_dev as run
from experiments.mechanism.run_upgrade_pilot import freeze,read_rows


def explicit_verdict(row):
    strict=run.decode(row)
    if strict['valid']:return strict['verdict'],False
    try:
        body=json.loads(row['response']['body']);c=body['choices'][0]
        if row['response']['http_status']!=200 or c['finish_reason']!='stop':return 'invalid',False
        value=json.loads(c['message']['content'])
        if (set(value)=={'type','verdict','reason'} and value['type']=='json_object' and
                value['verdict'] in ('accept','reject') and isinstance(value['reason'],str)):
            return value['verdict'],True
    except (ValueError,TypeError,KeyError,IndexError):pass
    return 'invalid',False


def main():
    raw=read_rows(run.OUT/'responses.jsonl');mapping={r['key']:r for r in raw}
    strict=json.loads((run.OUT/'calibration_trials.json').read_text())
    rows=[];recovered=set()
    for r in strict:
        suffix='independent' if r['verifier']=='independent' else r['verifier']+'/'+r['attack']
        key=r['question_id']+'/'+r['claim_id']+'/'+suffix
        verdict,changed=explicit_verdict(mapping[key])
        rows.append({**r,'verdict':verdict})
        if changed:recovered.add(key)
    matrix=calibration_matrix(rows,run.VERIFIERS,run.ATTACKS);game=solve_risk_game(matrix['losses'])
    qids=sorted({r['question_id'] for r in rows});totals={name:[F(0),F(0)] for name in ('game','best_fixed','uniform','independent')}
    for q in qids:
        training=calibration_matrix([r for r in rows if r['question_id']!=q],run.VERIFIERS,run.ATTACKS)
        evaluation=calibration_matrix([r for r in rows if r['question_id']==q],run.VERIFIERS,run.ATTACKS)
        pi=solve_risk_game(training['losses'])['verifier_policy']
        best=min(range(3),key=lambda i:(max(training['losses'][i]),i))
        policies={'game':pi,'best_fixed':tuple(F(int(i==best)) for i in range(3)),
                  'uniform':(F(1,3),)*3,'independent':(F(1),F(0),F(0))}
        for name,p in policies.items():
            totals[name]=[totals[name][j]+sum(p[i]*evaluation['losses'][i][j] for i in range(3)) for j in range(2)]
    result={'scope':'posthoc parser sensitivity, NOT replacement primary analysis or new independent test',
        'new_api_calls':0,'recovered_physical_verdicts':sorted(recovered),
        'matrix':matrix,'game':game,
        'incremental_empirical_game_gain_vs_best_pure':game['best_pure_worst_loss']-game['value'],
        'crossfit_expected_worst_risks':{name:max(losses)/len(qids) for name,losses in totals.items()},
        'conclusion':'any strict-parser mixed advantage must not be attributed to semantic game contribution',
        'all_original_questions_and_trials_retained':True}
    freeze(run.OUT/'verdict_sensitivity.json',portable(result))
    print(json.dumps(portable(result),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
