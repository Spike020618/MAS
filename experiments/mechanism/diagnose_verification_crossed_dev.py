"""Offline secondary question-level checks; no extra models or tuning."""
from fractions import Fraction as F
import json

from experiments.core.mechanism.verification_risk_game import calibration_matrix,solve_risk_game,portable
from experiments.mechanism import run_verification_crossed_dev as run
from experiments.mechanism.run_upgrade_pilot import freeze,read_rows


def main():
    rows=json.loads((run.OUT/'calibration_trials.json').read_text())
    main_result=json.loads((run.OUT/'result.json').read_text())
    qids=sorted({r['question_id'] for r in rows});rates={};by_q={}
    for v in run.VERIFIERS:
        rates[v]={}
        for a in run.ATTACKS:
            selected=[r for r in rows if r['verifier']==v and r['attack']==a]
            macro={0:[],1:[]}
            for q in qids:
                for label in (0,1):
                    cell=[r for r in selected if r['question_id']==q and r['truth']==label]
                    error=sum(r['verdict']=='invalid' or (r['verdict']=='accept')!=bool(label) for r in cell)
                    macro[label].append(F(error,len(cell)))
            rates[v][a]={'false_accept_question_macro':sum(macro[0])/len(qids),
                         'false_reject_question_macro':sum(macro[1])/len(qids),
                         'invalid':sum(r['verdict']=='invalid' for r in selected)}
    totals={name:[F(0) for _ in run.ATTACKS] for name in ('loo_game','loo_best_fixed','uniform','independent')}
    fold=[]
    for q in qids:
        training=calibration_matrix([r for r in rows if r['question_id']!=q],run.VERIFIERS,run.ATTACKS)
        evaluation=calibration_matrix([r for r in rows if r['question_id']==q],run.VERIFIERS,run.ATTACKS)
        policy=solve_risk_game(training['losses'])['verifier_policy']
        best=min(range(len(run.VERIFIERS)),key=lambda i:(max(training['losses'][i]),i))
        profiles={'loo_game':policy,'loo_best_fixed':tuple(F(int(i==best)) for i in range(len(run.VERIFIERS))),
                  'uniform':(F(1,len(run.VERIFIERS)),)*len(run.VERIFIERS),
                  'independent':tuple(F(int(i==0)) for i in range(len(run.VERIFIERS)))}
        losses={}
        for name,p in profiles.items():
            losses[name]=[sum(p[i]*evaluation['losses'][i][j] for i in range(len(p))) for j in range(len(run.ATTACKS))]
            totals[name]=[x+y for x,y in zip(totals[name],losses[name])]
        fold.append({'held_out_question':q,'policy':policy,'best_fixed':run.VERIFIERS[best],'expected_losses':losses})
    policy=tuple(F(p) for p in main_result['game']['verifier_policy'])
    expected_cost={a:(sum(policy[i]*main_result['selected_judge_costs'][v][a]['tokens']
                         for i,v in enumerate(run.VERIFIERS)) if all(
                         main_result['selected_judge_costs'][v][a]['tokens'] is not None for v in run.VERIFIERS) else None)
                   for a in run.ATTACKS}
    raw=read_rows(run.OUT/'responses.jsonl')
    author_usage=[json.loads(r['response']['body']).get('usage',{}).get('total_tokens') for r in raw if r['key'].endswith('/author')]
    author_tokens=sum(author_usage) if all(type(u)is int and u>=0 for u in author_usage) else None
    summary={'scope':'secondary leave-one-original-question-out diagnostic on already-open development; NOT independent test',
        'new_api_calls':0,'classification_rates':rates,
        'crossfit_expected_risks':{name:{'per_presentation':[x/len(qids) for x in losses],
            'worst_presentation':max(losses)/len(qids)} for name,losses in totals.items()},
        'expectation_not_actual_sampled_deployment':'fractional policy expectations over saved responses; not additional independent calls',
        'expected_judge_tokens_at_frozen_development_policy':expected_cost,
        'author_data_construction_tokens_separately_reported':author_tokens,
        'cost_scope':'no monetary or efficiency advantage claimed; author generation reported separately; no inference repeated to realize these expectations',
        'folds':fold}
    freeze(run.OUT/'diagnosis.json',portable(summary))
    print(json.dumps(portable({k:v for k,v in summary.items() if k!='folds'}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
