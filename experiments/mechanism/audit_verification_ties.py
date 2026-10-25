"""Expose tie-only apparent crossfit gains; never replace primary selection."""
from fractions import Fraction as F
import json

from experiments.core.mechanism.verification_risk_game import calibration_matrix,solve_risk_game,portable
from experiments.mechanism import run_verification_crossed_dev as run
from experiments.mechanism.verification_verdict_sensitivity import explicit_verdict
from experiments.mechanism.run_upgrade_pilot import read_rows,freeze


def best_pure(losses,lexicographic_policy=False):
    n=len(losses)
    def key(i):
        onehot=tuple(F(int(k==i)) for k in range(n))
        return (max(losses[i]),onehot if lexicographic_policy else i)
    return min(range(n),key=key)


def main():
    raw={r['key']:r for r in read_rows(run.OUT/'responses.jsonl')}
    strict=json.loads((run.OUT/'calibration_trials.json').read_text());rows=[]
    for r in strict:
        suffix='independent' if r['verifier']=='independent' else r['verifier']+'/'+r['attack']
        verdict,_=explicit_verdict(raw[r['question_id']+'/'+r['claim_id']+'/'+suffix])
        rows.append({**r,'verdict':verdict})
    qids=sorted({r['question_id'] for r in rows});totals={name:[F(0),F(0)] for name in ('game','fixed_first_tie','fixed_same_lex_tie')};folds=[]
    for q in qids:
        train=calibration_matrix([r for r in rows if r['question_id']!=q],run.VERIFIERS,run.ATTACKS)['losses']
        test=calibration_matrix([r for r in rows if r['question_id']==q],run.VERIFIERS,run.ATTACKS)['losses']
        pi=solve_risk_game(train)['verifier_policy'];a=best_pure(train);b=best_pure(train,True)
        ps={'game':pi,'fixed_first_tie':tuple(F(int(i==a)) for i in range(3)),
            'fixed_same_lex_tie':tuple(F(int(i==b)) for i in range(3))}
        for name,p in ps.items():totals[name]=[totals[name][j]+sum(p[i]*test[i][j] for i in range(3)) for j in range(2)]
        folds.append({'held_out':q,'game_policy':pi,'fixed_first':run.VERIFIERS[a],'fixed_same_lex':run.VERIFIERS[b],
                      'game_identical_to_same_tie_fixed':pi==ps['fixed_same_lex_tie']})
    result={'scope':'posthoc tie-policy audit of parser sensitivity, not independent evidence or new primary protocol',
        'new_api_calls':0,'expected_worst_risks':{name:max(v)/len(qids) for name,v in totals.items()},
        'game_identical_to_same_tie_fixed_folds':sum(r['game_identical_to_same_tie_fixed'] for r in folds),
        'n_folds':len(qids),'folds':folds,
        'warning':'different handling of empirical ties can manufacture apparent game advantage over a selected pure control'}
    freeze(run.OUT/'tie_audit.json',portable(result))
    print(json.dumps(portable({k:v for k,v in result.items() if k!='folds'}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
