"""Exact synthetic solver checks + a real single-column historical diagnostic.

No LLM calls. Synthetic matrices are deliberately labelled as constructed.
Existing12 has no crossed verifier/message-protocol experiment, so it CANNOT
validate the proposed multi-policy adversarial mechanism.
"""
from hashlib import sha256
import json
from pathlib import Path

from experiments.core.mechanism.verification_risk_game import calibration_matrix,solve_risk_game,portable
from experiments.mechanism.run_upgrade_pilot import freeze

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/audit/verification_risk_design_v1'
SOURCE=ROOT/'experiments/results/resource_allocation/signaling_pilot_v1'


def main():
    fixtures={
        'complementary_constructed':(('1/10','7/10'),('3/5','1/5')),
        'one_verifier_dominates_constructed':(('1/10','1/10'),('3/10','2/5')),
        'identical_no_mechanism_headroom_constructed':(('1/4','1/4'),('1/4','1/4'))}
    solved={name:portable(solve_risk_game(matrix)) for name,matrix in fixtures.items()}
    # Read already unsealed DEVELOPMENT labels solely to evaluate verification risk.
    cases=json.loads((SOURCE/'public_cases.json').read_text())
    outputs=json.loads((SOURCE/'sealed_outputs.json').read_text())
    labels={r['id']:r['gold'] for r in json.loads((SOURCE/'result.json').read_text())['records']}
    trials=[]
    for c in cases:
        for candidate in c['choices']:
            score=outputs[c['id']]['d/'+candidate]
            verdict=('accept' if score['probabilities']['A']>=.5 else 'reject') if score['usable'] else 'invalid'
            trials.append({'question_id':c['id'],'claim_id':candidate,'verifier':'direct_independent',
                'attack':'no_proposal_message','verdict':verdict,'truth':int(candidate==labels[c['id']]),
                'truth_kind':'official_answer','truth_ref':c['id']+'/registered-option'})
    matrix=calibration_matrix(trials,['direct_independent'],['no_proposal_message'])
    historical={'calibration_table':portable(matrix),'solution':portable(solve_risk_game(matrix['losses'])),
        'positive_claims':sum(r['truth']==1 for r in trials),'negative_claims':sum(r['truth']==0 for r in trials),
        'false_accept_count':sum(r['truth']==0 and r['verdict']=='accept' for r in trials),
        'false_reject_count':sum(r['truth']==1 and r['verdict']=='reject' for r in trials),
        'warning':'single verifier/single message condition; equilibrium is trivial; does not validate new method',
        'source_scope':'official answer correctness, NOT validity of an argument or independent factual adjudication'}
    result={'status':'objective_and_solver_prototype_only','new_api_calls':0,'new_independent_questions':0,
        'synthetic_fixtures':solved,'historical_single_column_diagnostic':historical,
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [SOURCE/'public_cases.json',SOURCE/'sealed_outputs.json',SOURCE/'result.json']}}
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'historical_candidate_trials.json',trials);freeze(OUT/'result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
