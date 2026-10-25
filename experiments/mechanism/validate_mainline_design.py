"""Validate mechanism-study invariants and block collection before freezing."""
import argparse
import json
from pathlib import Path


def validate(config):
    errors=[]
    environment=config['fixed_environment']
    required={
        'same_execution_for_all_contracts':True,
        'contract_visible_in_execution_prompt':False,
        'same_source_bank_for_all_contracts':True,
        'task_specific_checkers_allowed':False,
        'same_question_labels_visible_to_agents':False,
    }
    for field,value in required.items():
        if environment.get(field) is not value:errors.append(f'fixed environment violated: {field}')
    if config['payments'].get('primary_payment_uses_llm_support_judge') is not False:
        errors.append('primary payments cannot treat LLM support as truth')
    if config['equilibrium'].get('observed_policy_optimum_is_equilibrium') is not False:
        errors.append('observed policy optimization cannot certify equilibrium')
    if config['equilibrium'].get('per_question_gold_oracle_contract_selection_allowed') is not False:
        errors.append('gold-oracle contract selection forbidden')
    if 0 not in config['payments']['budget_levels_provisional']:
        errors.append('zero-payment candidate must remain available')
    n_profiles=len(config['actions'])**config['n_agents']
    if config['equilibrium']['joint_action_profiles']!=n_profiles:
        errors.append('joint action profile count mismatch')
    pilot=config['pilot']
    expected=pilot['n_questions']*pilot['n_repeats']*config['n_agents']*len(config['actions'])
    if pilot['forced_action_execution_units']!=expected:errors.append('pilot unit count mismatch')
    data=config['data']
    splits=['calibration_question_ids','development_question_ids','test_question_ids']
    for i,left in enumerate(splits):
        for right in splits[i+1:]:
            if data.get(left) is not None and data.get(right) is not None:
                if set(data[left]) & set(data[right]):errors.append(f'overlap: {left}/{right}')
    unresolved=[]
    for field in ['retriever_version','source_manifest_sha256','report_prompt_sha256']:
        if not environment.get(field):unresolved.append(field)
    if not config['payments'].get('unit_definition'):unresolved.append('unit_definition')
    if config['payments'].get('account_enforcement_verified') is not True:unresolved.append('account_enforcement')
    if config['leader_objective'].get('weights') is None:unresolved.append('leader_weights')
    if any(not data.get(field) for field in ['pilot_question_ids']+splits):unresolved.append('question_ids')
    if data.get('historical_exposure_audit_completed') is not True:unresolved.append('exposure_audit')
    if not config['statistics'].get('confirmatory_contrasts'):unresolved.append('confirmatory_contrasts')
    ready=not errors and not unresolved and config.get('status')=='frozen' and config.get('paid_collection_allowed') is True
    return {'design_consistency_pass':not errors,'violations':errors,
            'ready_for_paid_collection':ready,'unresolved_freeze_items':unresolved,
            'joint_action_profiles':n_profiles,'pilot_execution_units':expected,
            'note':'valid design is not an executed or frozen experiment'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('config',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--require-ready',action='store_true')
    args=p.parse_args()
    result=validate(json.loads(args.config.read_text()))
    text=json.dumps(result,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    print(text)
    if not result['design_consistency_pass'] or args.require_ready and not result['ready_for_paid_collection']:
        raise SystemExit(2)


if __name__=='__main__':main()
