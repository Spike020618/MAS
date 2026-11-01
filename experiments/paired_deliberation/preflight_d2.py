"""Immutable D2 purpose release protocol. No dispatch on missing/failed release."""
import argparse
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.calibration_d2 import build

VERIFIER={'service_id':'deepseek-flash-d2-nonthinking-2026-10-27', 'provider':'deepseek','requested_model':'deepseek-flash',
 'allowed_returned_models':['deepseek-flash'],'input_cny_per_million':2.0,'output_cny_per_million':8.0,
 'mode':'thinking disabled','price_basis':'peak cache-miss ceiling; actual bill unknown',
 'price_source':'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/','price_checked_date':'2026-10-27',
 'version_limit':'API alias; official documented weights V4.1-Flash-1017, not an immutable weight hash'}
ACTORS={
 'glm_flash':{'service_id':'glm-4-flash-250414-d2','provider':'glm_flash','requested_model':'glm-4-flash-250414',
  'allowed_returned_models':['glm-4-flash-250414'],'input_cny_per_million':0.0,'output_cny_per_million':0.0},
 'qwen_flash':{'service_id':'qwen-flash-2025-07-28-d2','provider':'qwen_flash','requested_model':'qwen-flash-2025-07-28',
  'allowed_returned_models':['qwen-flash-2025-07-28'],'input_cny_per_million':0.15,'output_cny_per_million':1.5}}


def source_names():
    names=('preflight_d2.py','calibration_d2.py','validate_d2.py','engineering_d2.py','transport_d2.py',
           'evidence_d2.py','verification_d2.py','state_d2.py','requests_d2.py','output_adapter_d2.py',
           'runner_d2.py','budget_d2.py','coordination_d2.py','durable_provider.py','client_v5.py',
           'atomic_evidence.py','atomic_relations_v4.py','screen_output_projection.py','training_evidence_view.py',
           'state_v6.py','contracts_v6.py','verified_state.py','requests_v5.py','validate_atomic_evidence.py','budget_v6.py','json_contract_d2.py','budget_wire_d2.py','runner_d2_live.py','source_alias_d2.py')
    return ['experiments/paired_deliberation/'+n for n in names]+['experiments/consensus_recommendation/'+n for n in ('common.py','protocol.py')]


def prepare(output,reuse=None):
    root=Path(output);offline=ROOT/'experiments/results/paired_deliberation_d2_joint_offline_r2'
    accepted=json.loads((offline/'offline_acceptance.json').read_text())
    oldplan=json.loads((offline/'plan.json').read_text())
    if not accepted['offline_pass'] or accepted['plan_hash']!=oldplan['plan_hash']:raise RuntimeError('D2 offline integration required')
    for n,h in oldplan['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('offline integrated source changed')
    inputs,labels=build()
    plan={'version':'d2_purpose_release_v1','phase':'purpose_calibration','verifier':VERIFIER,
          'profiles':{'actor':ACTORS['glm_flash'],'verifier':VERIFIER},
          'counts':{'total':96,'new_utterance':32,'new_current_preference':32,'catalog':8,'prediction_basis':8,'authority':8,'hard_guards':4,'regression':4},
          'maximum_physical_attempts':144,'maximum_estimated_cny':3.0,
          'limits':{'max_calls':144,'max_evidence_checks':96,'per_call_prompt_tokens':12000,'per_call_completion_tokens':4096,
                    'total_prompt_tokens':144*12000,'total_completion_tokens':144*4096,'max_estimated_cny':3.0},
          'thresholds':{'new_semantic_joint_accuracy_min':0.90,'each_soft_purpose_joint_accuracy_min':0.875,
                        'new_semantic_invalid_max':2,'catalog_errors_max':0,'prediction_authority_errors_max':0,
                        'hard_guard_errors_max':0,'critical_current_conflict_false_certification_max':0,'regression_errors_max':0,
                        'sentinel_errors_max':0},
          'gate_interpretation':'limited shared-initial engineering only, not general truth or recommendation effectiveness',
          'analysis':'new soft purposes joint correctness = valid output + faithful parsing + correct final verdict; source relations diagnostic not universal accuracy gate; Wilson intervals descriptive',
          'inputs_hash':fingerprint(inputs),'labels_hash':fingerprint(labels),
          'code_sha256':{n:sha(ROOT/n) for n in source_names()},'offline_acceptance_sha256':sha(offline/'offline_acceptance.json'),
          'legacy_input_hashes':{n:sha(ROOT/n) for n in
              ('experiments/results/paired_deliberation_commit_v4_joint_validation_r1/inputs.json',
               'experiments/results/paired_deliberation_commit_v4_joint_validation_r1/labels.json')},
          'runtime_labels_available':False,'pilot_allowed':False,'reserved_data_allowed':False,
          'selection':'all fixed generated cases; old known failures kept separately; no error-dependent sampling',
          'stop':'one batch, no retries/substitution/prompt changes/threshold changes or expansion; transport/pending error halts immediately',
          'scope':'known template family controls, four renamed objects; not independent natural language accuracy'}
    if reuse is not None:
        previous=Path(reuse).resolve();old=json.loads((previous/'plan.json').read_text())
        if old['inputs_hash']!=plan['inputs_hash'] or old['labels_hash']!=plan['labels_hash'] or old['thresholds']!=plan['thresholds']:
            raise ValueError('repair must preserve all calibration inputs, labels and gates')
        if list((previous/'slots').glob('*.intent.json')):raise RuntimeError('pending prior request prevents repair dispatch')
        plan.update(reuse_root=str(previous/'slots'),reuse_sha256={p.name:sha(p) for p in (previous/'slots').glob('*.raw.json')},
                    repair='JSON wire requirement, variable-size sentinel/scorer harness, unique current-source aliases; inputs/labels/gates unchanged; exact-response reuse',
                    cumulative_physical_cap=145,prior_physical_attempts=old.get('prior_physical_attempts',0)+sum(
                        'reused_from' not in json.loads(p.read_text()) for p in (previous/'slots').glob('*.raw.json')))
    plan['plan_hash']=fingerprint(plan)
    create(root/'inputs.json',inputs);create(root/'labels.json',labels);create(root/'plan.json',plan)
    for n in plan['code_sha256']:
        path=root/'source_snapshot'/n;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=(ROOT/n).read_bytes():raise ValueError('snapshot changed')
        path.write_bytes((ROOT/n).read_bytes())
    return plan


def verify(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    if plan.get('version')!='d2_purpose_release_v1' or plan['plan_hash']!=fingerprint({k:v for k,v in plan.items() if k!='plan_hash'}):raise ValueError('D2 plan version/hash mismatch')
    for n,h in plan['code_sha256'].items():
        if sha(ROOT/n)!=h or sha(root/'source_snapshot'/n)!=h:raise ValueError('frozen D2 source changed: '+n)
    for name,key in (('inputs.json','inputs_hash'),('labels.json','labels_hash')):
        if fingerprint(json.loads((root/name).read_text()))!=plan[key]:raise ValueError('frozen D2 inputs/labels changed')
    for name,h in plan.get('reuse_sha256',{}).items():
        if sha(Path(plan['reuse_root'])/name)!=h:raise ValueError('prior receipt changed')
    return plan


def require_engineering_release(root):
    root=Path(root);plan=verify(root)
    if not (root/'release_score.json').exists():raise RuntimeError('D2 purpose calibration incomplete; no engineering provider creation')
    saved=json.loads((root/'release_score.json').read_text())
    if (saved.get('version')!='d2_purpose_release_v1' or saved.get('plan_hash')!=plan['plan_hash'] or
        not saved.get('pass') or not saved.get('gates') or not all(saved['gates'].values())):
        raise RuntimeError('D2 purpose gate failed; no engineering dispatch')
    from experiments.paired_deliberation.validate_d2 import score
    recomputed=score(root,write=False)
    if fingerprint(saved)!=fingerprint(recomputed):raise ValueError('D2 release not reproducible from frozen evidence')
    if not saved['pass'] or saved['plan_hash']!=plan['plan_hash'] or not all(saved['gates'].values()):
        raise RuntimeError('D2 purpose gate failed; no engineering dispatch')
    return plan,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--reuse')
    args=p.parse_args()
    print(json.dumps(prepare(args.output,args.reuse),ensure_ascii=False,indent=2))
