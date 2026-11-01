"""Reproducible scripted D2 integration batch; NO model/API or natural quality score."""
import argparse
import copy
import json
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT,create,fingerprint,sha
from experiments.paired_deliberation.runner_d2 import clients,shared_initial,run_offline_suite
from experiments.paired_deliberation.budget_d2 import JointBudget
from experiments.paired_deliberation.contracts_v6 import METHODS
from experiments.tests.test_d2_budget_coordination import LIMITS,PROFILES
from experiments.tests.test_d2_runner import FixtureActor,FixtureJudge
from experiments.tests.test_d2_contract_state import registry,claim,report
from experiments.tests.test_design_v6 import evidence


class IrrelevantActor(FixtureActor):
    def initial(self,owner):
        c=claim(self.r,kind='hypothesis' if owner=='movie_fit' else 'catalog_fact',text='Movie 6 has catalog year 2001.')
        c['candidate_ids']=['6']
        if c['claim_kind']=='catalog_fact':c['evidence_refs']=[{'source_id':self.r.resolve('movie:6'),'use':'supports'}]
        return report(self.r,[c])


def run(output):
    output=Path(output);all_results=[];initial_records=[];provider_invocations=0;isolation_errors=[];permission_errors=[]
    for kind in ('catalog','semantic_conflict','irrelevant'):
        e=evidence();e['case_id']='offline_d2:'+kind
        if kind=='semantic_conflict':e['observations'][0]['text']='I want Drama. I do not want Drama.'
        r=registry(e);actor_cls=IrrelevantActor if kind=='irrelevant' else FixtureActor
        semantic=kind=='semantic_conflict'
        initial_actor,initial_judge=actor_cls(r,semantic),FixtureJudge()
        initial_ledger=JointBudget(LIMITS,PROFILES)
        common=shared_initial(r,clients(initial_ledger,{'actor':initial_actor,'verifier':initial_judge})['actor'],kind+'_common')
        initial_records.append({'case':kind,'shared':common,'logical_generation_usage':initial_ledger.summary()})
        providers={}
        def factory(method):
            pair={'actor':actor_cls(r,semantic),'verifier':FixtureJudge()};providers[method]=pair;return pair
        results=run_offline_suite(r,factory,LIMITS,PROFILES,kind,common,max_rounds=1)
        all_results.append({'case':kind,'registry_hash':r.integrity,'methods':results})
        provider_invocations+=len(initial_actor.calls)+len(initial_judge.calls)
        for method,pair in providers.items():
            provider_invocations+=sum(len(p.calls) for p in pair.values())
            for _,request in pair['actor'].calls:
                payload=json.loads(request['messages'][1]['content'])
                if (payload.get('evidence_registry',{}).get('integrity')!=r.integrity or
                    payload.get('training_prediction_evidence')!=r.training_view):permission_errors.append([kind,method])
            for _,request in pair['verifier'].calls:
                payload=json.loads(request['messages'][1]['content'])
                if 'training_prediction_evidence' in payload or 'evidence_registry' in payload:isolation_errors.append([kind,method])
                if request['kind']=='atomic_interpretation' and set(payload)!={'original_claim','schema'}:isolation_errors.append([kind,method])
                if any(key.startswith('train') for key in payload.get('sources',{})):isolation_errors.append([kind,method])
    paths=[record['methods'].get(method) for record in all_results for method in METHODS]
    completed=[p['result'] for p in paths if p and p['status']=='completed']
    within=all(row['usage']['attempts']<=LIMITS['max_calls'] and row['usage']['reserved_prompt_upper']<=LIMITS['total_prompt_tokens'] and
               row['usage']['reserved_output']<=LIMITS['total_completion_tokens'] and row['usage']['reserved_cost_cny']<=LIMITS['max_estimated_cny']
               for row in completed)
    accounting=all(row['usage']['attempts']==sum(ch['attempts'] for ch in row['usage']['by_channel'].values()) and
                   sum(r['shared'] for r in row['usage']['records'])==3 for row in completed)
    irrelevant=all_results[-1]['methods']
    gates={'all_15_paths_completed':len(completed)==15,'zero_contract_rejections':all(not row['rejected'] for row in completed),
           'joint_reservations_within_limits':within,'both_channels_accounted_and_shared_charged':accounting,
           'known_fixture_usage':all(row['usage']['usage_complete'] for row in completed),
           'same_actor_evidence_permissions':not permission_errors,'verifier_isolation':not isolation_errors,
           'adaptive_can_stop_on_irrelevant_fixture':irrelevant['verified_adaptive']['result']['stop']=='no_decision_relevant_target'}
    sources=['experiments/paired_deliberation/'+n for n in ('budget_d2.py','coordination_d2.py','runner_d2.py',
        'offline_engineering_d2.py','evidence_d2.py','state_d2.py','verification_d2.py','output_adapter_d2.py','requests_d2.py')]
    sources+=['experiments/tests/'+n for n in ('test_d2_budget_coordination.py','test_d2_runner.py','test_d2_contract_state.py','test_design_v6.py')]
    sources+=['experiments/paired_deliberation/'+n for n in ('atomic_evidence.py','atomic_relations_v4.py',
        'state_v6.py','contracts_v6.py','verified_state.py','requests_v5.py','screen_output_projection.py','training_evidence_view.py',
        'validate_atomic_evidence.py','budget_v6.py')]
    sources+=['experiments/consensus_recommendation/'+n for n in ('common.py','protocol.py')]
    plan={'version':'d2_scripted_joint_integration_v1','cases':['catalog','semantic_conflict','irrelevant'],
          'methods':list(METHODS),'rounds':1,'limits':LIMITS,'profiles':PROFILES,
          'prices_are_synthetic_budget_test_units':True,'fixture_provider_invocations_are_not_api_calls':True,
          'source_sha256':{n:sha(ROOT/n) for n in sources},'new_model_calls':0,'online_release':False,
          'scope':'scripted state/accounting/isolation integration; final rankings are scripted, not task outcomes'}
    plan['plan_hash']=fingerprint(plan)
    create(output/'plan.json',plan);create(output/'shared.json',initial_records);create(output/'cases.json',all_results)
    acceptance={'offline_pass':all(gates.values()),'gates':gates,'completed_paths':len(completed),'plan_hash':plan['plan_hash'],
                'scripted_provider_invocations':provider_invocations,'new_model_calls':0,'online_release':False,
                'semantic_accuracy_validated':False,'recommendation_gain_validated':False,
                'fixture_prices_are_not_real_model_prices':True,'dev_labels_read':False,'reserved_data_read':False,
                'limits':['not a new 15-path live engineering result','semantic and ranking responses are scripts',
                          'purpose calibration and real model engineering remain pending']}
    create(output/'offline_acceptance.json',acceptance)
    for name in sources:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=(ROOT/name).read_bytes():raise ValueError('snapshot changed')
        target.write_bytes((ROOT/name).read_bytes())
    return acceptance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=False,indent=2))
