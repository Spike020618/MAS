import copy
import json
import unittest

from experiments.paired_deliberation.budget_d2 import JointBudget
from experiments.paired_deliberation.evidence_d2 import EvidenceRegistry
from experiments.paired_deliberation.runner_d2 import clients,shared_initial,run_method,run_suite,run_offline_suite
from experiments.paired_deliberation.requests_d2 import initial_request
from experiments.paired_deliberation.contracts_v6 import OWNERS,METHODS
from experiments.tests.test_d2_contract_state import registry,claim,report
from experiments.tests.test_d2_budget_coordination import LIMITS,PROFILES
from experiments.tests.test_design_v6 import evidence


class FixtureActor:
    offline_only=True
    def __init__(self,r,semantic=False,invalid_patch=False,pending=False):
        self.r,self.semantic,self.invalid_patch,self.pending=r,semantic,invalid_patch,pending;self.calls=[]
    def initial(self,owner):
        if owner=='current_request' and self.semantic:
            c=claim(self.r,kind='current_preference',text='The seeker wants Drama.')
        elif owner=='movie_fit':c=claim(self.r,kind='hypothesis',text='Movie 1 may suit the request.')
        else:c=claim(self.r,text='Movie 1 has catalog year 2001.' if owner=='current_request' else 'Movie 1 has catalog year 2000.')
        return report(self.r,[c])
    def complete(self,slot,request):
        self.calls.append((slot,copy.deepcopy(request)));p=json.loads(request['messages'][1]['content']);kind=request['kind']
        if kind=='initial_d2':obj=self.initial(p['owner'])
        elif kind=='initial_central_d2':obj={'reports':{o:self.initial(o) for o in OWNERS}}
        elif kind=='select_d2':obj={'target_claim_id':p['allowed_targets'][0]}
        elif kind=='patch_d2':
            if self.pending:raise RuntimeError('pending fixture; no resend')
            obj=copy.deepcopy(p['schema'])
            if self.invalid_patch:
                obj['replacement']=copy.deepcopy(p['target']['claim']);obj['replacement']['stance']='refute'
        elif kind=='whole_review_d2':obj=report(self.r,[copy.deepcopy(p['target']['claim'])])
        elif kind=='ranking_d2':obj={'ranking':self.initial('current_request')['ranking']}
        else:raise AssertionError(kind)
        return {'report':obj,'model':'offline-small','usage':{'prompt_tokens':10,'completion_tokens':5},'elapsed_ms':1}


class FixtureJudge:
    offline_only=True
    def __init__(self):self.calls=[]
    def complete(self,slot,request):
        self.calls.append((slot,copy.deepcopy(request)));p=json.loads(request['messages'][1]['content'])
        if request['kind']=='atomic_interpretation':
            assert set(p)=={'original_claim','schema'}
            obj={'scope':'preference_state','subject':'seeker','predicate':'wants','object':'Drama','polarity':True,'time':'current','reason':'Fixture interpretation.'}
        elif request['kind']=='atomic_source_relations':
            assert not any(s.startswith('train') for s in p['sources'])
            sid=next(iter(p['sources']))
            obj={'relations':[{'source_id':sid,'relation':'both','support_quote':'I want Drama.','refute_quote':'I do not want Drama.'}],'reason':'Fixture same-time conflict.'}
        else:raise AssertionError(request['kind'])
        return {'report':obj,'model':'offline-judge','usage':{'prompt_tokens':10,'completion_tokens':5},'elapsed_ms':1}


class RunnerTests(unittest.TestCase):
    def common(self,r,semantic=False):
        actor=FixtureActor(r,semantic);judge=FixtureJudge();ledger=JointBudget(LIMITS,PROFILES)
        shared=shared_initial(r,clients(ledger,{'actor':actor,'verifier':judge})['actor'],'common')
        return shared,actor,judge

    def test_five_conditions_complete_with_shared_generation_logically_charged(self):
        r=registry();common,initial_actor,_=self.common(r)
        result=run_offline_suite(r,lambda method:{'actor':FixtureActor(r),'verifier':FixtureJudge()},LIMITS,PROFILES,'suite',common,max_rounds=1)
        self.assertEqual(set(result),set(METHODS));self.assertTrue(all(x['status']=='completed' for x in result.values()))
        self.assertEqual(len(initial_actor.calls),3)
        for item in result.values():
            usage=item['result']['usage'];self.assertEqual(sum(rec['shared'] for rec in usage['records']),3)
            self.assertTrue(usage['usage_complete']);self.assertFalse(item['result']['online_release'])
        self.assertEqual(result['verified_fixed']['result']['state']['retracted_ids'],['current_request:c1'])
        self.assertFalse(any(e['action'].startswith('observe_') for e in result['peer']['result']['events']))

    def test_actor_and_public_judge_use_distinct_models_but_joint_budget(self):
        e=evidence();e['observations'][0]['text']='I want Drama. I do not want Drama.';r=registry(e)
        shared,_,_=self.common(r,True);a,j=FixtureActor(r,True),FixtureJudge()
        result=run_method(r,'verified_fixed',{'actor':a,'verifier':j},LIMITS,PROFILES,'semantic',shared,max_rounds=1)
        usage=result['usage'];self.assertEqual(usage['attempts'],7)
        self.assertEqual(usage['by_channel']['actor']['attempts'],5);self.assertEqual(usage['by_channel']['verifier']['attempts'],2)
        self.assertEqual(usage['by_channel']['actor']['returned_models'],['offline-small'])
        self.assertEqual(usage['by_channel']['verifier']['returned_models'],['offline-judge'])
        self.assertEqual(result['state']['uncertain_claims']['current_request:c1']['status'],'unresolved')
        payload=json.loads(a.calls[0][1]['messages'][1]['content'])
        self.assertEqual(payload['target']['status'],'unresolved')
        self.assertEqual(len(j.calls),2)

    def test_rejected_repair_and_pending_call_preserve_observation_with_failure_ledger(self):
        r=registry();common,_,_=self.common(r)
        result=run_method(r,'verified_fixed',{'actor':FixtureActor(r,invalid_patch=True),'verifier':FixtureJudge()},LIMITS,PROFILES,'bad',common,max_rounds=1)
        self.assertEqual(len(result['rejected']),1);self.assertEqual(result['state']['retracted_ids'],['current_request:c1'])
        with self.assertRaises(RuntimeError) as caught:
            run_method(r,'verified_fixed',{'actor':FixtureActor(r,pending=True),'verifier':FixtureJudge()},LIMITS,PROFILES,'pending',common,max_rounds=1)
        self.assertEqual(caught.exception.d2_partial['state']['retracted_ids'],['current_request:c1'])
        self.assertEqual(caught.exception.d2_usage['attempts'],4)
        self.assertFalse(caught.exception.d2_usage['usage_complete'])

    def test_final_reservation_avoids_starting_unaffordable_action(self):
        r=registry();common,_,_=self.common(r);a=FixtureActor(r)
        result=run_method(r,'verified_fixed',{'actor':a,'verifier':FixtureJudge()},{**LIMITS,'max_calls':4},PROFILES,'tight',common,max_rounds=3)
        self.assertEqual(result['stop'],'budget_reserved_for_final');self.assertEqual(result['usage']['attempts'],4)
        self.assertEqual([q['kind'] for _,q in a.calls],['ranking_d2'])

    def test_end_to_end_central_generates_nine_capacity_in_one_call(self):
        r=registry();a=FixtureActor(r)
        result=run_method(r,'central',{'actor':a,'verifier':FixtureJudge()},LIMITS,PROFILES,'central',max_rounds=1)
        self.assertEqual([q['kind'] for _,q in a.calls],['initial_central_d2','select_d2','patch_d2','ranking_d2'])
        self.assertEqual(a.calls[0][1]['max_tokens'],3840)
        self.assertEqual(result['usage']['attempts'],4)

    def test_center_can_choose_affordable_catalog_when_semantic_target_does_not_fit(self):
        e=evidence();e['observations'][0]['text']='I want Drama. I do not want Drama.';r=registry(e)
        common,_,_=self.common(r,True);a,j=FixtureActor(r,True),FixtureJudge()
        result=run_method(r,'central',{'actor':a,'verifier':j},{**LIMITS,'max_calls':6},PROFILES,'feasible',common,max_rounds=1)
        selection=json.loads(a.calls[0][1]['messages'][1]['content'])
        self.assertEqual(selection['allowed_targets'],['history_applicability:c1'])
        self.assertFalse(j.calls);self.assertEqual(result['usage']['attempts'],6)

    def test_shared_content_tampering_and_old_release_cannot_dispatch(self):
        r=registry();common,_,_=self.common(r)
        altered=copy.deepcopy(common);altered['initial']['current_request']['claims'][0]['text']='changed'
        with self.assertRaisesRegex(ValueError,'mismatch'):
            run_method(r,'independent',{'actor':FixtureActor(r),'verifier':FixtureJudge()},LIMITS,PROFILES,'tamper',altered)
        with self.assertRaisesRegex(RuntimeError,'online release unavailable'):
            run_suite(r,lambda _:self.fail('no provider construction'),LIMITS,PROFILES,'release',release={'pass':True,'version':'atomic_v3'})


if __name__=='__main__':unittest.main()
