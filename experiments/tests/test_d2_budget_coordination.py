import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.paired_deliberation.budget_d2 import JointBudget,ChannelClient
from experiments.paired_deliberation.budget_v6 import BudgetExceeded
from experiments.paired_deliberation.coordination_d2 import CoordinatorD2,conflict
from experiments.paired_deliberation.requests_d2 import initial_request
from experiments.paired_deliberation.atomic_evidence import parser_request
from experiments.paired_deliberation.state_d2 import StateD2
from experiments.tests.test_d2_contract_state import registry,claim,report

LIMITS={'max_calls':30,'max_evidence_checks':6,'per_call_prompt_tokens':64000,'per_call_completion_tokens':4096,
        'total_prompt_tokens':30*64000,'total_completion_tokens':30*4096,'max_estimated_cny':10.0}
PROFILES={'actor':{'service_id':'offline-small-v1','allowed_returned_models':['offline-small'],
                  'input_cny_per_million':0.1,'output_cny_per_million':0.2},
          'verifier':{'service_id':'offline-judge-v1','allowed_returned_models':['offline-judge'],
                     'input_cny_per_million':1.0,'output_cny_per_million':2.0}}


class Provider:
    offline_only=True
    def __init__(self,model):self.model,self.calls=model,[]
    def complete(self,slot,request):
        self.calls.append((slot,request))
        return {'report':{},'model':self.model,'usage':{'prompt_tokens':10,'completion_tokens':5},'elapsed_ms':1}


class BudgetTests(unittest.TestCase):
    def test_different_service_models_share_one_cap_and_channel_identity(self):
        r=registry();ledger=JointBudget({**LIMITS,'max_calls':2},PROFILES)
        actor,judge=Provider('offline-small'),Provider('offline-judge')
        ChannelClient(ledger,'actor',actor).complete('initial',initial_request(r,'current_request'))
        ChannelClient(ledger,'verifier',judge).complete('parse',parser_request({'text':'The seeker wants Drama.'},{}))
        with self.assertRaises(BudgetExceeded):ChannelClient(ledger,'actor',actor).complete('extra',initial_request(r,'current_request'))
        summary=ledger.summary();self.assertEqual(summary['attempts'],2)
        self.assertEqual(summary['by_channel']['actor']['attempts'],1);self.assertEqual(summary['by_channel']['verifier']['attempts'],1)
        self.assertEqual(len(actor.calls),1);self.assertTrue(summary['usage_complete'])
        self.assertGreater(summary['estimated_list_price_cny']['total'],0)

    def test_shared_initial_consumes_budget_without_network_and_unknown_failure_is_not_zero(self):
        r=registry();req=initial_request(r,'current_request');ledger=JointBudget(LIMITS,PROFILES)
        provider=Provider('offline-small');client=ChannelClient(ledger,'actor',provider)
        record=provider.complete('fixture',req);provider.calls=[]
        client.account_shared('common',req,record);self.assertFalse(provider.calls)
        class Pending(Provider):
            def complete(self,slot,request):raise RuntimeError('pending; billing unknown')
        with self.assertRaises(RuntimeError):ChannelClient(ledger,'actor',Pending('offline-small')).complete('pending',req)
        summary=ledger.summary();self.assertEqual(summary['attempts'],2)
        self.assertFalse(summary['usage_complete']);self.assertIsNone(summary['prompt_tokens']['total'])
        self.assertIsNone(summary['estimated_list_price_cny']['total']);self.assertTrue(summary['dispatch_halted'])

    def test_currency_ceiling_blocks_before_network(self):
        r=registry();ledger=JointBudget({**LIMITS,'max_estimated_cny':0.0},PROFILES);p=Provider('offline-small')
        with self.assertRaises(BudgetExceeded):ChannelClient(ledger,'actor',p).complete('x',initial_request(r,'current_request'))
        self.assertFalse(p.calls)

    def test_failure_recovers_persisted_usage_and_model_change_halts(self):
        r=registry();req=initial_request(r,'current_request')
        with tempfile.TemporaryDirectory() as directory:
            class Malformed:
                root=directory
                def complete(self,slot,request):raise ValueError('invalid JSON persisted')
            Path(directory,'bad.raw.json').write_text(json.dumps({'raw':{'body':json.dumps({'model':'offline-small','usage':{'prompt_tokens':11,'completion_tokens':6}}),'elapsed_ms':3}}))
            ledger=JointBudget(LIMITS,PROFILES)
            with self.assertRaises(ValueError):ChannelClient(ledger,'actor',Malformed()).complete('bad',req)
            self.assertEqual(ledger.summary()['prompt_tokens']['total'],11)
        profiles=copy.deepcopy(PROFILES);profiles['actor']['allowed_returned_models'].append('offline-small-new')
        ledger=JointBudget(LIMITS,profiles)
        ChannelClient(ledger,'actor',Provider('offline-small')).complete('first',req)
        with self.assertRaisesRegex(RuntimeError,'model changed'):ChannelClient(ledger,'actor',Provider('offline-small-new')).complete('changed',req)
        self.assertTrue(ledger.halted)


class CoordinationTests(unittest.TestCase):
    def test_same_movie_opposite_stances_are_not_a_scope_conflict(self):
        r=registry();a=claim(r,kind='current_preference');b=copy.deepcopy(a);b['scope']['object']='Comedy';b['stance']='oppose'
        self.assertFalse(conflict(a,b));b['scope']['object']='Drama';b['scope']['polarity']=False
        self.assertTrue(conflict(a,b))
        a['claim_kind']=b['claim_kind']='utterance';a['scope']['scope']=b['scope']['scope']='utterance'
        self.assertFalse(conflict(a,b))

    def test_multiple_genres_and_unspecified_times_do_not_imply_contradiction(self):
        r=registry();a=claim(r,text='Movie 1 has catalog genre Drama.');b=claim(r,text='Movie 1 has catalog genre Comedy.')
        self.assertFalse(conflict(a,b)) # A catalog item may have both genres.
        a=claim(r,kind='current_preference');b=copy.deepcopy(a)
        a['scope']['time']=b['scope']['time']='unspecified';b['scope']['polarity']=False
        self.assertFalse(conflict(a,b))

    def test_adaptive_stops_on_irrelevant_claim_fixed_processes_it(self):
        r=registry();c=claim(r);c['candidate_ids']=['6'];reports={'current_request':report(r,[c])}
        state=StateD2(reports,r)
        self.assertIsNone(CoordinatorD2('adaptive',reports).next(state))
        fixed=CoordinatorD2('fixed',reports)
        self.assertEqual(fixed.next(state),'current_request:c1')
        self.assertIsNone(fixed.next(state))


if __name__=='__main__':unittest.main()
