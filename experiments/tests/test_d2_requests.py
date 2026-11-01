import json
import unittest
from experiments.paired_deliberation.requests_d2 import initial_request,central_initial_request,review_request,ranking_request
from experiments.paired_deliberation.state_d2 import StateD2
from experiments.paired_deliberation.verification_d2 import VerificationD2
from experiments.consensus_recommendation.common import fingerprint
from experiments.tests.test_d2_contract_state import registry,claim,report


class RequestTests(unittest.TestCase):
    def test_actor_views_identical_and_central_generation_capacity_aligned(self):
        r=registry();requests=[initial_request(r,o) for o in ('current_request','history_applicability','movie_fit')]
        bodies=[json.loads(q['messages'][1]['content']) for q in requests]
        for key in ('evidence','training_prediction_evidence','evidence_registry'):
            self.assertEqual(bodies[0][key],bodies[1][key]);self.assertEqual(bodies[1][key],bodies[2][key])
        central=central_initial_request(r)
        self.assertEqual(central['max_tokens'],sum(q['max_tokens'] for q in requests))
        self.assertEqual(set(json.loads(central['messages'][1]['content'])['schema']['reports']),{'current_request','history_applicability','movie_fit'})
    def test_repair_hash_is_after_observation_and_decision_predictions_remain_soft(self):
        r=registry();s=StateD2({'current_request':report(r,[claim(r),claim(r,'p','hypothesis')])},r)
        target='current_request:c1';before=fingerprint(s.claims[target])
        s.observe('central',target,before,VerificationD2(r))
        payload=json.loads(review_request(r,s,target,'central')['messages'][1]['content'])
        self.assertNotEqual(before,payload['schema']['expected_hash'])
        self.assertEqual(payload['target']['status'],'retracted')
        decision=json.loads(ranking_request(r,s)['messages'][1]['content'])['decision_state']
        self.assertTrue(decision['prediction_basis_claims']);self.assertFalse(decision['source_supported_claims'])
        self.assertFalse(decision['hard_exclusions'])


if __name__=='__main__':unittest.main()
