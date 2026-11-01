import copy
import json
import unittest
from experiments.paired_deliberation.explore_light_actor import check_raw,register
from experiments.paired_deliberation.explore_light_parser import request_v2
from experiments.tests.test_d2_contract_state import registry


class LightInterfaceTests(unittest.TestCase):
    def raw(self):return {'ranking':['1','2','3','4','5'],'claims':[{'text':'The seeker wants Drama.','source_ids':['train:1'],'candidate_ids':['1']}]}
    def test_parser_receives_text_only_and_missing_kind_is_never_guessed(self):
        raw=self.raw();payload=json.loads(request_v2(raw['claims'])['messages'][1]['content'])
        self.assertEqual(set(payload),{'original_claims','schema'})
        self.assertEqual(set(payload['original_claims'][0]),{'claim_id','text'})
        with self.assertRaisesRegex(ValueError,'row fields'):register(raw,{'claims':[{'claim_id':'c1','scope':None}]},registry())
    def test_public_parser_cannot_promote_historical_basis_to_user_fact(self):
        r=registry();raw=self.raw();check_raw(raw,r)
        scope={'subject':'seeker','predicate':'wants','object':'Drama','polarity':'affirmative','time':'current'}
        parsed={'claims':[{'claim_id':'c1','kind':'current_preference','scope':scope}]}
        result=register(raw,parsed,r)[0]
        self.assertIsNone(result['registered']);self.assertIn('authority',result['error'])
        parsed['claims'][0].update(kind='hypothesis',scope=None)
        result=register(raw,parsed,r)[0]
        self.assertIsNotNone(result['registered']);self.assertFalse(result['fact_verified']);self.assertFalse(result['hard_constraint_active'])
    def test_unsupported_original_and_outside_pool_are_retained_not_repaired(self):
        r=registry();raw=self.raw()
        result=register(raw,{'claims':[{'claim_id':'c1','kind':'unsupported','scope':None}]},r)[0]
        self.assertIsNone(result['registered']);self.assertEqual(result['original'],raw['claims'][0])
        raw['claims'][0]['candidate_ids']=['not-in-pool']
        with self.assertRaises(ValueError):check_raw(raw,r)


if __name__=='__main__':unittest.main()
