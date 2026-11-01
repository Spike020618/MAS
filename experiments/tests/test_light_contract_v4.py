import copy
import json
import unittest

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.light_contract import single_request, parse_single, register_single
from experiments.paired_deliberation.state_d2 import StateD2
from experiments.paired_deliberation.verification_d2 import VerificationD2
from experiments.tests.test_d2_contract_state import registry, claim, report


class LightContractV4Tests(unittest.TestCase):
    def test_single_prompt_has_no_batch_contract_or_evidence(self):
        req = single_request('The seeker likes Drama.')
        payload = json.loads(req['messages'][1]['content'])
        self.assertEqual(set(payload), {'original_claim', 'schema'})
        self.assertNotIn('one row per original', req['messages'][0]['content'])
        for obj in ({'scope': None}, {'claim_id': None, 'kind': 'unsupported', 'scope': None},
                    {'claims': [{'kind': 'unsupported', 'scope': None}]},
                    {'kind': 'catalog_fact', 'scope': {'subject': 'seeker'}}):
            with self.assertRaises(ValueError): parse_single(obj)

    def test_history_cannot_be_registered_as_user_fact_and_original_is_preserved(self):
        r = registry()
        original = {'text': 'The seeker likes Drama.', 'source_ids': ['train:1'], 'candidate_ids': ['1']}
        parsed = {'kind': 'current_preference', 'scope': {'subject': 'seeker', 'predicate': 'likes',
            'object': 'Drama', 'polarity': 'affirmative', 'time': 'current'}}
        result = register_single(original, parsed, r, 'c2')
        self.assertIsNone(result['registered'])
        self.assertEqual(result['original'], original)
        self.assertIn('authority', result['error'])

    def test_refutation_bad_patch_and_repair_preserve_every_other_row(self):
        r = registry()
        bad = claim(r, text='Movie 1 has catalog year 2001.')
        good = claim(r, cid='c2', text='Movie 1 has catalog year 2000.')
        state = StateD2({'current_request': report(r, [bad, good])}, r)
        verifier = VerificationD2(r)
        other = copy.deepcopy(state.claims['current_request:c2'])
        target = 'current_request:c1'
        state.observe('central', target, fingerprint(state.claims[target]), verifier)
        self.assertIn(target, state.payload()['retracted_ids'])
        proposal = {'target_claim_id': target, 'expected_hash': fingerprint(state.claims[target]),
                    'replacement': copy.deepcopy(state.claims[target]['claim'])}
        proposal['replacement']['candidate_ids'] = ['2']
        with self.assertRaises(ValueError): state.propose('central', target, proposal, verifier)
        self.assertIn(target, state.payload()['retracted_ids'])
        self.assertEqual(state.claims['current_request:c2'], other)
        proposal['replacement'] = copy.deepcopy(state.claims[target]['claim'])
        proposal['replacement']['text'] = 'Movie 1 has catalog year 2000.'
        state.propose('central', target, proposal, verifier)
        self.assertEqual(state.claims[target]['status'], 'source_supported')
        self.assertEqual(state.claims['current_request:c2'], other)


if __name__ == '__main__': unittest.main()
