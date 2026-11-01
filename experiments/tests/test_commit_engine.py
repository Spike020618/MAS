import copy
import json
import tempfile
import unittest

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.commit_engine import run_method, run_suite, shared_initial
from experiments.tests.test_design_v6 import Provider, LIMITS, evidence, claim


class CommitEngineTests(unittest.TestCase):
    def test_refutation_is_in_repair_prompt_and_survives_invalid_proposal(self):
        class P(Provider):
            def complete(self, slot, request):
                result = super().complete(slot, request)
                if request['kind'] == 'patch_v6':
                    payload = json.loads(request['messages'][1]['content'])
                    if payload['target']['claim']['text'].endswith('2001.'):
                        assert payload['target']['status'] == 'retracted'
                    assert payload['schema']['expected_hash'] == fingerprint(payload['target'])
                    result['report']['replacement'] = claim(text='Movie 1 has catalog year 2000.', stance='refute')
                return result
        p = P()
        result = run_method(evidence(), 'verified_fixed', AttemptClient(p, LIMITS), 'fixed')
        self.assertEqual(len(result['rejected']), 3)
        self.assertIn('current_request:c1', result['state']['retracted_ids'])
        self.assertEqual([e['action'] for e in result['events']], ['observe_retract', 'observe_keep', 'observe_retract'])
        self.assertEqual(result['usage']['attempts'], 7)
        final = json.loads(p.calls[-1][1]['messages'][1]['content'])
        self.assertIn('current_request:c1', final['claim_state']['retracted_ids'])

    def test_central_can_consume_same_committed_observation_api(self):
        p = Provider()
        shared = shared_initial(evidence(), AttemptClient(p, LIMITS), 'common')
        result = run_method(evidence(), 'central', AttemptClient(p, LIMITS), 'central', shared, max_rounds=1)
        self.assertEqual(result['events'][0]['action'], 'observe_retract')
        self.assertTrue(any(req['kind'] == 'patch_v6' for _, req in p.calls))
        self.assertFalse(any(req['kind'] == 'whole_review_v6' for _, req in p.calls))

    def test_repair_transport_failure_occurs_after_observation_commit(self):
        # Test actual ordering via the post-observation prompt. A failed call is
        # still surfaced to the caller, never substituted with a fake response.
        class P(Provider):
            def complete(self, slot, request):
                if request['kind'] == 'patch_v6':
                    target = json.loads(request['messages'][1]['content'])['target']
                    assert target['own_status'] == 'retracted'
                    raise RuntimeError('pending request')
                return super().complete(slot, request)
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            run_method(evidence(), 'verified_fixed', AttemptClient(P(), LIMITS), 'failure')

    def test_natural_verifier_is_isolated_from_supplemental_training_evidence(self):
        class P(Provider):
            def complete(self, slot, request):
                if request['kind'] in {'atomic_interpretation', 'atomic_source_relations'}:
                    self.calls.append((slot, copy.deepcopy(request)))
                    payload = json.loads(request['messages'][1]['content'])
                    assert 'training_prediction_evidence' not in payload
                    if request['kind'] == 'atomic_interpretation':
                        assert 'sources' not in payload
                        obj = {'scope': 'preference_state', 'subject': 'seeker', 'predicate': 'wants',
                               'object': 'Drama', 'polarity': True, 'time': 'current', 'reason': 'Original meaning.'}
                    else:
                        obj = {'relations': [{'source_id': 'message:1', 'relation': 'supports',
                                              'support_quote': 'I want Drama.', 'refute_quote': ''}],
                               'reason': 'Matching target.'}
                    return {'report': obj, 'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                            'model': 'offline-fixture', 'elapsed_ms': 1}
                result = super().complete(slot, request)
                if request['kind'] == 'initial_v6':
                    c = claim(text='The seeker wants Drama.', kind='explicit_preference')
                    c['source_ids'] = ['message:1']
                    result['report']['claims'] = [c]
                return result
        e, p = evidence(), P()
        e['observations'][0]['text'] = 'I want Drama.'
        view = {'kind': 'historical_training_prediction_evidence', 'case_id': e['case_id'],
                'current_case_labels_available': False, 'candidate_rows': [[str(i)] for i in range(1, 7)]}
        result = run_method(e, 'verified_fixed', AttemptClient(p, LIMITS), 'natural', max_rounds=1, training_view=view)
        self.assertEqual(result['events'][0]['action'], 'observe_keep')
        self.assertEqual(result['usage']['attempts'], 7)
        record = next(iter(result['verification_cache'].values()))
        self.assertEqual(record['relation_contract'], 'polarity_witness_v4_candidate')
        self.assertEqual(len([r for _, r in p.calls if r['kind'].startswith('atomic_')]), 2)

    def test_old_semantic_gate_cannot_dispatch_new_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, 'no online release'):
                run_suite(evidence(), lambda method: self.fail('must not create client'), root)

    def test_training_view_attaches_to_all_actors_but_not_first_party_verifier(self):
        e, p = evidence(), Provider()
        view = {'kind': 'historical_training_prediction_evidence', 'case_id': e['case_id'],
                'current_case_labels_available': False, 'candidate_rows': [[str(i)] for i in range(1, 7)]}
        shared = shared_initial(e, AttemptClient(p, LIMITS), 'common', training_view=view)
        for method in ('independent', 'peer', 'central', 'verified_fixed', 'verified_adaptive'):
            run_method(e, method, AttemptClient(p, LIMITS), method, shared, max_rounds=1, training_view=view)
        for _, req in p.calls:
            payload = json.loads(req['messages'][1]['content'])
            self.assertEqual(payload['training_prediction_evidence'], view)
            self.assertNotIn('training_prediction_evidence', payload['evidence'])
        altered = copy.deepcopy(view)
        altered['candidate_rows'].reverse()
        with self.assertRaisesRegex(ValueError, 'order mismatch'):
            run_method(e, 'independent', AttemptClient(p, LIMITS), 'bad', shared, training_view=altered)
        with self.assertRaisesRegex(ValueError, 'different training evidence'):
            run_method(e, 'independent', AttemptClient(p, LIMITS), 'bad', shared)


if __name__ == '__main__': unittest.main()
