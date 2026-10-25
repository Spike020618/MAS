import json
import random
import unittest

from experiments.core.mechanism.upgrade_game import Contract, digest, freeze_state, improve, upgrade_request
from experiments.core.mechanism.upgrade_protocol_v3 import decode_content, decode_row, majority, request, snapshot
from experiments.core.mechanism.upgrade_coordination import recommend
from experiments.core.mechanism.upgrade_execution_v3 import prepare, finish


def decoded(vote='yes'):
    return decode_content(json.dumps({'A': 'a', 'E': [], 'I': 'i', 'C': vote}), ['yes', 'no'])


class DecodeV3Tests(unittest.TestCase):
    def test_partial_report_keeps_C_without_filling_I(self):
        r = decode_content('{"A":"a","E":[],"C":"yes"}', ['yes', 'no'])
        self.assertTrue(r['conclusion_usable'])
        self.assertFalse(r['aeic_complete'])
        self.assertNotIn('I', r['available_fields'])
        self.assertEqual(majority([r, decoded(), decoded('no')], ['yes', 'no']), 'yes')

    def test_duplicate_or_unknown_C_is_not_guessed(self):
        for content in ('{"C":"yes","C":"no"}', '{"C":"maybe"}', 'not json', '["yes"]'):
            r = decode_content(content, ['yes', 'no'])
            self.assertFalse(r['conclusion_usable'])
            self.assertEqual(r['conclusion'], 'abstain')

    def test_other_field_errors_do_not_erase_C(self):
        for content in ('{"C":"yes","E":123}', '{"C":"yes","A":"a","A":"b"}',
                        '{"C":"yes","extra":"x"}'):
            r = decode_content(content, ['yes', 'no'])
            self.assertTrue(r['conclusion_usable'])
            self.assertFalse(r['aeic_complete'])

    def test_fixed_quorum_and_no_hidden_schema_keys(self):
        invalid = decode_content('{}', ['yes', 'no'])
        self.assertEqual(majority([decoded(), invalid, invalid], ['yes', 'no']), 'abstain')
        forged = decoded(); forged['label'] = 'yes'
        with self.assertRaises(ValueError): snapshot('Q?', ['yes', 'no'], [forged, decoded(), decoded()])

    def test_complete_state_requests_are_exactly_legacy_compatible(self):
        initial = [decoded(), decoded('no'), decoded()]
        frozen = snapshot('Q?', ['yes', 'no'], initial)
        old = freeze_state('Q?', ['yes', 'no'], [r['available_fields'] for r in initial])
        for i in range(3):
            self.assertEqual(request(frozen, i), upgrade_request(old, i))

    def test_partial_initial_state_has_different_request_and_no_fake_reasoning(self):
        partial = decode_content('{"C":"yes"}', ['yes', 'no'])
        frozen = snapshot('Q?', ['yes', 'no'], [partial, decoded(), decoded()])
        public = json.loads(request(frozen, 0)['messages'][1]['content'])
        self.assertEqual(public['state']['reports'][0]['available_fields'], {'C': 'yes'})
        self.assertFalse(public['state']['reports'][0]['aeic_complete'])


class CoordinationV3Tests(unittest.TestCase):
    def test_joint_recommendation_escapes_unanimity_trap_with_consent(self):
        c = Contract((0, 0, 0, 1, 0, 1, 1, 1), ((0, .01),) * 3, 1, 'mathematical-fixture')
        self.assertEqual(improve(c)['actions'], [0, 0, 0])
        result = recommend(c)
        self.assertEqual(sum(result['actions']), 2)
        self.assertTrue(all(a['accept'] for a in result['agent_approvals']))
        self.assertEqual(result['audit']['selected']['regret'], 0)
        self.assertEqual(result['audit']['global_objective_gap'], 0)

    def test_random_tables_budget_global_optimum_and_participation(self):
        rng = random.Random(20261022)
        for _ in range(100):
            c = Contract(tuple(rng.random() for _ in range(8)),
                         tuple((rng.random() / 50, rng.random() / 10) for _ in range(3)), 1, 'synthetic-property-check')
            r = recommend(c)
            self.assertLessEqual(r['audit']['selected']['regret'], 1e-12)
            self.assertLessEqual(sum(r['settlement']['payments']), 1 + 1e-12)
            self.assertTrue(all(a['accept'] and a['utility'] >= a['keep_reference_utility'] - 1e-12 for a in r['agent_approvals']))

    def test_flat_value_selects_no_resources_not_forced_upgrade(self):
        c = Contract((.5,) * 8, ((0, .01),) * 3, 1, 'fixture')
        self.assertEqual(recommend(c)['actions'], [0, 0, 0])

    def test_executable_receipts_and_saved_plan_roundtrip(self):
        c = Contract((0, 0, 0, 1, 0, 1, 1, 1), ((0, .01),) * 3, 1, 'fixture')
        plan = json.loads(json.dumps(prepare('Q?', ['yes', 'no'], [decoded('no')] * 3, c)))
        receipts = []
        for job in plan['jobs']:
            response = {'http_status': 200, 'body': json.dumps({'model': 'offline-mock',
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{"C":"yes"}'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})}
            row = {'request': job['request'], 'request_hash': digest(job['request']),
                   'response': response, 'response_hash': digest(response)}
            receipts.append({'agent': job['agent'], 'row': row})
        output = finish(plan, c, receipts)
        self.assertEqual(output['decision'], 'yes')
        self.assertFalse(output['reports'][plan['jobs'][0]['agent']]['aeic_complete'])
        for bad in (receipts[:-1], receipts + receipts[:1]):
            with self.assertRaises(ValueError): finish(plan, c, bad)
        receipts[0]['row']['request_hash'] = 'bad'
        with self.assertRaises(ValueError): finish(plan, c, receipts)


if __name__ == '__main__': unittest.main()
