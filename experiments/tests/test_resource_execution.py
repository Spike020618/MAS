import unittest
from experiments.core.mechanism.resource_execution import request, ResourcePath


class ResourceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.packet = {'question': 'Generic question?', 'sources': [], 'peers': []}
        self.draft = {'A': 'a', 'E': ['e'], 'I': 'i', 'C': 'abstain'}

    def test_low_and_high_have_same_report_messages(self):
        low = request('low_effort', self.packet)
        high = request('aeic_effort', self.packet)
        self.assertEqual(low['messages'], high['messages'])
        self.assertNotEqual(low['backend'], high['backend'])

    def test_fixed_compute_is_exactly_fixed_execution(self):
        paths = {a: ResourcePath('deepseek', 1000) for a in ['low_effort', 'aeic_effort']}
        low = request('low_effort', self.packet, paths)
        high = request('aeic_effort', self.packet, paths)
        self.assertEqual(low, high)

    def test_no_scoring_or_contract_metadata(self):
        for field in ['ground_truth', 'reward', 'reputation', 'condition']:
            with self.assertRaises(ValueError):
                request('aeic_effort', dict(self.packet, **{field: 'forbidden'}))
        packet = dict(self.packet, sources=[{'source_id': 's', 'text': 'data', 'label': 'yes'}])
        with self.assertRaises(ValueError): request('aeic_effort', packet)

    def test_self_check_requires_valid_draft_and_action(self):
        audit = request('verify', self.packet, draft=self.draft)
        self.assertEqual(audit['stage'], 'self_check')
        self.assertIn('not independent external verification', audit['messages'][0]['content'])
        with self.assertRaises(ValueError): request('low_effort', self.packet, draft=self.draft)
        with self.assertRaises(ValueError): request('verify', self.packet, draft={})

    def test_request_cannot_mutate_caller_sources_or_draft(self):
        req = request('verify', self.packet, draft=self.draft)
        self.packet['question'] = 'changed'
        self.draft['C'] = 'yes'
        self.assertNotIn('changed', req['messages'][1]['content'])
        self.assertIn('abstain', req['messages'][1]['content'])

    def test_invalid_path(self):
        with self.assertRaises(ValueError): ResourcePath('deepseek', 0)
        with self.assertRaises(ValueError): ResourcePath('deepseek', True)
        with self.assertRaises(ValueError): request('defer_imitate', self.packet)


if __name__ == '__main__': unittest.main()
