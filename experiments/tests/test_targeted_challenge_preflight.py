import unittest
from unittest.mock import patch
from experiments.mechanism import review_targeted_challenge_preflight as review


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.case = {'question': 'Which?', 'choices': dict(zip('ABCD', 'wxyz')), 'article': 'source fact'}

    def test_all_paths_full_source(self):
        permissions = {'author': (), 'ordinary0': ('author',), 'ordinary1': ('author', 'ordinary0'),
                       'ordinary2': ('author', 'ordinary0', 'ordinary1'), 'fixed_judge': ('author',),
                       'challenge': ('author',), 'challenge_judge': ('author', 'challenge'),
                       'restatement': ('author',), 'reply': ('author', 'challenge'),
                       'nonadaptive': ('author', 'challenge', 'restatement'),
                       'interactive': ('author', 'challenge', 'reply')}
        for stage in review.STAGES:
            payload = review.public_payload(self.case, 'A', stage, {k: 'text' for k in permissions[stage]}, [])
            self.assertEqual(payload['article'], self.case['article'])

    def test_labels_rejected(self):
        with self.assertRaises(ValueError):
            review.public_payload({**self.case, 'gold_label': 'B'}, 'A', 'author', {}, [])

    def test_cross_arm_history_rejected(self):
        with self.assertRaises(ValueError):
            review.public_payload(self.case, 'A', 'restatement', {'author': 'x', 'challenge': 'y'}, [])

    def test_diagnostic_metadata_rejected(self):
        with self.assertRaises(ValueError):
            review.public_payload(self.case, 'A', 'reply', {'author': {'gold': 'A'}, 'challenge': 'x'}, [])

    def test_fabricated_window_rejected(self):
        with self.assertRaises(ValueError):
            review.public_payload(self.case, 'A', 'author', {}, [{'start': 0, 'end': 6, 'text': 'wrong!'}])

    def test_budget_and_dependency_accounting(self):
        m = review.manifest()
        self.assertEqual(m['proposed_physical_call_limit'], 360)
        self.assertEqual(m['output_token_ceiling'], 360000)
        self.assertEqual(m['selected_path_calls_per_group']['ordinary'], 16)
        self.assertEqual(m['selected_path_calls_per_group']['interactive'], 16)
        self.assertFalse(m['transport_implemented'])
        self.assertIn('NOT_APPROVED', m['status'])

    def test_audit_network_disabled(self):
        with patch('socket.socket.connect', side_effect=AssertionError('offline only')), \
             patch('socket.getaddrinfo', side_effect=AssertionError('offline only')):
            review.audit()


if __name__ == '__main__':
    unittest.main()
