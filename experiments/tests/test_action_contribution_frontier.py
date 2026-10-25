import unittest
from unittest.mock import patch
from experiments.mechanism import audit_action_contribution_frontier as audit


class ActionFrontierTests(unittest.TestCase):
    def test_invalid_correction_not_semantic_evidence(self):
        report = lambda a, valid: {'answer': a, 'valid': valid, 'tokens': 10}
        output = {'one': {'fast': report('invalid', False), 'think': report('A', True)},
                  'two': {'fast': report('B', True), 'think': report('A', True)}}
        r = audit.paired(output, {'one': 'A', 'two': 'A'}, 'fast', 'think')
        self.assertEqual(r['corrected'], 2)
        self.assertEqual(r['both_valid_corrected'], 1)
        self.assertEqual(r['correction_with_invalid_fast'], 1)

    def test_union_not_net_gain(self):
        report = lambda a: {'answer': a, 'valid': True, 'tokens': 10}
        output = {'one': {'fast': report('B'), 'think': report('A')},
                  'two': {'fast': report('A'), 'think': report('B')}}
        r = audit.paired(output, {'one': 'A', 'two': 'A'}, 'fast', 'think')
        self.assertEqual(r['corrected'], r['harmed'])
        self.assertEqual(r['oracle_union_correct_not_deployable'], 2)
        self.assertEqual(r['thinking_correct'], 1)

    def test_offline_audit(self):
        with patch('socket.socket.connect', side_effect=AssertionError('offline')), patch('socket.getaddrinfo', side_effect=AssertionError('offline')):
            result = audit.build()
        self.assertEqual(result['new_api_calls'], 0)
        self.assertEqual(result['paired_strata']['explicit_all']['thinking_correct'], 32)
        self.assertFalse(result['decision']['coordination_necessity_established'])


if __name__ == '__main__': unittest.main()
