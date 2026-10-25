import json
import unittest

from experiments.core.mechanism.upgrade_decode import parse
from experiments.mechanism.robustness_upgrade_p2 import c_only, majority


class RobustnessTests(unittest.TestCase):
    def test_missing_AEIC_field_does_not_erase_valid_C_in_simple_baseline(self):
        row = {'response': {'http_status': 200, 'body': json.dumps({
            'model': 'mock', 'choices': [{'finish_reason': 'stop', 'message': {'content': '{"C":"yes"}'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})}}
        self.assertEqual(c_only(row, ['yes', 'no']), 'yes')
        self.assertEqual(parse(row, ['yes', 'no'])[0]['C'], 'abstain')

    def test_fixed_majority_preserves_abstentions_in_denominator(self):
        self.assertEqual(majority(['yes', 'abstain', 'abstain'], ['yes', 'no']), 'abstain')
        self.assertEqual(majority(['yes', 'yes', 'abstain'], ['yes', 'no']), 'yes')


if __name__ == '__main__': unittest.main()
