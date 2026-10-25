import json
import unittest
from experiments.mechanism.premise_review import request, decode, paired_summary


class PremiseReviewTests(unittest.TestCase):
    def test_equal_information_and_resource_cap(self):
        args = ('A question', ['yes', 'no'], [{'answer': 'yes', 'reasoning': 'unverified'}])
        a = request(*args, 'ordinary', 'locked-model')
        b = request(*args, 'premise', 'locked-model')
        self.assertEqual(a['messages'][1], b['messages'][1])
        for k in ('model', 'temperature', 'max_tokens', 'response_format'):
            self.assertEqual(a[k], b[k])
        self.assertNotEqual(a['messages'][0], b['messages'][0])

    def test_label_extra_field_rejected(self):
        with self.assertRaises(ValueError):
            request('Q', ['yes', 'no'], [{'answer': 'yes', 'reasoning': 'x', 'gold': 'yes'}], 'premise', 'm')

    def test_no_reasoning_word_override(self):
        self.assertEqual(decode('{"answer":"yes","reasoning":"No evidence against it"}', ['yes', 'no'])['answer'], 'yes')
        self.assertFalse(decode('{"answer":"yes"', ['yes', 'no'])['valid'])

    def test_pair_and_shared_costs(self):
        reviews = {a: {'raw': json.dumps({'answer': v, 'reasoning': 'x'}), 'tokens': 30, 'calls': 1}
                   for a, v in [('ordinary', 'yes'), ('premise', 'no')]}
        row = {'id': 'x', 'gold': 'yes', 'choices': ['yes', 'no'], 'initial_answer': 'no',
               'initial_tokens': 60, 'initial_calls': 3, 'reviews': reviews}
        r = paired_summary([row])
        self.assertEqual(r['counts']['premise_vs_ordinary_harms'], 1)
        self.assertEqual(r['total_tokens_with_initial'], {'ordinary': 90, 'premise': 90})
        self.assertEqual(r['total_calls_with_initial']['ordinary'], 4)
        with self.assertRaises(ValueError):
            paired_summary([row, row])

    def test_missing_arm_rejected(self):
        with self.assertRaises(ValueError):
            paired_summary([{'id': 'x', 'reviews': {}}])


if __name__ == '__main__':
    unittest.main()
