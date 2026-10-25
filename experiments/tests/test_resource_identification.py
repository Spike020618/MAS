import unittest
from experiments.mechanism.audit_resource_identification import execution_messages, proxy_counterexamples, exact_p


class IdentificationTests(unittest.TestCase):
    def test_fixed_compute_not_same_prompt(self):
        low = execution_messages('low_effort', fixed=True)[0]
        high = execution_messages('aeic_effort', fixed=True)[0]
        self.assertEqual(low['max_tokens'], high['max_tokens'])
        self.assertNotEqual(low['messages'], high['messages'])

    def test_matched_low_resource_has_different_instruction(self):
        a = execution_messages('low_effort')[0]
        b = execution_messages('low_effort', matched=True)[0]
        self.assertNotEqual(a['messages'], b['messages'])

    def test_matched_independent_and_verify_same_report_sequence(self):
        for action in ['aeic_effort', 'verify']:
            a = execution_messages(action)
            b = execution_messages(action, matched=True)
            self.assertEqual([x['messages'] for x in a], [x['messages'] for x in b])

    def test_proxy_counterexamples_not_truth(self):
        result = proxy_counterexamples()
        self.assertEqual(result['wrong_complete_report_credit'], .4)
        self.assertEqual(result['irrelevant_unseen_token_novelty'], 1)
        self.assertTrue(result['flip_without_external_text_flagged'])
        self.assertFalse(result['same_flip_with_irrelevant_nonempty_text_flagged'])

    def test_exact_pair_test(self):
        self.assertEqual(exact_p(3, 0), .25)
        self.assertEqual(exact_p(0, 0), 1)


if __name__ == '__main__': unittest.main()
