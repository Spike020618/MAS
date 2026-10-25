from pathlib import Path
import unittest
from experiments.mechanism.complete_research_scope import classification, history_check, block_summary, quantile, pair_outcomes


class CloseoutTests(unittest.TestCase):
    def test_scope_excludes_application_and_pauses_alternatives(self):
        self.assertIsNone(classification(Path('movie/summary.json')))
        self.assertIsNone(classification(Path('system_traces/case.json')))
        self.assertEqual(classification(Path('mechanism/stackelberg/run.json')), 'paused_alternative_mechanisms_not_mainline')
        self.assertEqual(classification(Path('review/panel/x.json')), 'historical_review_diagnosis_not_current_QA_method')

    def test_reputation_chain_uses_declared_order_not_file_order(self):
        rows = [
            {'question_id': 'b', 'rounds': [{'round_id': 1, 'agent_id': 0, 'reputation_before': 1, 'reputation_after': 2}]},
            {'question_id': 'a', 'rounds': [{'round_id': 1, 'agent_id': 0, 'reputation_before': 0, 'reputation_after': 1}]}]
        self.assertTrue(history_check(rows, ['a', 'b'])['prefix_chain_matches'])
        self.assertFalse(history_check(rows, ['b', 'a'])['prefix_chain_matches'])
        with self.assertRaises(ValueError): history_check(rows, ['a'])

    def test_constant_block_bootstrap_and_repeatability(self):
        result = block_summary([[1]*10, [1]*10], 5, 20)
        self.assertEqual(result['pointwise_percentile_interval_95'], [1, 1])
        self.assertEqual(result, block_summary([[1]*10, [1]*10], 5, 20))
        self.assertEqual(quantile([0, 1], .25), .25)

    def test_invalid_block_length_fails_instead_of_looping(self):
        for length in [0, -1, 4, True]:
            with self.assertRaises(ValueError): block_summary([[0, 1, 0]], length, 10)

    def test_filtered_pair_score_retains_both_harm_and_benefit(self):
        result = pair_outcomes([(False, True)]*3+[(True, False)])
        self.assertEqual(result['wrong_to_correct'], 3)
        self.assertEqual(result['correct_to_wrong'], 1)
        self.assertEqual(result['delta'], .5)
        self.assertEqual(result['n_pairs'], 4)


if __name__ == '__main__': unittest.main()
