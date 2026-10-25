import unittest
from experiments.mechanism.probe_marginal_credit import solve, table_for


class MarginalProbeTests(unittest.TestCase):
    def test_expected_game_potential_and_no_profitable_terminal_deviation(self):
        cases = [{'dataset': 'fixture', 'question_id': str(i), 'ground_truth': 'yes'} for i in range(2)]
        lookup = {}
        for case in cases:
            qid = case['dataset']+'/'+case['question_id']
            for agent in range(3):
                for action in ['D', 'R']:
                    for repeat in range(2):
                        lookup[qid, agent, action, repeat] = {
                            'schema_valid': True, 'finish_reason': 'stop',
                            'report': {'A': 'a', 'E': 'e', 'I': 'i', 'C': 'yes' if action == 'R' else 'no'},
                            'usage': {'total_tokens': 100 if action == 'D' else 200}}
        result = solve(table_for(cases, lookup, .01))
        self.assertLess(result['max_potential_identity_error'], 1e-10)
        self.assertLessEqual(max(result['terminal_regrets']), 1e-10)
        # A low-quality equilibrium can be locally stable: one correct vote
        # cannot rescue a wrong majority. Do not claim global quality optimum.
        self.assertEqual(result['profile'], [0, 0, 0])

    def test_replay_rejects_incomplete_response(self):
        from experiments.mechanism.probe_marginal_credit import normalized
        with self.assertRaises(ValueError): normalized({'schema_valid': True, 'finish_reason': 'length'})


if __name__ == '__main__': unittest.main()
