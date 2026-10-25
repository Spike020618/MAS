import unittest
from experiments.mechanism.pilot_verified_contract import expected_payoffs, settle
from experiments.mechanism.solve_finite_stackelberg import solve


class VerifiedContractPilotTests(unittest.TestCase):
    def test_reward_crosses_best_reply_threshold(self):
        for reward, best in [(1, 'low'), (6, 'verify')]:
            p = expected_payoffs(reward)
            self.assertEqual(max(p, key=lambda a: p[a]['follower']), best)

    def test_only_correctness_is_paid_and_budget_conserves(self):
        correct = settle(1, 1, 'verify', 6)
        wrong = settle(0, 1, 'verify', 6)
        self.assertEqual(correct['wallet_after'], 14)
        self.assertEqual(wrong['wallet_after'], 8)
        self.assertEqual(correct['wallet_after']-correct['wallet_before'],
                         correct['platform_payment']-correct['resource_debit'])

    def test_leader_solution_follows_incentivized_best_response(self):
        solution = solve({'actions': ['low','verify'], 'contracts': [
            {'id':f'r{r}', 'payoffs':expected_payoffs(r)} for r in [1,6]]})
        self.assertEqual(solution['solutions']['optimistic']['contract_id'], 'r6')
        self.assertEqual(solution['solutions']['optimistic']['action'], 'verify')


if __name__ == '__main__':
    unittest.main()
