from itertools import permutations
import json
import random
import unittest

from experiments.core.mechanism.upgrade_game import Contract, improve, freeze_state, upgrade_request, aggregate
from experiments.mechanism.audit_upgrade_game import audit


class UpgradeGameTests(unittest.TestCase):
    def test_random_tables_exhaustive_identity_budget_and_all_starts(self):
        rng = random.Random(20261021)
        for _ in range(30):
            c = Contract(tuple(rng.random() for _ in range(8)),
                         tuple((0, rng.random() / 10) for _ in range(3)), 1, 'test-fixture')
            for a in c.profiles():
                before = c.settlement(a)
                self.assertLessEqual(sum(before['payments']), c.budget + 1e-12)
                self.assertTrue(all(0 <= p <= c.budget / c.n for p in before['payments']))
                for i in range(3):
                    b = list(a); b[i] = 1 - b[i]
                    after = c.settlement(b)
                    self.assertAlmostEqual(after['utilities'][i] - before['utilities'][i],
                                           after['potential'] - before['potential'])
                for order in permutations(range(3)):
                    result = improve(c, a, order)
                    check = audit(c, result['actions'])
                    self.assertLessEqual(check['selected']['regret'], 1e-12)
                    self.assertLessEqual(len(result['trace']), 7)
                    self.assertTrue(all(t['potential_after'] > t['potential_before'] for t in result['trace']))

    def test_ties_stay_and_local_is_not_global(self):
        c = Contract((0, 0, 0, 0, 0, 0, 0, 1), ((0, .01),) * 3, 1, 'fixture')
        self.assertEqual(improve(c)['actions'], [0, 0, 0])
        self.assertGreater(audit(c, (0, 0, 0))['global_objective_gap'], 0)
        ties = Contract((.5,) * 8, ((0, 0),) * 3, 1, 'fixture')
        self.assertEqual(improve(ties, (1, 0, 1))['trace'], [])

    def test_true_regret_bound_exhaustive(self):
        proxy = Contract((.2, .4, .3, .7, .3, .5, .6, .9), ((0, .02),) * 3, 1, 'fixture')
        a = improve(proxy)['actions']
        epsilon = .08
        for sign in proxy.profiles():
            # Check representative perturbations; theorem is algebraic, not inferred from tests.
            values = tuple(min(1, max(0, v + (epsilon if sign[j % 3] else -epsilon))) for j, v in enumerate(proxy.values))
            true = Contract(values, proxy.costs, proxy.budget, 'fixture-true')
            self.assertLessEqual(audit(true, a)['selected']['regret'], 2 * proxy.beta * epsilon + 1e-12)

    def test_state_immutable_and_no_action_or_label_input(self):
        reports = [{'A': 'a', 'E': [], 'I': 'i', 'C': c} for c in ['yes', 'no', 'abstain']]
        frozen = freeze_state('question?', ['yes', 'no'], reports)
        before = upgrade_request(frozen, 1)
        reports[0]['C'] = 'no'
        self.assertEqual(before, upgrade_request(frozen, 1))
        state = json.loads(frozen); state['label'] = 'yes'
        with self.assertRaises(ValueError): upgrade_request(json.dumps(state), 1)
        with self.assertRaises(ValueError): upgrade_request(frozen, -1)
        self.assertEqual(aggregate(reports, ['yes', 'no']), 'no')
        self.assertEqual(aggregate(json.loads(frozen)['reports'], ['yes', 'no']), 'abstain')

    def test_invalid_contracts_fail_closed(self):
        for values in [(0,) * 7, (float('nan'),) * 8, (1.1,) * 8]:
            with self.assertRaises(ValueError): Contract(values, ((0, .1),) * 3, 1, 'fixture')
        with self.assertRaises(ValueError): Contract((0,) * 8, ((0, -.1),) * 3, 1, 'fixture')
        c = Contract([0] * 8, [[0, 0]] * 3, 1, 'fixture')
        with self.assertRaises(ValueError): improve(c, order=[0, 0, 1])
        with self.assertRaises(ValueError): improve(c, tolerance=-1)


if __name__ == '__main__':
    unittest.main()
