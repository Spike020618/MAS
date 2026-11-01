from itertools import product
import unittest
from experiments.mechanism.audit_counterfactual_credit import statistics


class CreditAuditTests(unittest.TestCase):
    def test_counterexample_prevents_false_variance_claim(self):
        table = {(0,): 0., (1,): 1.}; policies = [[.1, .9]]
        plain = statistics(table, policies, 0, 'none')
        coma = statistics(table, policies, 0, 'coma')
        weighted = statistics(table, policies, 0, 'score_weighted')
        self.assertAlmostEqual(plain['trace_variance'], .0018)
        self.assertAlmostEqual(coma['trace_variance'], .1152)
        self.assertLess(weighted['trace_variance'], 1e-20)

    def test_expected_gradient_and_optimal_variance_multiple_games(self):
        policies = [[.2, .3, .5], [.7, .3]]
        for seed in range(20):
            table = {p: ((p[0] * 13 + p[1] * 7 + seed * (p[0] + 1)) % 17) / 17 for p in product(range(3), range(2))}
            for player in range(2):
                values = [statistics(table, policies, player, m) for m in ('none', 'coma', 'score_weighted')]
                for value in values:
                    for a, b in zip(value['expected_gradient'], values[0]['expected_gradient']): self.assertAlmostEqual(a, b, places=12)
                self.assertLessEqual(values[2]['trace_variance'], min(v['trace_variance'] for v in values[:2]) + 1e-12)

    def test_missing_cells_rejected(self):
        with self.assertRaises(ValueError): statistics({(0,): 1}, [[.5, .5]], 0, 'coma')


if __name__ == '__main__': unittest.main()
