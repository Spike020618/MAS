from itertools import combinations
import unittest
from experiments.mechanism.audit_observed_evidence_shapley import shapley


class ShapleyTests(unittest.TestCase):
    def values(self,fn):return {frozenset(s):fn(set(s)) for k in range(4) for s in combinations(range(3),k)}
    def test_additive_and_dummy(self):
        values=self.values(lambda s:2+sum([1,-3,0][i] for i in s))
        self.assertEqual(shapley(values),[1.,-3.,0.])
    def test_zero_total_does_not_mean_zero_individual_credit(self):
        values=self.values(lambda s:sum([1,-1,0][i] for i in s))
        for actual,expected in zip(shapley(values),[1.,-1.,0.]):self.assertAlmostEqual(actual,expected)
    def test_missing_rejected(self):
        with self.assertRaises(ValueError):shapley({frozenset():0})


if __name__=='__main__':unittest.main()
