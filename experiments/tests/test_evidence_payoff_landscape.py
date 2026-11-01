import unittest
from experiments.mechanism.audit_evidence_payoff_landscape import pure_best_response_profiles


class LandscapeTests(unittest.TestCase):
    def test_plateau_has_equilibria_without_improvement(self):
        table = {(i, j): .625 if i != j or i == 1 else .5 for i in range(3) for j in range(3)}
        self.assertEqual(len(pure_best_response_profiles(table)), 7)
        self.assertIn((1, 1), pure_best_response_profiles(table))

    def test_missing_joint_action_fails(self):
        with self.assertRaises(ValueError): pure_best_response_profiles({(0, 0): 1})


if __name__ == '__main__': unittest.main()
