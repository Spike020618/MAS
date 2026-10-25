import unittest
from experiments.mechanism.audit_certified_procurement_control import menu_reply


class ProcurementControlTest(unittest.TestCase):
    def test_any_selected_verified_cost_is_covered_without_gold(self):
        for cost in (1, 200, 340, 2000):
            menu = menu_reply(1, cost)
            self.assertEqual(menu['chosen'], 1)
            self.assertEqual(menu['upgrade_incremental_utility'], 1)
            self.assertEqual(menu['credit_if_upgrade'], cost+1)

    def test_unselected_route_remains_costly_and_is_rejected(self):
        for cost in (1, 200, 340, 2000):
            self.assertEqual(menu_reply(0, cost)['chosen'], 0)

    def test_not_dependent_on_tuning_a_reward_to_question_content(self):
        for fee in (.0001, 1, 10):
            self.assertEqual(menu_reply(1, 340, fee)['chosen'], 1)

    def test_corrupt_receipt_or_nonpositive_compensation_rejected(self):
        for cost in (None, -1, 0, '300'):
            with self.assertRaises(ValueError): menu_reply(1, cost)
        with self.assertRaises(ValueError): menu_reply(1, 300, 0)


if __name__ == '__main__': unittest.main()
