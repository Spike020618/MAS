import unittest

from experiments.core.mechanism.upgrade_game import Contract, aggregate, improve
from experiments.core.mechanism.upgrade_value import outcomes
from experiments.mechanism.audit_upgrade_game import audit


def report(vote):
    return {'A': '', 'E': [], 'I': '', 'C': vote}


class CoordinationBarrierTests(unittest.TestCase):
    def test_single_upgrade_cannot_change_unanimous_three_agent_majority(self):
        for choices in (['yes', 'no'], ['option_0', 'option_1', 'option_2']):
            for initial in choices + ['abstain']:
                for replacement in choices + ['abstain']:
                    for agent in range(3):
                        reports = [report(initial) for _ in range(3)]
                        reports[agent] = report(replacement)
                        self.assertEqual(aggregate(reports, choices), initial)

    def test_zero_regret_can_coexist_with_valuable_resource_action(self):
        # This is a mathematical counterexample, not fabricated experimental data.
        table = outcomes([report('no')] * 3, [report('yes')] * 3, ['yes', 'no'], 'yes')
        c = Contract(tuple(table), ((0, .01),) * 3, 1, 'mathematical-counterexample-only')
        selected = improve(c)
        self.assertEqual(selected['actions'], [0, 0, 0])
        checked = audit(c, selected['actions'])
        self.assertEqual(checked['selected']['regret'], 0)
        self.assertTrue(all(g < 0 for g in checked['selected']['gains']))
        self.assertGreater(checked['global_objective_gap'], 0)
        self.assertEqual(table[0], 0)
        self.assertEqual(table[-1], 1)


if __name__ == '__main__': unittest.main()
