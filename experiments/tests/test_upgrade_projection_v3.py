from itertools import product
import unittest

from experiments.core.mechanism.upgrade_game import aggregate
from experiments.core.mechanism.upgrade_value import state_key, PROFILES
from experiments.core.mechanism.upgrade_value_v3 import invariant_to_baseline, fit


def report(c):
    return {'A': '', 'E': [], 'I': '', 'C': c}


class ProjectionTests(unittest.TestCase):
    def test_projected_equalities_hold_for_all_three_agent_replacements(self):
        choices = ['A', 'B', 'C']
        votes = choices + ['abstain']
        for initial in product(votes, repeat=3):
            state = state_key([report(c) for c in initial])
            before = aggregate([report(c) for c in initial], choices)
            for profile in PROFILES:
                if not invariant_to_baseline(state, profile):
                    continue
                for changed in product(votes, repeat=sum(profile)):
                    values = iter(changed)
                    after = [report(next(values) if a else c) for c, a in zip(initial, profile)]
                    self.assertEqual(aggregate(after, choices), before)

    def test_shrinkage_cannot_create_unanimous_singleton_gain(self):
        rows = [{'id': '1', 'state': '000', 'scores': [0, 0, 0, 1, 0, 1, 1, 1]},
                {'id': '2', 'state': '001', 'scores': [0, 1, 1, 1, 1, 1, 1, 1]}]
        model = fit(rows)
        values = model['states']['000']['values']
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[0], values[2])
        self.assertEqual(values[0], values[4])
        self.assertGreater(values[7], values[0])


if __name__ == '__main__': unittest.main()
