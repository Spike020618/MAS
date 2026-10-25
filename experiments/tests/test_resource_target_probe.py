from itertools import product
import unittest
from experiments.mechanism.probe_resource_target import metrics, greedy_cover, central_cover, decision_diagnosis


class ResourceTargetTests(unittest.TestCase):
    def test_complete_source_owner_equals_simple_allocation(self):
        agents = ['0', '1', '2']
        bundles = {'0': ['a'], '1': ['b'], '2': ['a', 'b']}
        weights = {'a': .5, 'b': .5}
        costs = {a: [0, .75] for a in agents}
        self.assertEqual(greedy_cover(agents, bundles, weights, costs, .75), [0, 0, 1])
        self.assertEqual(central_cover(agents, bundles, weights, costs, .75)[0], (0, 0, 1))

    def test_duplicate_cost_is_not_unique_coverage(self):
        m = metrics([1, 1], ['0', '1'], {'0': ['a'], '1': ['a']}, {'a': 1}, {'0': [0, 1], '1': [0, 1]})
        self.assertEqual(m['duplicate_acquisitions'], 1)
        self.assertEqual(m['coverage_weight'], 1)
        self.assertEqual(m['modeled_cost'], 2)

    def test_central_optimum_budget_feasibility(self):
        agents = ['a', 'b', 'c']
        bundles = {'a': ['x'], 'b': ['y'], 'c': ['x', 'y']}
        weights = {'x': .3, 'y': .7}
        costs = {'a': [0, .2], 'b': [0, .4], 'c': [0, 1]}
        for cap in [0, .2, .4, .6, 1]:
            p, m = central_cover(agents, bundles, weights, costs, cap)
            self.assertLessEqual(m['modeled_cost'], cap + 1e-10)
            for other in product((0, 1), repeat=3):
                alt = metrics(other, agents, bundles, weights, costs)
                if alt['modeled_cost'] <= cap + 1e-10:
                    self.assertLessEqual(alt['coverage_weight'], m['coverage_weight'] + 1e-10)

    def test_correct_minority_is_only_oracle_headroom(self):
        result = decision_diagnosis([{'stance': v} for v in ['no', 'no', 'yes']], 'yes')
        self.assertTrue(result['correct_answer_not_selected'])
        self.assertFalse(result['majority_correct'])
        self.assertFalse(result['no_correct_agent'])

    def test_no_correct_agent_not_aggregation_recoverable(self):
        result = decision_diagnosis([{'stance': 'no'}]*3, 'yes')
        self.assertTrue(result['unanimously_wrong'])
        self.assertFalse(result['correct_answer_not_selected'])


if __name__ == '__main__': unittest.main()
