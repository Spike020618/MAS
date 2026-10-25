from dataclasses import replace
import inspect
import unittest

from experiments.core.mechanism.budgeted_information_sim import Contract, Decision, Observation, Report
from experiments.mechanism.replay_budgeted_information import fit_table, plan, confidence_cell
from experiments.mechanism.diagnose_budgeted_information import (
    random_same_role_counts, bootstrap_calibration, billing_ratio_sensitivity,
)


def data():
    observations, high, labels = {}, {}, {}
    for j in range(6):
        cid = str(j)
        observations[cid] = [Observation(cid, i, ('a', 'b'), Report('a', (.6, .4)), 100) for i in range(3)]
        for i in range(3): high[(cid, i)] = (Report('b', (.1, .9)), 200, 'source')
        labels[cid] = 'b'
    return observations, high, labels


class BudgetedReplayTest(unittest.TestCase):
    def test_planning_interface_does_not_accept_gold_or_upgraded_reports(self):
        self.assertEqual(set(inspect.signature(plan).parameters), {'observations', 'table', 'contract', 'conservative'})

    def test_calibration_not_heldout_outcome_controls_actions(self):
        observations, high, labels = data()
        table = fit_table(observations, high, labels)
        actions = plan(observations, table, Contract(1000))
        self.assertTrue(all(d.action == 'U' for ds in actions.values() for d in ds))
        # Mutating future resource outputs cannot change an already available
        # private view or historical table; no resource output is an input.
        alternate_high = {k: (Report('a', (.99, .01)), 1999, 'changed') for k in high}
        flipped_test_labels = {cid: 'a' for cid in observations}
        self.assertEqual(actions, plan(observations, table, Contract(1000)))
        self.assertNotEqual(high, alternate_high)
        self.assertNotEqual(labels, flipped_test_labels)

    def test_invalid_high_not_selected_out_of_calibration(self):
        observations, high, labels = data()
        high = {k: (Report('b', None), 200, 'bad-output') for k in high}
        table = fit_table(observations, high, labels)
        for e in table.values():
            self.assertEqual(e.n, 6)
            self.assertLess(e.mean_score_gain, 0.)

    def test_invalid_low_is_an_explicit_unmodeled_fallback(self):
        observations, high, labels = data()
        observations['0'][0] = replace(observations['0'][0], initial=Report('a', (.9999, 0.)))
        self.assertEqual(confidence_cell(observations['0'][0]), 'invalid')
        table = fit_table(observations, high, labels)
        self.assertEqual(table[(0, 'uncertain')].n, 5)
        ds = plan(observations, table, Contract(10000))
        self.assertEqual(ds['0'][0].reason, 'invalid_initial_probability_no_imputation')

    def test_fixed_quotas_random_expectation_counts_calls(self):
        observations, high, labels = data()
        ds = {cid: [Decision('U' if int(cid) < 2 else 'K', 0., 'test') for _ in range(3)] for cid in observations}
        result = random_same_role_counts(observations, high, ds, labels)
        self.assertAlmostEqual(result['expected_high_calls'], 6)
        self.assertAlmostEqual(result['tokens_expectation_including_initial'], 3000)
        # Majority of independent role subsets at p=1/3.
        self.assertAlmostEqual(result['correct_expectation_not_observed_count'], 6*(3*(1/3)**2*(2/3)+(1/3)**3))

    def test_bootstrap_is_deterministic_and_label_free_on_test(self):
        observations, high, labels = data()
        a = bootstrap_calibration(observations, high, labels, observations, 8)
        b = bootstrap_calibration(observations, high, labels, observations, 8)
        self.assertEqual(a, b)
        self.assertEqual(set(inspect.signature(bootstrap_calibration).parameters), {'calibration', 'train_upgrades', 'train_labels', 'test', 'reps'})

    def test_billing_break_even_is_not_token_count_equality(self):
        observations, high, labels = data()
        ds = {cid: [Decision('K', 0., 'test') for _ in range(3)] for cid in observations}
        r = billing_ratio_sensitivity(observations, high, ds)
        self.assertEqual(r['weak_to_strong_unit_price_break_even'], 2.)
        self.assertFalse(r['actual_price_or_billing_claim'])


if __name__ == '__main__': unittest.main()
