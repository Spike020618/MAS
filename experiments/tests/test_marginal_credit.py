from itertools import product
import unittest
from experiments.core.mechanism.marginal_credit import seal, settle, VerifiedOutcome


def reports(votes):
    return {str(i): {'A': 'a', 'E': ['e'], 'I': 'i', 'C': v} for i, v in enumerate(votes)}


class MarginalCreditTests(unittest.TestCase):
    def settle(self, votes, costs=None, label='yes'):
        rows = reports(votes)
        return settle(rows, costs or {a: 0 for a in rows}, 1, seal(rows), VerifiedOutcome(label, 'test-fixture'))

    def test_exact_potential_and_budget_exhaustive(self):
        actions = ['yes', 'no', 'abstain']
        costs = {'yes': .1, 'no': .2, 'abstain': 0}
        for label in ['yes', 'no']:
            for votes in product(actions, repeat=3):
                own_costs = {str(i): costs[v] for i, v in enumerate(votes)}
                before = self.settle(votes, own_costs, label)
                self.assertLessEqual(before['paid'], 1+1e-12)
                self.assertTrue(all(p >= 0 for p in before['credits'].values()))
                for i in range(3):
                    for action in actions:
                        changed = list(votes); changed[i] = action
                        new_costs = dict(own_costs); new_costs[str(i)] = costs[action]
                        after = self.settle(changed, new_costs, label)
                        self.assertAlmostEqual(after['potential']-before['potential'],
                            after['utilities'][str(i)]-before['utilities'][str(i)])

    def test_pivotal_correct_report_rewarded_and_harm_penalized(self):
        row = self.settle(['yes', 'yes', 'no'])
        self.assertEqual(row['components']['0']['delta'], 1)
        bad = self.settle(['yes', 'yes', 'no', 'abstain'])
        self.assertEqual(bad['components']['2']['delta'], -1)

    def test_redundant_correct_report_has_no_extra_performance_credit(self):
        row = self.settle(['yes', 'yes', 'yes'])
        self.assertEqual(row['components']['0']['delta'], 0)

    def test_no_verified_outcome_means_no_settlement(self):
        rows = reports(['yes', 'no', 'yes'])
        row = settle(rows, {a: 0 for a in rows}, 1, seal(rows))
        self.assertEqual(row['status'], 'pending_verified_outcome')
        self.assertIsNone(row['credits'])

    def test_commitment_and_schema_fail_closed(self):
        rows = reports(['yes', 'no', 'yes']); commitment = seal(rows)
        rows['0']['C'] = 'no'
        with self.assertRaises(ValueError): settle(rows, {a: 0 for a in rows}, 1, commitment)
        rows['0']['C'] = None
        with self.assertRaises(ValueError): seal(rows)

    def test_longer_or_novel_text_does_not_raise_payment(self):
        rows = reports(['yes', 'yes', 'no'])
        first = settle(rows, {a: 0 for a in rows}, 1, seal(rows), VerifiedOutcome('yes', 'fixture'))
        rows['0']['E'] = ['unrelated new words'] * 100
        second = settle(rows, {a: 0 for a in rows}, 1, seal(rows), VerifiedOutcome('yes', 'fixture'))
        self.assertEqual(first['credits'], second['credits'])

    def test_abstentions_cannot_turn_single_vote_into_majority(self):
        row = self.settle(['yes', 'abstain', 'abstain'])
        self.assertIsNone(row['decision'])
        self.assertEqual(row['score'], 0)

    def test_more_agents_budget_bound_and_own_cost_identity(self):
        for votes in product(['yes', 'no', 'abstain'], repeat=4):
            before = self.settle(votes)
            self.assertLessEqual(before['paid'], 1+1e-12)
            for i in range(4):
                changed = list(votes); changed[i] = 'yes'
                after = self.settle(changed)
                self.assertAlmostEqual(after['potential']-before['potential'],
                    after['utilities'][str(i)]-before['utilities'][str(i)])

    def test_private_label_is_not_part_of_report_commitment(self):
        rows = reports(['yes', 'yes', 'no'])
        commitment = seal(rows)
        yes = settle(rows, {a: 0 for a in rows}, 1, commitment, VerifiedOutcome('yes', 'fixture'))
        no = settle(rows, {a: 0 for a in rows}, 1, commitment, VerifiedOutcome('no', 'fixture'))
        self.assertEqual(yes['commitment'], no['commitment'])
        self.assertNotEqual(yes['credits'], no['credits'])

    def test_brier_breaks_threshold_plateau_but_is_not_accuracy(self):
        from experiments.core.mechanism.marginal_credit import team_score
        self.assertGreater(team_score(['yes', 'no', 'no'], 'yes', 'brier'),
                           team_score(['no', 'no', 'no'], 'yes', 'brier'))
        self.assertEqual(team_score(['yes', 'no', 'no'], 'yes', 'accuracy'), 0)

    def test_brier_exact_potential_and_budget(self):
        for votes in product(['yes', 'no', 'abstain'], repeat=3):
            rows = reports(votes)
            before = settle(rows, {a: 0 for a in rows}, 1, seal(rows), VerifiedOutcome('yes', 'fixture'), 'brier')
            self.assertLessEqual(before['paid'], 1+1e-12)
            for i in range(3):
                other = reports(votes); other[str(i)]['C'] = 'yes'
                after = settle(other, {a: 0 for a in other}, 1, seal(other), VerifiedOutcome('yes', 'fixture'), 'brier')
                self.assertAlmostEqual(after['potential']-before['potential'],
                    after['utilities'][str(i)]-before['utilities'][str(i)])


if __name__ == '__main__': unittest.main()
