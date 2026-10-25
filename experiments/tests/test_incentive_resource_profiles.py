import unittest
from experiments.mechanism.audit_incentive_resource_profiles import no_pay_edges


class ResourceProfileAuditTest(unittest.TestCase):
    def test_edges_and_pivotal_cost_conflict(self):
        rows=no_pay_edges([['no']*3,['yes']*3],[10,20,30],['yes','no'],'yes')
        self.assertEqual(len(rows),12)
        self.assertTrue(all(r['profit_tokens_if_agent_bears_cost']>0 for r in rows))
        self.assertTrue(any(r['before_correct']==1 and r['after_correct']==0 for r in rows))

    def test_no_artificial_harm_if_same_answers(self):
        rows=no_pay_edges([['yes']*3,['yes']*3],[10,20,30],['yes','no'],'yes')
        self.assertFalse(any(r['before_correct']>r['after_correct'] for r in rows))

    def test_costs_must_be_positive(self):
        with self.assertRaises(ValueError):no_pay_edges([['yes']*3,['no']*3],[0,1,1],['yes','no'],'yes')
