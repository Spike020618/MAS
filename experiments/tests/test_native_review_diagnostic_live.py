import unittest
from experiments.paired_deliberation.native_review_diagnostic_live import policy_commitment,MODELS

class LivePolicyTests(unittest.TestCase):
    def rows(self):
        return [{'task_id':'task'+str(i),'target':{'predicted_a_count':.1,'predicted_b_count':.1,'delta_member_sd':.2}}for i in range(8)]
    def test_policies_have_same_query_permissions_and_call_cap(self):
        rates={m:{'recovery_if_b_observed_more':.8,'safe_retention_if_a_observed_more':.8}for m in MODELS}
        p=policy_commitment(self.rows(),rates,{m:0. for m in MODELS})
        self.assertEqual(len(p['groups']),2)
        for g in p['groups']:
            for actions in g['policies'].values():
                self.assertLessEqual(len(actions),3)
                self.assertTrue(all(a['task_id']in g['task_ids']for a in actions))
            self.assertTrue(g['game_solution']['epsilon_equilibrium'])
        self.assertEqual(p,policy_commitment(self.rows(),rates,{m:0. for m in MODELS}))
    def test_no_effective_recovery_does_not_trigger_review(self):
        rates={m:{'recovery_if_b_observed_more':0.,'safe_retention_if_a_observed_more':1.}for m in MODELS}
        p=policy_commitment(self.rows(),rates,{m:0. for m in MODELS})
        for g in p['groups']:
            self.assertEqual(g['policies']['game'],[])
            self.assertEqual(g['policies']['central'],[])
            self.assertEqual(len(g['policies']['fixed_qwen_plus']),3)
