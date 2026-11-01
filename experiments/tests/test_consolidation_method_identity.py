import unittest
from experiments.audits.method_identity_audit import audit
from experiments.coordination_game_v1.stage3_run import assignments_for
class IdentityTests(unittest.TestCase):
    def test_all_static_labels_share_assignments_on_frozen_cohort(self):
        result=audit()
        self.assertEqual(result['same_objective_cases'],100)
        for row in result['observed_60_pairs'].values():
            self.assertEqual(row['n'],60)
            self.assertEqual(row['same_assignment_set'],60)
            self.assertEqual(row['same_read_request_multiset'],60)
    def test_negative_predictions_still_force_two_reads_in_old_runner(self):
        task={'id':'fixture','candidates':['a','b','c']}
        for method in ('greedy','central','game','cbba'):
            assignments=assignments_for(task,method,0,lambda t,m,c:-1.)
            self.assertEqual(len(assignments),2)
if __name__=='__main__':unittest.main()
